# Copyright © 2023 Huawei Technologies Co, Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from addict import Dict
from utils.arg_parser import parse_args_and_config

import mindspore as ms
from mindspore import context, nn
from mindspore.amp import auto_mixed_precision

import mindspore as ms
from mindspore import nn, context
from mindspore.amp import auto_mixed_precision

from datasets.dataset_pvae import TrainPVAEDataset
from models.audio2pose.audio2pose import Audio2Pose
from argparse import ArgumentParser
from utils.callbacks import EvalSaveCallback
from models.audio2pose.trainer import (
    GWithLossCell,
    DWithLossCell,
    GTrainOneStepCell,
    DTrainOneStepCell,
    VAEGTrainer,
)


def posevae_trainer(net, optimizer_G, optimizer_D):
    is_train_finetune = False

    generator, discriminator = (
        net,
        net.netD_motion,
    )
    generator_w_loss = GWithLossCell(generator, discriminator)
    discriminator_w_loss = DWithLossCell(discriminator)
    generator_t_step = GTrainOneStepCell(generator_w_loss, optimizer_G)
    discriminator_t_step = DTrainOneStepCell(discriminator_w_loss, optimizer_D)

    trainer = VAEGTrainer(generator_t_step, discriminator_t_step, is_train_finetune)
    return trainer


def train(args, config):
    context.set_context(
        mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id
    )

    # init model
    net_audio2pose = Audio2Pose(config.audio2pose)
    net_audio2pose.set_train(True)
    amp_level = config.system.get("amp_level", "O0")
    auto_mixed_precision(net_audio2pose, amp_level)

    # create train data loader
    batch_size = 2
    train_data_path = "./data_train/images.txt"
    dataset = TrainPVAEDataset(train_data_path)

    dataset_column_names = dataset.get_output_columns()
    ds = ms.dataset.GeneratorDataset(
        dataset, column_names=dataset_column_names, shuffle=True
    )
    loader_train = ds.batch(
        batch_size=batch_size,
        drop_remainder=True,
    )

    # cfg_pvae.steps_per_epoch = loader_train.get_dataset_size()

    # create loss
    # loss = create_loss(loss_name=cfg.loss.name, **cfg.loss.cfg_dict)

    # create learning rate schedule
    # optimizer_params, learning_rate = create_scheduler_by_name(model_name=cfg.model.name, cfg=cfg)

    # create optimizer
    # get loss scales
    loss_scale_manager = nn.FixedLossScaleUpdateCell(128)

    # lr scheduler
    min_lr = 0.0
    max_lr = 0.0005
    num_epochs = 10
    decay_epoch = 2
    total_step = loader_train.get_dataset_size() * num_epochs
    step_per_epoch = loader_train.get_dataset_size()
    lr_scheduler = nn.cosine_decay_lr(
        min_lr, max_lr, total_step, step_per_epoch, decay_epoch
    )

    # build optimizer
    D_params = list(net_audio2pose.netD_motion.trainable_params())
    D_optimizer_params = [{"params": D_params, "lr": lr_scheduler}]

    G_params = list(net_audio2pose.netG.trainable_params())
    G_optimizer_params = [{"params": G_params, "lr": lr_scheduler}]

    optimizer_G = nn.Adam(G_optimizer_params, learning_rate=max_lr)
    optimizer_D = nn.Adam(D_optimizer_params, learning_rate=max_lr)

    eval_cb = EvalSaveCallback(net_audio2pose)

    # define trainer
    trainer = posevae_trainer(net_audio2pose, optimizer_G, optimizer_D)
    initial_epoch = 0
    print(" training...")
    trainer.train(
        num_epochs * loader_train.get_dataset_size(),
        loader_train,
        callbacks=[eval_cb],
        dataset_sink_mode=False,
        initial_epoch=initial_epoch,
    )


if __name__ == "__main__":
    args, cfg = parse_args_and_config()
    config = Dict(cfg)
    train(args, config)
