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

import argparse
import os
import shutil
import sys
import shutil
import cv2
from time import strftime
from yacs.config import CfgNode as CN
import mindspore as ms
from mindspore import nn, context, CheckpointConfig
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
from train_expnet import init_path


def posevae_trainer(net, optimizer_G, optimizer_D, cfg):
    is_train_finetune = False

    generator, discriminator = (
        net,
        net.netD_motion,
    )
    generator_w_loss = GWithLossCell(generator, discriminator, cfg)
    discriminator_w_loss = DWithLossCell(discriminator)
    generator_t_step = GTrainOneStepCell(generator_w_loss, optimizer_G)
    discriminator_t_step = DTrainOneStepCell(discriminator_w_loss, optimizer_D)

    trainer = VAEGTrainer(
        generator_t_step, discriminator_t_step, cfg, is_train_finetune
    )
    return trainer


def train(args):
    # context.set_context(mode=context.PYNATIVE_MODE,
    #                     device_target="Ascend", device_id=6)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=6)

    current_root_path = os.path.split(sys.argv[0])[0]
    sadtalker_paths = init_path(
        args.checkpoint_dir, os.path.join(current_root_path, "config"), args.preprocess
    )

    # create train data loader
    batch_size = 1
    train_data_path = "./data_train/images.txt"
    dataset = TrainPVAEDataset(train_data_path)

    dataset_column_names = ["data"]
    ds = ms.dataset.GeneratorDataset(
        dataset, column_names=dataset_column_names, shuffle=True
    )
    loader_train = ds.batch(
        batch_size,
        drop_remainder=True,
    )

    # create model
    fcfg_pvae = open(sadtalker_paths["audio2pose_yaml_path"])
    cfg_pvae = CN.load_cfg(fcfg_pvae)
    cfg_pvae.freeze()

    net_audio2pose = Audio2Pose(cfg_pvae)
    net_audio2pose.set_train(True)
    auto_mixed_precision(net_audio2pose, "O3")

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
    num_epochs = 2
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

    # define callbacks
    # save_checkpoint_steps = cfg.train_params.save_epoch_frq * loader_train.get_dataset_size()
    # config_ck = CheckpointConfig(
    #     save_checkpoint_steps=save_checkpoint_steps, keep_checkpoint_max=cfg.train_params.keep_checkpoint_max
    # )
    # summary_writer = None
    # if rank_id == 0:
    #     summary_writer = SummaryWriter(os.path.join(cfg.train_params.ckpt_save_dir, "summary"))
    # callbacks = [ms.TimeMonitor()]
    # if cfg.train_params.need_val:
    #     callbacks.append(EvalAsInValPyCallBack(cfg, net, eval_network, summary_writer=summary_writer))
    # if profile:
    #     callbacks.append(ProfileCallback(**cfg.train_params.profile.cfg_dict))
    # if rank_id == 0:
    #     callbacks.append(
    #         TrainingMonitor(
    #             cfg.train_params.epoch_size,
    #             cfg.steps_per_epoch,
    #             print_frequency=cfg.train_params.print_frequency,
    #             summary_writer=summary_writer,
    #         )
    #     )
    #     callbacks.append(
    #         ms.ModelCheckpoint(
    #             prefix=cfg.model.name + "_" + cfg.dataset.dataset_name,
    #             directory=cfg.train_params.ckpt_save_dir,
    #             config=config_ck,
    #         )
    #     )
    eval_cb = EvalSaveCallback(net_audio2pose)

    # define trainer
    trainer = posevae_trainer(net_audio2pose, optimizer_G, optimizer_D, cfg_pvae)
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
    parser = ArgumentParser()
    parser.add_argument(
        "--driven_audio",
        default="./examples/driven_audio/bus_chinese.wav",
        help="path to driven audio",
    )
    parser.add_argument(
        "--source_image",
        default="./examples/source_image/full_body_1.png",
        help="path to source image",
    )
    parser.add_argument(
        "--ref_eyeblink",
        default=None,
        help="path to reference video providing eye blinking",
    )
    parser.add_argument(
        "--ref_pose", default=None, help="path to reference video providing pose"
    )
    parser.add_argument(
        "--checkpoint_dir", default="./checkpoints", help="path to output"
    )
    parser.add_argument("--result_dir", default="./results", help="path to output")
    parser.add_argument(
        "--pose_style", type=int, default=0, help="input pose style from [0, 46)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="the batch size of facerender"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="the image size of the facerender"
    )
    parser.add_argument(
        "--expression_scale",
        type=float,
        default=1.0,
        help="the batch size of facerender",
    )
    parser.add_argument(
        "--input_yaw",
        nargs="+",
        type=int,
        default=None,
        help="the input yaw degree of the user ",
    )
    parser.add_argument(
        "--input_pitch",
        nargs="+",
        type=int,
        default=None,
        help="the input pitch degree of the user",
    )
    parser.add_argument(
        "--input_roll",
        nargs="+",
        type=int,
        default=None,
        help="the input roll degree of the user",
    )
    parser.add_argument(
        "--enhancer",
        type=str,
        default=None,
        help="Face enhancer, [gfpgan, RestoreFormer]",
    )
    parser.add_argument(
        "--background_enhancer",
        type=str,
        default=None,
        help="background enhancer, [realesrgan]",
    )
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.add_argument(
        "--face3dvis", action="store_true", help="generate 3d face and 3d landmarks"
    )
    parser.add_argument(
        "--still",
        action="store_true",
        help="can crop back to the original videos for the full body aniamtion",
    )
    parser.add_argument(
        "--preprocess",
        default="crop",
        choices=["crop", "extcrop", "resize", "full", "extfull"],
        help="how to preprocess the images",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="saving the intermedia output or not"
    )

    # net structure and parameters
    parser.add_argument(
        "--net_recon",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
        help="useless",
    )
    parser.add_argument("--init_path", type=str, default=None, help="Useless")
    parser.add_argument(
        "--use_last_fc", default=False, help="zero initialize the last fc"
    )
    parser.add_argument("--bfm_folder", type=str, default="./checkpoints/BFM_Fitting/")
    parser.add_argument(
        "--bfm_model", type=str, default="BFM_model_front.mat", help="bfm model"
    )

    # default renderer parameters
    parser.add_argument("--focal", type=float, default=1015.0)
    parser.add_argument("--center", type=float, default=112.0)
    parser.add_argument("--camera_d", type=float, default=10.0)
    parser.add_argument("--z_near", type=float, default=5.0)
    parser.add_argument("--z_far", type=float, default=15.0)

    args = parser.parse_args()

    train(args)
