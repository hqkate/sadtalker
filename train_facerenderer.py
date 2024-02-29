import os
from time import strftime
from addict import Dict
from utils.arg_parser import parse_args_and_config

import mindspore as ms
from mindspore import context, nn
from mindspore.amp import auto_mixed_precision

from datasets.dataset_facerender import TrainFaceRenderDataset
from models.facerender.trainer import FaceRenderTrainer
from models.facerender.animate import AnimateModel
from models.facerender.networks import NLayerDiscriminator
from models.facerender.vgg19_extractor import get_feature_extractor
from utils.callbacks import EvalSaveCallback
from utils.preprocess import CropAndExtract
from models.facerender.trainer import (
    GWithLossCell,
    DWithLossCell,
    GTrainOneStepCell,
    DTrainOneStepCell,
)


def facerender_trainer(
    generator, discriminator, vgg_feat_extractor, optimizer_G, optimizer_D, cfg
):
    is_train_finetune = False

    generator_w_loss = GWithLossCell(generator, discriminator, vgg_feat_extractor, cfg)
    discriminator_w_loss = DWithLossCell(discriminator)
    generator_t_step = GTrainOneStepCell(generator_w_loss, optimizer_G)
    discriminator_t_step = DTrainOneStepCell(discriminator_w_loss, optimizer_D)

    trainer = FaceRenderTrainer(
        generator_t_step, discriminator_t_step, cfg, is_train_finetune
    )
    return trainer


def train(args, config):
    context.set_context(
        mode=context.GRAPH_MODE,
        pynative_synchronize=True,
        device_target="CPU",
        device_id=args.device_id
    )

    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)

    # init model
    animate_model = AnimateModel(config.facerender)
    feat_extractor = CropAndExtract(config.preprocess)

    # amp level
    amp_level = config.system.get("amp_level", "O0")
    auto_mixed_precision(animate_model, amp_level)

    # dataset
    dataset = TrainFaceRenderDataset(
        args=args,
        config=config,
        train_list=args.train_list,  # (img_folder, first_coeff_path, net_coeff_path)
        batch_size=args.batch_size,
        expression_scale=args.expression_scale,
        still_mode=args.still,
        preprocess=args.preprocess,
        size=args.size,
        semantic_radius=13,
        syncnet_T=5,
        extractor=feat_extractor,
    )

    dataset_column_names = dataset.get_output_columns()
    ds = ms.dataset.GeneratorDataset(
        dataset, column_names=dataset_column_names, shuffle=True
    )

    dataloader = ds.batch(args.batch_size, drop_remainder=True)

    # lr scheduler
    min_lr = 0.0
    max_lr = 0.0005
    num_epochs = 2
    decay_epoch = 2
    total_step = dataloader.get_dataset_size() * num_epochs
    step_per_epoch = dataloader.get_dataset_size()
    lr_scheduler = nn.cosine_decay_lr(
        min_lr, max_lr, total_step, step_per_epoch, decay_epoch
    )

    # build optimizer
    discriminator = NLayerDiscriminator(3)
    D_params = list(discriminator.trainable_params())
    D_optimizer_params = [{"params": D_params, "lr": lr_scheduler}]

    G_params = list(animate_model.trainable_params())
    G_optimizer_params = [{"params": G_params, "lr": lr_scheduler}]

    optimizer_G = nn.Adam(G_optimizer_params, learning_rate=max_lr)
    optimizer_D = nn.Adam(D_optimizer_params, learning_rate=max_lr)

    eval_cb = EvalSaveCallback(animate_model)

    # define trainer
    vgg_feat_extractor = get_feature_extractor(config.facerender)
    trainer = facerender_trainer(
        animate_model,
        discriminator,
        vgg_feat_extractor,
        optimizer_G,
        optimizer_D,
        config,
    )
    initial_epoch = 0
    print(" training...")
    trainer.train(
        num_epochs * dataloader.get_dataset_size(),
        dataloader,
        callbacks=[eval_cb],
        dataset_sink_mode=False,
        initial_epoch=initial_epoch,
    )


if __name__ == "__main__":
    args, cfg = parse_args_and_config()
    config = Dict(cfg)
    train(args, config)
