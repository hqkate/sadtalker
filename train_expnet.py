import os
from time import strftime
from addict import Dict
from utils.arg_parser import parse_args_and_config

import mindspore as ms
from mindspore import context, nn
from mindspore.amp import auto_mixed_precision

from utils.preprocess import CropAndExtract
from utils.callbacks import EvalSaveCallback
from datasets.dataset_audio2coeff import TrainAudioCoeffDataset

from models.audio2exp.expnet import ExpNet
from models.audio2exp.audio2exp import Audio2Exp
from models.audio2exp.trainer import ExpNetWithLossCell, ExpNetTrainer
from models.face3d.bfm import ParametricFaceModel


def expnet_trainer(audio2exp, optimizer, config, args):

    bfm = ParametricFaceModel(bfm_folder="checkpoints/BFM_Fitting", is_train=False)

    expnet_w_loss = ExpNetWithLossCell(audio2exp, bfm, config, args)
    expnet_t_step = nn.TrainOneStepCell(expnet_w_loss, optimizer)

    trainer = ExpNetTrainer(expnet_t_step, config)

    return trainer


def train(args, config):
    context.set_context(
        mode=context.PYNATIVE_MODE,
        pynative_synchronize=True,
        device_target="CPU",
        device_id=args.device_id,
    )

    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)

    # init model
    preprocess_model = CropAndExtract(config.preprocess)

    # load audio2exp_model
    netG = ExpNet()
    netG.set_train(True)

    audio2exp_model = Audio2Exp(netG, config.audio2exp.model, is_train=True)

    for param in audio2exp_model.get_parameters():
        param.requires_grad = True

    audio2exp_model.set_train(True)

    # amp level
    amp_level = config.system.get("amp_level", "O0")
    auto_mixed_precision(audio2exp_model, amp_level)

    # dataset
    ds_train = TrainAudioCoeffDataset(
        args=args,
        preprocessor=preprocess_model,
        save_dir=save_dir,
        use_wav2lip=True,
    )

    dataset_column_names = ds_train.get_output_columns()
    ds_generator = ms.dataset.GeneratorDataset(
        ds_train, column_names=dataset_column_names, shuffle=True
    )

    dataloader = ds_generator.batch(
        batch_size=args.batch_size,
        drop_remainder=True,
    )

    # lr scheduler
    min_lr = 0.0
    max_lr = 0.0005
    num_epochs = 10
    decay_epoch = 2
    total_step = dataloader.get_dataset_size() * num_epochs
    step_per_epoch = dataloader.get_dataset_size()
    lr_scheduler = nn.cosine_decay_lr(
        min_lr, max_lr, total_step, step_per_epoch, decay_epoch
    )

    # build optimizer
    model_params = list(audio2exp_model.trainable_params())
    optimizer_params = [{"params": model_params, "lr": lr_scheduler}]
    optimizer = nn.Adam(optimizer_params, learning_rate=max_lr)

    trainer = expnet_trainer(
        audio2exp_model,
        optimizer,
        config,
        args,
    )

    # build callbacks
    eval_cb = EvalSaveCallback(audio2exp_model)

    # training
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
