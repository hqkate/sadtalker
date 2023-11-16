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

from datasets.generate_batch import get_data
from datasets.dataset_wrapper import DatasetWrapper
from utils.preprocess import CropAndExtract
from models.audio2pose.audio2pose import Audio2Pose
from argparse import ArgumentParser
from utils.callbacks import EvalSaveCallback
from trainer import GWithLossCell, DWithLossCell, GTrainOneStepCell, DTrainOneStepCell, VAEGTrainer
from train_expnet import init_path


def posevae_trainer(net, optimizer, cfg):
    is_train_finetune = False

    generator, discriminator = (
        net.netG,
        net.netD_motion,
    )
    generator_w_loss = GWithLossCell(generator, discriminator, cfg)
    discriminator_w_loss = DWithLossCell(discriminator)
    generator_t_step = GTrainOneStepCell(generator_w_loss, optimizer)
    discriminator_t_step = DTrainOneStepCell(discriminator_w_loss, optimizer)
    # generator_t_step = GTrainOneStepCell(generator_w_loss, optimizer.get("generator"))
    # discriminator_t_step = DTrainOneStepCell(discriminator_w_loss, optimizer.get("discriminator"))
    trainer = VAEGTrainer(generator_t_step, discriminator_t_step, cfg, is_train_finetune)
    return trainer


def train(cfg):
    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target="Ascend", device_id=1)
    # context.set_context(mode=context.GRAPH_MODE,
    #                     device_target="CPU", device_id=7)

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(
        current_root_path, 'config'), args.preprocess)

    # init preprocess model
    preprocess_model = CropAndExtract(sadtalker_paths)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, _, _ = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,
                                                       source_image_flag=True, pic_size=args.size)

    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(
            os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(
            ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path = None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(
                os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ = preprocess_model.generate(
                ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path = None

    # create train data loader
    batch_size = 1
    batch = get_data(first_coeff_path, audio_path,
                     ref_eyeblink_coeff_path, still=args.still)
    dataset = DatasetWrapper(batch)
    dataset_column_names = dataset.get_output_columns()
    ds = ms.dataset.GeneratorDataset(
        dataset,
        column_names=dataset_column_names,
        shuffle=True
    )
    loader_train = ds.batch(
        batch_size,
        drop_remainder=True,
    )

    # create model
    fcfg_pvae = open(sadtalker_paths['audio2pose_yaml_path'])
    cfg_pvae = CN.load_cfg(fcfg_pvae)
    cfg_pvae.freeze()

    net_audio2pose = Audio2Pose(cfg_pvae)
    net_audio2pose.set_train(True)
    auto_mixed_precision(net_audio2pose, "O0")

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
    total_step = len(batch) * num_epochs
    step_per_epoch = len(batch)
    lr_scheduler = nn.cosine_decay_lr(
        min_lr, max_lr, total_step, step_per_epoch, decay_epoch)

    optimizer_params = [p for p in net_audio2pose.get_parameters()]
    optimizer_params = [{"params": optimizer_params, "lr": lr_scheduler}]

    # build optimizer
    optimizer = nn.AdamWeightDecay(
        optimizer_params, learning_rate=max_lr)


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
    eval_cb = EvalSaveCallback(
        net_audio2pose
    )

    # define trainer
    trainer = posevae_trainer(net_audio2pose, optimizer, cfg_pvae)
    initial_epoch = 0
    epoch_size = 5
    print(" training...")
    trainer.train(
        epoch_size * loader_train.get_dataset_size(),
        loader_train,
        callbacks=[eval_cb],
        dataset_sink_mode=False,
        initial_epoch=initial_epoch,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument(
        "--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None,
                        help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None,
                        help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir",
                        default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results',
                        help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,
                        help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,
                        help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float,
                        default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int,
                        default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int,
                        default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int,
                        default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None,
                        help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str,
                        default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.add_argument("--face3dvis", action="store_true",
                        help="generate 3d face and 3d landmarks")
    parser.add_argument("--still", action="store_true",
                        help="can crop back to the original videos for the full body aniamtion")
    parser.add_argument("--preprocess", default='crop', choices=[
                        'crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images")
    parser.add_argument("--verbose", action="store_true",
                        help="saving the intermedia output or not")

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc', default=False,
                        help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str,
                        default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str,
                        default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    train(args)
