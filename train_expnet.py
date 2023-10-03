import os
import sys
import shutil
from time import strftime
from yacs.config import CfgNode as CN
import mindspore as ms
from mindspore import context
from mindspore.amp import auto_mixed_precision
from utils.generate_batch import get_data
from utils.preprocess import CropAndExtract
from models.audio2exp.expnet import ExpNet
from models.audio2exp.wav2lip import Wav2Lip
from models.audio2exp.audio2exp import Audio2Exp
from argparse import ArgumentParser
from models.face3d.networks import define_net_recon


def init_path(checkpoint_dir="./checkpoints/", config_dir="./config/", preprocess='crop'):

    sadtalker_paths = {
        'wav2lip_checkpoint': os.path.join(checkpoint_dir, 'ms/ms_wav2lip.ckpt'),
        'audio2pose_checkpoint': os.path.join(checkpoint_dir, 'ms/ms_audio2pose.ckpt'),
        'audio2exp_checkpoint': os.path.join(checkpoint_dir, 'ms/ms_audio2exp.ckpt'),
        'path_of_net_recon_model': os.path.join(checkpoint_dir, 'ms/ms_net_recon.ckpt'),
        'dir_of_BFM_fitting': os.path.join(checkpoint_dir, 'BFM_Fitting'),
        'generator_checkpoint': os.path.join(checkpoint_dir, 'ms/ms_generator.ckpt'),
        'kp_extractor_checkpoint': os.path.join(checkpoint_dir, 'ms/ms_kp_extractor.ckpt'),
        'he_estimator_checkpoint': os.path.join(checkpoint_dir, 'ms/ms_he_estimator.ckpt')
    }
    sadtalker_paths['audio2pose_yaml_path'] = os.path.join(
        config_dir, 'audio2pose.yaml')
    sadtalker_paths['audio2exp_yaml_path'] = os.path.join(
        config_dir, 'audio2exp.yaml')

    if 'full' in preprocess:
        sadtalker_paths['mappingnet_checkpoint'] = os.path.join(
            checkpoint_dir, 'ms/ms_mapping_full.ckpt')
        sadtalker_paths['facerender_yaml'] = os.path.join(
            config_dir, 'facerender_still.yaml')
    else:
        sadtalker_paths['mappingnet_checkpoint'] = os.path.join(
            checkpoint_dir, 'ms/ms_mapping.ckpt')
        sadtalker_paths['facerender_yaml'] = os.path.join(
            config_dir, 'facerender.yaml')

    return sadtalker_paths


def main(args):
    # context.set_context(mode=context.PYNATIVE_MODE,
    #                     device_target="Ascend", device_id=7)
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", device_id=7)

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

    # init model
    preprocess_model = CropAndExtract(sadtalker_paths)

    # load audio2exp_model
    netG = ExpNet()
    for param in netG.get_parameters():
        netG.requires_grad = False
    netG.set_train(False)

    # load wav2lip model
    wav2lip = Wav2Lip()
    param_dict = ms.load_checkpoint(sadtalker_paths['wav2lip_checkpoint'])
    ms.load_param_into_net(wav2lip, param_dict)

    # load 3DMM Encoder
    coeff_enc = define_net_recon(
        net_recon='resnet50', use_last_fc=False, init_path='')
    param_dict = ms.load_checkpoint(
        sadtalker_paths['path_of_net_recon_model'])
    ms.load_param_into_net(coeff_enc, param_dict)
    coeff_enc.set_train(False)

    fcfg_exp = open(sadtalker_paths['audio2exp_yaml_path'])
    cfg_exp = CN.load_cfg(fcfg_exp)
    cfg_exp.freeze()

    audio2exp_model = Audio2Exp(
        netG, cfg_exp, wav2lip=wav2lip, coeff_enc=coeff_enc, is_train=True)

    for param in audio2exp_model.get_parameters():
        param.requires_grad = True

    audio2exp_model.set_train(True)

    auto_mixed_precision(audio2exp_model, "O0")

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,
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

    # audio2ceoff
    batch = get_data(first_coeff_path, audio_path,
                     ref_eyeblink_coeff_path, still=args.still)

    batch['pic_name'] = os.path.join(
        first_frame_dir, batch['pic_name'] + '.png')

    results_dicts = audio2exp_model.getloss(batch)


if __name__ == '__main__':

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

    main(args)
