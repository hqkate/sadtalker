import os
import sys
import shutil
from time import strftime
from addict import Dict
from utils.arg_parser import parse_args_and_config

args, cfg = parse_args_and_config()

import mindspore as ms
from mindspore import context
from mindspore.amp import auto_mixed_precision

from utils.preprocess import CropAndExtract
<<<<<<< HEAD
from datasets.dataset_audio2coeff import TestDataset
from datasets.dataset_facerender import TestFaceRenderDataset
from models.audio2coeff import Audio2Coeff
from models.facerender.animate import AnimateFromCoeff
from argparse import ArgumentParser


def init_path(
    checkpoint_dir="./checkpoints/", config_dir="./config/", preprocess="crop"
):
    sadtalker_paths = {
        "wav2lip_checkpoint": os.path.join(checkpoint_dir, "wav2lip.pth"),
        "audio2pose_checkpoint": os.path.join(checkpoint_dir, "ms/ms_audio2pose.ckpt"),
        "audio2exp_checkpoint": os.path.join(checkpoint_dir, "ms/ms_audio2exp.ckpt"),
        "path_of_net_recon_model": os.path.join(checkpoint_dir, "ms/ms_net_recon.ckpt"),
        "dir_of_BFM_fitting": os.path.join(checkpoint_dir, "BFM_Fitting"),
        "generator_checkpoint": os.path.join(checkpoint_dir, "ms/ms_generator.ckpt"),
        "kp_extractor_checkpoint": os.path.join(
            checkpoint_dir, "ms/ms_kp_extractor.ckpt"
        ),
        "he_estimator_checkpoint": os.path.join(
            checkpoint_dir, "ms/ms_he_estimator.ckpt"
        ),
    }
    sadtalker_paths["audio2pose_yaml_path"] = os.path.join(
        config_dir, "audio2pose.yaml"
    )
    sadtalker_paths["audio2exp_yaml_path"] = os.path.join(config_dir, "audio2exp.yaml")

    if "full" in preprocess:
        sadtalker_paths["mappingnet_checkpoint"] = os.path.join(
            checkpoint_dir, "ms/ms_mapping_full.ckpt"
        )
        sadtalker_paths["facerender_yaml"] = os.path.join(
            config_dir, "facerender_still.yaml"
        )
    else:
        sadtalker_paths["mappingnet_checkpoint"] = os.path.join(
            checkpoint_dir, "ms/ms_mapping.ckpt"
        )
        sadtalker_paths["facerender_yaml"] = os.path.join(config_dir, "facerender.yaml")
=======
from datasets.generate_batch import get_data
from datasets.generate_facerender_batch import get_facerender_data
from models.audio2coeff import Audio2Coeff
from models.facerender.animate import AnimateFromCoeff
>>>>>>> refactor-infer


<<<<<<< HEAD

def main(args):
    # context.set_context(mode=context.PYNATIVE_MODE,
    #                     device_target="Ascend", device_id=7)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=7)
=======
def main(args, config):
    context.set_context(mode=config.system.mode,
                        device_target="Ascend",
                        device_id=int(args.device_id)
                        )
>>>>>>> refactor-infer

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style

<<<<<<< HEAD
    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(
        args.checkpoint_dir, os.path.join(current_root_path, "config"), args.preprocess
    )

=======
>>>>>>> refactor-infer
    # init model
    preprocess_model = CropAndExtract(config.preprocess)
    audio_to_coeff = Audio2Coeff(config)
    animate_from_coeff = AnimateFromCoeff(config.facerender)

    amp_level = config.system.get("amp_level", "O0")
    auto_mixed_precision(audio_to_coeff.audio2exp_model, amp_level)
    auto_mixed_precision(audio_to_coeff.audio2pose_model, amp_level)
    auto_mixed_precision(animate_from_coeff.generator, amp_level)

    testdataset = TestDataset(
        args=args,
        preprocessor=preprocess_model,
        save_dir=save_dir,
    )

    batch = testdataset.__getitem__(0)
    ref_pose_coeff_path = batch["ref_pose_coeff_path"]
    crop_pic_path = batch["crop_pic_path"]
    first_coeff_path = batch["first_coeff_path"]
    crop_info = batch["crop_info"]

<<<<<<< HEAD
    coeff_path = audio_to_coeff.generate(
        batch, save_dir, pose_style, ref_pose_coeff_path
    )

=======
>>>>>>> refactor-infer
    # coeff2video
    facerender_dataset = TestFaceRenderDataset(
        args=args,
        coeff_path=coeff_path,
        pic_path=crop_pic_path,
        first_coeff_path=first_coeff_path,
        audio_path=args.driven_audio,
        batch_size=args.batch_size,
        expression_scale=args.expression_scale,
        still_mode=args.still,
        preprocess=args.preprocess,
        size=args.size,
    )

    data = facerender_dataset.__getitem__(0)
    result = animate_from_coeff.generate(
        data,
        save_dir,
        pic_path,
        crop_info,
        enhancer=args.enhancer,
        background_enhancer=args.background_enhancer,
        preprocess=args.preprocess,
        img_size=args.size,
    )

    shutil.move(result, save_dir + ".mp4")
    print("The generated video is named:", save_dir + ".mp4")

    if not args.verbose:
        shutil.rmtree(save_dir)


<<<<<<< HEAD
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

    main(args)
=======
if __name__ == '__main__':

    config = Dict(cfg)
    main(args, config)
>>>>>>> refactor-infer
