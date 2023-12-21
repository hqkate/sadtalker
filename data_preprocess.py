import subprocess, cv2, os
from multiprocessing.pool import Pool
from functools import partial
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from addict import Dict
from utils.arg_parser import parse_args_and_config

args, cfg = parse_args_and_config()

from mindspore import context
from train_expnet import init_path
from utils.get_file import get_img_paths
from utils.preprocess import CropAndExtract
from datasets.dataset_audio2coeff import TestDataset
from models.audio2coeff import Audio2Coeff


template = "./ffmpeg-6.0-amd64-static/ffmpeg -loglevel panic -y -i {} -strict -2 {}"  # for save audio


def extract_audios_from_videos_multi(video_paths, audio_save_dir, video_org_dir):
    os.makedirs(audio_save_dir, exist_ok=True)
    with Pool(processes=1) as p:
        with tqdm(total=len(video_paths)) as pbar:
            func = partial(
                extract_audios_from_videos,
                audio_save_dir=audio_save_dir,
                video_org_dir=video_org_dir,
            )
            for v in p.imap_unordered(func, video_paths):
                pbar.update()


def extract_audios_from_videos(vfile, audio_save_dir, video_org_dir):
    try:
        vidname = os.path.basename(vfile).split(".")[0]

        fulldir = vfile.replace(video_org_dir, audio_save_dir)
        fulldir = fulldir.split(".")[0]
        os.makedirs(fulldir, exist_ok=True)

        wavpath = os.path.join(fulldir, "audio.wav")
        command = template.format(vfile, wavpath)
        subprocess.call(command, shell=True)
    except Exception as e:
        print(e)
        return


def mp_handler(job):
    vfile, save_root, org_root, gpu_id = job
    print(
        "processing ====>{}, current_npu:{}, process indx {}".format(
            vfile, gpu_id, os.getpid()
        )
    )
    fa[gpu_id].generate_for_train(
        vfile, save_root, org_root, "crop", source_image_flag=False, pic_size=size
    )


def mp_handler_audio2coeff(job):
    vfile, save_root, org_root, gpu_id = job
    print(
        "processing ====>{}, current_npu:{}, process indx {}".format(
            vfile, gpu_id, os.getpid()
        )
    )
    fa_audio2coeff[gpu_id].generate_for_train(
        vfile, save_root, org_root, "crop", source_image_flag=False, pic_size=size
    )


if __name__ == "__main__":
    config = Dict(cfg)
    context.set_context(
        mode=config.system.mode, device_target="Ascend", device_id=int(args.device_id)
    )

    # 预处理结果保存的路径
    preprocess_save_dir = "/disk1/katekong/sadtalker/data_train/"
    os.makedirs(preprocess_save_dir, exist_ok=True)

    # #step1:提取audio
    input_dir = "/disk1/katekong/sadtalker/data_train/video/"
    video_org_dir = "/disk1/katekong/sadtalker/data_train/video/"
    audio_save_dir = preprocess_save_dir + "/audio/"
    coeff_save_dir = preprocess_save_dir + "/coeffs/"
    video_paths = get_img_paths(input_dir, ext="mp4")
    # extract_audios_from_videos_multi(video_paths, audio_save_dir, video_org_dir)

    # step2： 这将花费相当长的时间
    # 读取预处理模型
    checkpoint_dir = "./checkpoints"
    config_dir = "./config/"
    size = 256
    old_version = False
    preprocess = "crop"
    ngpu = 1  # 采用GPU的数量
    fa = [CropAndExtract(config.preprocess) for _ in range(ngpu)]  # 构建GPU

    pose_save_dir = preprocess_save_dir + "/pose/"  # 保存ρ的路径
    os.makedirs(pose_save_dir, exist_ok=True)
    jobs = [
        (vfile, pose_save_dir, video_org_dir, i % ngpu)
        for i, vfile in enumerate(video_paths)
    ]
    p = ThreadPoolExecutor(ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    # step3: audio2coeff
    audio_to_coeff = Audio2Coeff(config)
    fa_audio2coeff = [audio_to_coeff for _ in range(ngpu)]  # 构建GPU

    predcoeff_save_dir = preprocess_save_dir + "/pred_coeffs/"
    os.makedirs(predcoeff_save_dir, exist_ok=True)
    jobs = [
        (vfile, predcoeff_save_dir, video_org_dir, i % ngpu)
        for i, vfile in enumerate(video_paths)
    ]
    p = ThreadPoolExecutor(ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
