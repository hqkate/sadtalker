import os
import cv2
import time
import random
import numpy as np
import mindspore as ms
from mindspore import ops

from glob import glob
from scipy.io import loadmat
from utils.get_file import get_img_paths


class TrainPVAEDataset:
    def __init__(self, input_path, syncnet_T=32, syncnet_mel_step_size=16):
        self.all_videos = get_img_paths(input_path)
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size

    # 得到帧数的id
    def get_frame_id(self, frame):
        return int(os.path.basename(frame).split("_")[-1].split(".")[0])  #

    # 得到window
    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidpath = os.path.dirname(start_frame)
        vidname = vidpath.split("/")[-1]

        window_fnames = []
        end_id = start_id + self.syncnet_T
        for frame_id in range(start_id, end_id):
            frame = os.path.join(
                vidpath, "{}.png".format(vidname + "_" + str(frame_id).zfill(6))
            )
            if not os.path.isfile(frame):
                return None, None, None
            window_fnames.append(frame)
        return window_fnames, start_id, end_id

    # 读取window的图像
    def read_window(self, window_fnames):
        if window_fnames is None:
            return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            h, w, c = img.shape
            if h != 256 or w != 256:
                img = cv2.resize(img, (256, 256))
            window.append(img)
        return window

    # 获取某一帧的audio
    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80.0 * (start_frame_num / float(25)))

        end_idx = start_idx + self.syncnet_mel_step_size

        seq = list(range(start_idx, end_idx))
        seq = [min(max(item, 0), spec.shape[0] - 1) for item in seq]
        return spec[seq, :]

    # 获取window内的audio
    def get_segmented_mels(self, spec, start_frame):
        mels = []
        start_frame_num = start_frame + 1
        if start_frame_num - 2 < 0:
            return None
        for i in range(start_frame_num, start_frame_num + self.syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != self.syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)
        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.0
        x = np.transpose(x, (3, 0, 1, 2))
        return x

    def parse_audio_length(self, audio_length, sr, fps):
        # time = audio_length / sr  #视频的长度
        # 那么对应的图像共有： num_frames = time * fps = audio_length / sr * fps
        bit_per_frames = sr / fps

        num_frames = int(audio_length / bit_per_frames)
        audio_length = int(num_frames * bit_per_frames)

        return audio_length, num_frames

    def crop_pad_audio(self, wav, audio_length):
        if len(wav) > audio_length:
            wav = wav[:audio_length]
        elif len(wav) < audio_length:
            wav = np.pad(
                wav, [0, audio_length - len(wav)], mode="constant", constant_values=0
            )
        return wav

    def __next__(self):
        if self._index >= len(self.all_videos):
            raise StopIteration
        else:
            item = self.__getitem__(self._index)
            self._index += 1
            return item

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        # 选择一个视频 找到对应的audio
        video_dir, label = self.all_videos[idx].split(" ")
        dirname = video_dir.split("/")[-1]
        image_paths = glob(os.path.join(video_dir, "*.png"))  # 图像的路径

        print(f"start process data {idx}.")

        # 读取音频并进行预处理
        start_time = time.time()
        wavpath = os.path.join(
            video_dir.replace("/images/", "/orig_mel/"), "orig_mel.npy"
        )
        # wav = audio.load_wav(wavpath, 16000)
        # wav_length, num_frames = self.parse_audio_length(len(wav), 16000, 25)    #将音频重采样并对准到25fps
        # wav = self.crop_pad_audio(wav, wav_length)
        # start_time = time.time()
        # orig_mel = audio.melspectrogram(wav).T    #得到特征
        orig_mel = np.load(wavpath)

        # 读取pose
        pose_path = os.path.join(
            video_dir.replace("/images/", "/pose/"), dirname + ".mat"
        )  #

        pose = loadmat(pose_path)
        coeff_3dmm = pose["coeff_3dmm"]  # (200, 73) 200 frames
        if coeff_3dmm.shape[0] != len(image_paths):
            print("mismatch coeff_3dmm and len(image_paths)", pose_path)

        start_time = time.time()
        ##随机选取一帧,得到窗口并读取图片
        valid_paths = list(sorted(image_paths))[: len(image_paths) - self.syncnet_T]
        img_path = random.choice(valid_paths)  # 随机选取一帧
        window_fnames, start_id, end_id = self.get_window(img_path)  #
        if window_fnames is None:
            raise ValueError("invalid image with none window fnames.")
        # window = self.read_window(window_fnames)
        # if window is None:  #读取的图像有误
        #     continue

        # 得到winow起始帧的wav以及整个window的wav
        # mel = self.crop_audio_window(orig_mel.copy(), 1)  #起始帧的mel
        # if (mel.shape[0] != self.syncnet_mel_step_size):
        #     continue
        first_indiv_mels = self.get_segmented_mels(orig_mel.copy(), 1)[:1, :, :]
        indiv_mels = self.get_segmented_mels(orig_mel.copy(), start_id)  # 整个window的mel
        if indiv_mels is None:
            raise ValueError("indiv_mels is none!")
        if indiv_mels.shape != (self.syncnet_T, 80, self.syncnet_mel_step_size):
            # print('indiv_mels mismatch', video_dir)
            raise ValueError(f"indiv_mels mismatch {video_dir}")

        # 得到第一帧的ceoff_3dmm以及整个window的ceoff_3dmm
        first_coeff_3dmm = np.expand_dims(coeff_3dmm[0], 0)  # ρ0
        window_coeff_3dmm = coeff_3dmm[start_id:end_id]  #
        ref_coeff = np.repeat(first_coeff_3dmm, self.syncnet_T, axis=0)  #
        if ref_coeff.shape != (self.syncnet_T, 73):
            # print('ref_coeff mismatch', video_dir)
            raise ValueError(f"ref_coeff mismatch {video_dir}.")
        if window_coeff_3dmm.shape != (self.syncnet_T, 73):
            # print('window_coeff_3dmm mismatch', video_dir)
            raise ValueError(f"window_coeff_3dmm mismatch {video_dir}.")

        # window = self.prepare_window(window)  #预处理图片
        first_coeff_3dmm = ms.Tensor(first_coeff_3dmm.astype(np.float32), ms.float32)
        window_coeff_3dmm = ms.Tensor(window_coeff_3dmm.astype(np.float32), ms.float32)
        first_indiv_mels = ms.Tensor(first_indiv_mels.astype(np.float32), ms.float32)
        indiv_mels = ms.Tensor(indiv_mels.astype(np.float32), ms.float32)
        first = ms.Tensor(first_coeff_3dmm.astype(np.float32), ms.float32)
        label = int(label)

        return {
            "num_frames": 32,
            "indiv_mels": ops.cat((first_indiv_mels, indiv_mels), 0).unsqueeze(0),
            "gt": ops.cat((first, window_coeff_3dmm), 0)[:, :70],
            "class": label,
        }

        # return indiv_mels, ref_coeff, window_coeff_3dmm, label
