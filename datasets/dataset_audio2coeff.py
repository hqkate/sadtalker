import os
import cv2
import random
from tqdm import tqdm
from glob import glob
import scipy.io as scio
import numpy as np
import mindspore as ms
import utils.audio as audio
from PIL import Image
from skimage import img_as_float32
from utils.preprocess import split_coeff


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(
            wav, [0, audio_length - len(wav)], mode="constant", constant_values=0
        )
    return wav


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames


def generate_blink_seq(num_frames):
    ratio = np.zeros((num_frames, 1))
    frame_id = 0
    while frame_id in range(num_frames):
        start = 80
        if frame_id + start + 9 <= num_frames - 1:
            ratio[frame_id + start : frame_id + start + 9, 0] = [
                0.5,
                0.6,
                0.7,
                0.9,
                1,
                0.9,
                0.7,
                0.6,
                0.5,
            ]
            frame_id = frame_id + start + 9
        else:
            break
    return ratio


def generate_blink_seq_randomly(num_frames):
    ratio = np.zeros((num_frames, 1))
    if num_frames <= 20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        start = random.choice(range(min(10, num_frames), min(int(num_frames / 2), 70)))
        if frame_id + start + 5 <= num_frames - 1:
            ratio[frame_id + start : frame_id + start + 5, 0] = [
                0.5,
                0.9,
                1.0,
                0.9,
                0.5,
            ]
            frame_id = frame_id + start + 5
        else:
            break
    return ratio


def read_filelist(input_path):
    audios = []
    images = []

    if os.path.isfile(input_path):
        ext = input_path.split(".")[-1]
        if ext == "txt":
            with open(input_path, "r") as f:
                for line in f.read().splitlines():
                    audio_path, image_path = line.split(" ")

                    if os.path.isfile(audio_path) and os.path.isfile(image_path):
                        audios.append(audio_path)
                        images.append(image_path)

                    # if os.path.isfile(audio_path) and os.path.isdir(image_dir):
                    #     audios.append(audio_path)
                    #     image_paths = glob(os.path.join(image_dir, "*.png"))
                    #     images.append(image_paths)

    return audios, images


class AudioCoeffDataset:
    def __init__(
        self,
        args,
        preprocessor,
        save_dir,
        syncnet_mel_step_size=16,
        fps=25,
        idlemode=False,
        length_of_audio=False,
        use_blink=True,
    ):
        self.args = args
        self.preprocessor = preprocessor
        self.save_dir = save_dir
        self.first_frame_dir = os.path.join(self.save_dir, "first_frame_dir")

        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.fps = fps
        self.still = self.args.still
        self.idlemode = idlemode
        self.length_of_audio = length_of_audio
        self.use_blink = use_blink

    def crop_and_extract(self, source_image):
        os.makedirs(self.first_frame_dir, exist_ok=True)
        print("3DMM Extraction for source image")

        first_coeff_path, crop_pic_path, crop_info, crop_frames = self.preprocessor.generate(
            source_image,
            self.first_frame_dir,
            self.args.preprocess,
            source_image_flag=True,
            pic_size=self.args.size,
        )

        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if self.args.ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(
                os.path.split(self.args.ref_eyeblink)[-1]
            )[0]
            ref_eyeblink_frame_dir = os.path.join(self.save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print("3DMM Extraction for the reference video providing eye blinking")
            ref_eyeblink_coeff_path, _, _ = self.preprocessor.generate(
                self.args.ref_eyeblink,
                ref_eyeblink_frame_dir,
                self.args.preprocess,
                source_image_flag=False,
            )
        else:
            ref_eyeblink_coeff_path = None

        if self.args.ref_pose is not None:
            if self.args.ref_pose == self.args.ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(
                    os.path.split(self.args.ref_pose)[-1]
                )[0]
                ref_pose_frame_dir = os.path.join(self.save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print("3DMM Extraction for the reference video providing pose")
                ref_pose_coeff_path, _, _ = self.preprocessor.generate(
                    self.args.ref_pose,
                    ref_pose_frame_dir,
                    self.args.preprocess,
                    source_image_flag=False,
                )
        else:
            ref_pose_coeff_path = None

        return (
            first_coeff_path,
            crop_pic_path,
            crop_info,
            ref_eyeblink_coeff_path,
            ref_pose_coeff_path,
            crop_frames
        )

    def _get_idx_seq(self, start_frame_num, mel_size):
        start_idx = int(80.0 * (start_frame_num / float(self.fps)))
        end_idx = start_idx + self.syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [min(max(item, 0), mel_size - 1) for item in seq]
        return seq

    def process_audio(self, audio_path, is_train=False):
        wav = audio.load_wav(audio_path, 16000)
        wav_length, num_frames = parse_audio_length(len(wav), 16000, self.fps)
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = audio.melspectrogram(wav).T
        spec = orig_mel.copy()  # nframes 80
        indiv_mels = []
        frame_idx = 0

        if not is_train:  # process all frames
            for i in tqdm(range(num_frames), "mel:"):
                start_frame_num = i - 2
                seq = self._get_idx_seq(start_frame_num, orig_mel.shape[0])
                m = spec[seq, :]
                indiv_mels.append(m.T)
            indiv_mels = np.asarray(indiv_mels)  # T 80 16

            indiv_mels = (
                ms.Tensor(indiv_mels, ms.float32).unsqueeze(1).unsqueeze(0)
            )  # bs T 1 80 16

        else:  # randomly select 5 frames for training
            frame_idx = random.choice(range(5, num_frames))
            for i in tqdm(range(frame_idx, frame_idx+5), "mel:"):
                start_frame_num = i - 2
                seq = self._get_idx_seq(start_frame_num, orig_mel.shape[0])
                m = spec[seq, :]
                indiv_mels.append(m.T)
            indiv_mels = np.asarray(indiv_mels)  # T 80 16

            indiv_mels = (
                ms.Tensor(indiv_mels, ms.float32).unsqueeze(1)
            )  # 5 1 80 16

            num_frames = 5

        return num_frames, indiv_mels, frame_idx, wav

    def gen_ref_coeffs(
        self, num_frames, first_coeff_path, ref_eyeblink_coeff_path, is_train=False
    ):
        ratio = generate_blink_seq_randomly(num_frames)  # T
        source_semantics_path = first_coeff_path
        source_semantics_dict = scio.loadmat(source_semantics_path)

        ref_coeff = source_semantics_dict["coeff_3dmm"][:1, :70]  # 1 70
        ref_coeff = np.repeat(ref_coeff, num_frames, axis=0)

        if ref_eyeblink_coeff_path is not None:
            ratio[:num_frames] = 0
            refeyeblink_coeff_dict = scio.loadmat(ref_eyeblink_coeff_path)
            refeyeblink_coeff = refeyeblink_coeff_dict["coeff_3dmm"][:, :64]
            refeyeblink_num_frames = refeyeblink_coeff.shape[0]
            if refeyeblink_num_frames < num_frames:
                div = num_frames // refeyeblink_num_frames
                re = num_frames % refeyeblink_num_frames
                refeyeblink_coeff_list = [refeyeblink_coeff for i in range(div)]
                refeyeblink_coeff_list.append(refeyeblink_coeff[:re, :64])
                refeyeblink_coeff = np.concatenate(refeyeblink_coeff_list, axis=0)
                print(refeyeblink_coeff.shape[0])

            ref_coeff[:, :64] = refeyeblink_coeff[:num_frames, :64]

        if self.use_blink:
            ratio = ms.Tensor(ratio, ms.float32).unsqueeze(0)  # bs T
        else:
            ratio = ms.Tensor(ratio, ms.float32).unsqueeze(0).fill_(0.0)
            # bs T

        ref_coeff = np.asarray(ref_coeff).astype("float32")

        if not is_train:
            ref_coeff = ms.Tensor(ref_coeff).unsqueeze(0)  # bs 1 70
        else:
            ref_coeff = ms.Tensor(ref_coeff)

        return ratio, ref_coeff

    def process_data(self, image_path, audio_path):
        # 1. crop and extract 3dMM coefficients
        (
            first_coeff_path,
            crop_pic_path,
            crop_info,
            ref_eyeblink_coeff_path,
            ref_pose_coeff_path,
            _,
        ) = self.crop_and_extract(image_path)

        # 2. process audio
        pic_name = os.path.splitext(os.path.split(image_path)[-1])[0]
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]

        if self.idlemode:
            num_frames = int(self.length_of_audio * self.fps)
            indiv_mels = np.zeros((num_frames, num_frames, self.syncnet_mel_step_size))
        else:
            num_frames, indiv_mels, _, _ = self.process_audio(audio_path, is_train=False)  # bs T 1 80 16

        # 3. generate ref coeffs
        ratio, ref_coeff = self.gen_ref_coeffs(
            num_frames, first_coeff_path, ref_eyeblink_coeff_path, is_train=False
        )

        return {
            "indiv_mels": indiv_mels,
            "ref": ref_coeff,
            "num_frames": num_frames,
            "ratio_gt": ratio,
            "audio_name": audio_name,
            "pic_name": pic_name,
            "ref_pose_coeff_path": ref_pose_coeff_path,
            "first_coeff_path": first_coeff_path,
            "crop_pic_path": crop_pic_path,
            "crop_info": crop_info,
        }


class TrainAudioCoeffDataset(AudioCoeffDataset):
    def __init__(
        self,
        args,
        preprocessor,
        save_dir,
        syncnet_mel_step_size=16,
        fps=25,
        idlemode=False,
        length_of_audio=False,
        use_blink=True,
        use_wav2lip=True,
    ):
        super().__init__(
            args=args,
            preprocessor=preprocessor,
            save_dir=save_dir,
            syncnet_mel_step_size=syncnet_mel_step_size,
            fps=fps,
            idlemode=idlemode,
            length_of_audio=length_of_audio,
            use_blink=use_blink,
        )

        self.audios, self.images = read_filelist(args.train_list)
        self.use_wav2lip = use_wav2lip

    def get_output_columns(self):
        cols = [
            "indiv_mels",
            "ref",
            "num_frames",
            "ratio_gt",
            "source_image",
        ]
        if self.use_wav2lip:
            cols.append("masked_src_img")
            # cols.append("exp_coeff_wav2lip")
        return cols

    def process_data(self, image_path, audio_path):

        frame_idx = 0

        # 1. crop and extract 3dMM coefficients
        (
            first_coeff_path,
            crop_pic_path,
            crop_info,
            ref_eyeblink_coeff_path,
            ref_pose_coeff_path,
            crop_frames,
        ) = self.crop_and_extract(image_path)

        # 2. process audio
        if self.idlemode:
            num_frames = int(self.length_of_audio * self.fps)
            indiv_mels = np.zeros((num_frames, num_frames, self.syncnet_mel_step_size))
        else:
            num_frames, indiv_mels, frame_idx, audio_wav = self.process_audio(audio_path, is_train=True)  # bs T 1 80 16

        # 3. generate ref coeffs
        ratio, ref_coeff = self.gen_ref_coeffs(
            num_frames, first_coeff_path, ref_eyeblink_coeff_path, is_train=True
        )

        # 4. images
        src_image = crop_frames[0]
        src_image = np.array(src_image)
        src_image_ts = ms.Tensor(img_as_float32(src_image))

        # crop target image (ground truth) according to source image
        # tgt_image = crop_frames[frame_idx]
        # tgt_image = ms.Tensor(img_as_float32(tgt_image))

        data_dict = {
            "indiv_mels": indiv_mels,
            "ref": ref_coeff,
            "num_frames": num_frames,
            "frame_idx": frame_idx,
            "ratio_gt": ratio,
            "audio_wav": audio_wav,
            "ref_pose_coeff_path": ref_pose_coeff_path,
            "first_coeff_path": first_coeff_path,
            "crop_pic_path": crop_pic_path,
            "crop_info": crop_info,
            "source_image": src_image_ts,
        }

        # 5. wav2lip
        # mask image for wav2lip
        img_size = 96
        src_image_resized = cv2.resize(src_image, (img_size, img_size))
        masked_image = src_image_resized.copy()
        masked_image = cv2.resize(masked_image, (img_size, img_size))
        masked_image[:, img_size // 2 :] = 0.0
        masked_src_img = np.concatenate((masked_image, src_image_resized), axis=2) / 255.0
        masked_src_img = np.repeat(np.expand_dims(np.transpose(masked_src_img, (2, 0, 1)), 0), num_frames, axis=0)
        masked_src_img = ms.Tensor(masked_src_img, ms.float32)

        if self.use_wav2lip:
            data_dict["masked_src_img"] = masked_src_img

        return data_dict

    def __next__(self):
        if self._index >= len(self.all_videos):
            raise StopIteration
        else:
            item = self.__getitem__(self._index)
            self._index += 1
            return item

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        image_paths = self.images[idx]
        audio_path = self.audios[idx]

        data_dict = self.process_data(image_paths, audio_path)
        output_tuple = tuple(data_dict[k] for k in self.get_output_columns())
        return output_tuple

        # while True:
        #     try:
        #         data_dict = self.process_data(image_paths, audio_path)
        #         output_tuple = tuple(data_dict[k] for k in self.get_output_columns())
        #         return output_tuple

        #     except Exception as e:
        #         random_idx = random.choice(list(range(len(self.audios))))
        #         return self.__getitem__(random_idx)
