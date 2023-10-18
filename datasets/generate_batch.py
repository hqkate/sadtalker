import os

from tqdm import tqdm
import mindspore as ms
from mindspore import ops
import numpy as np
import random
import cv2
import scipy.io as scio
import utils.audio as audio


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)],
                     mode='constant', constant_values=0)
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
        if frame_id+start+9 <= num_frames - 1:
            ratio[frame_id+start:frame_id+start+9, 0] = [0.5,
                                                         0.6, 0.7, 0.9, 1, 0.9, 0.7, 0.6, 0.5]
            frame_id = frame_id+start+9
        else:
            break
    return ratio


def generate_blink_seq_randomly(num_frames):
    ratio = np.zeros((num_frames, 1))
    if num_frames <= 20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        start = random.choice(
            range(min(10, num_frames), min(int(num_frames/2), 70)))
        if frame_id+start+5 <= num_frames - 1:
            ratio[frame_id+start:frame_id+start +
                  5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
            frame_id = frame_id+start+5
        else:
            break
    return ratio


def get_data(first_coeff_path, audio_path, ref_eyeblink_coeff_path, still=False, idlemode=False, length_of_audio=False, use_blink=True):

    syncnet_mel_step_size = 16
    fps = 25

    first_frame_dir = os.path.dirname(first_coeff_path)
    pic_name = os.path.splitext(os.path.split(first_coeff_path)[-1])[0]
    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]

    if idlemode:
        num_frames = int(length_of_audio * 25)
        indiv_mels = np.zeros((num_frames, 80, 16))
    else:
        wav = audio.load_wav(audio_path, 16000)
        wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = audio.melspectrogram(wav).T
        spec = orig_mel.copy()         # nframes 80
        indiv_mels = []

        for i in tqdm(range(num_frames), 'mel:'):
            start_frame_num = i-2
            start_idx = int(80. * (start_frame_num / float(fps)))
            end_idx = start_idx + syncnet_mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [min(max(item, 0), orig_mel.shape[0]-1) for item in seq]
            m = spec[seq, :]
            indiv_mels.append(m.T)
        indiv_mels = np.asarray(indiv_mels)         # T 80 16

    ratio = generate_blink_seq_randomly(num_frames)      # T
    source_semantics_path = first_coeff_path
    source_semantics_dict = scio.loadmat(source_semantics_path)
    ref_coeff = source_semantics_dict['coeff_3dmm'][:1, :70]  # 1 70
    ref_coeff = np.repeat(ref_coeff, num_frames, axis=0)

    if ref_eyeblink_coeff_path is not None:
        ratio[:num_frames] = 0
        refeyeblink_coeff_dict = scio.loadmat(ref_eyeblink_coeff_path)
        refeyeblink_coeff = refeyeblink_coeff_dict['coeff_3dmm'][:, :64]
        refeyeblink_num_frames = refeyeblink_coeff.shape[0]
        if refeyeblink_num_frames < num_frames:
            div = num_frames//refeyeblink_num_frames
            re = num_frames % refeyeblink_num_frames
            refeyeblink_coeff_list = [refeyeblink_coeff for i in range(div)]
            refeyeblink_coeff_list.append(refeyeblink_coeff[:re, :64])
            refeyeblink_coeff = np.concatenate(refeyeblink_coeff_list, axis=0)
            print(refeyeblink_coeff.shape[0])

        ref_coeff[:, :64] = refeyeblink_coeff[:num_frames, :64]

    indiv_mels = ms.Tensor(indiv_mels, ms.float32).unsqueeze(
        1).unsqueeze(0)  # bs T 1 80 16

    if use_blink:
        ratio = ms.Tensor(ratio, ms.float32).unsqueeze(
            0)                       # bs T
    else:
        ratio = ms.Tensor(ratio, ms.float32).unsqueeze(0).fill_(0.)
        # bs T

    ref_coeff = np.asarray(ref_coeff).astype('float32')
    ref_coeff = ms.Tensor(ref_coeff).unsqueeze(0)                # bs 1 70

    pic_name = os.path.join(first_frame_dir, pic_name + '.png')
    bs = indiv_mels.shape[0]
    img = cv2.imread(pic_name)
    img = np.asarray([cv2.resize(img, (96, 96))] * bs)
    img_masked = img.copy()
    img_masked[:, 96 // 2:] = 0
    img_input = np.concatenate((img_masked, img), axis=3) / 255.
    first_frame_img = ms.Tensor(np.transpose(
        img_input, (0, 3, 1, 2)), dtype=ms.float32)

    img = ms.Tensor((img), ms.float32)
    img_masked = ms.Tensor((img_masked), ms.float32)

    # split frames
    T = 12
    indiv_mels = ops.cat(ops.tensor_split(
        indiv_mels, num_frames // T, axis=1), axis=0)
    ref_coeff = ops.cat(ops.tensor_split(
        ref_coeff, num_frames // T, axis=1), axis=0)[:, :, :64]
    ratio = ops.cat(ops.tensor_split(
        ratio, num_frames // T, axis=1), axis=0)
    first_frame_img = first_frame_img.repeat(num_frames, axis=0)

    return {'indiv_mels': indiv_mels,
            'ref': ref_coeff,
            'num_frames': num_frames,
            'ratio_gt': ratio,
            # 'audio_name': audio_name,
            'img_array': img,
            'first_frame_img': first_frame_img
            }
