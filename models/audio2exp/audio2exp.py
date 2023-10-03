from tqdm import tqdm
import cv2
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from utils.preprocess import split_coeff
from models.face3d.bfm import ParametricFaceModel


class Audio2Exp(nn.Cell):
    """ ExpNet implementation (training)
    """

    def __init__(self, netG, cfg, wav2lip=None, coeff_enc=None, coeff_dec=None, is_train=False):
        super(Audio2Exp, self).__init__()
        self.cfg = cfg
        self.netG = netG

        self.is_train = is_train
        self.wav2lip = wav2lip
        self.coeff_enc = coeff_enc
        self.bfm = ParametricFaceModel(bfm_folder="checkpoints/BFM_Fitting")

    def test(self, batch):

        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]

        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10), 'audio2exp:'):  # every 10 frames

            current_mel_input = mel_input[:, i:i+10]

            # ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch['ref'][:, :, :64][:, i:i+10]
            ratio = batch['ratio_gt'][:, i:i+10]  # bs T

            # bs*T 1 80 16
            audiox = current_mel_input.view(-1, 1, 80, 16)
            curr_exp_coeff_pred = self.netG(
                audiox, ref, ratio)         # bs T 64

            exp_coeff_pred += [curr_exp_coeff_pred]

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': ops.cat(exp_coeff_pred, axis=1)
        }
        return results_dict

    def getloss(self, batch):

        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        pic_name = batch['pic_name']
        bs = mel_input.shape[0]
        T = mel_input.shape[1]

        img = cv2.imread(pic_name)
        img = np.asarray([cv2.resize(img, (96, 96))] * bs)

        img_masked = img.copy()
        img_masked[:, 96 // 2:] = 0
        img_input = np.concatenate((img_masked, img), axis=3) / 255.
        first_frame_img = ms.Tensor(np.transpose(
            img_input, (0, 3, 1, 2)), dtype=ms.float32)

        exp_coeff_pred = []
        wav2lip_coeff = []
        landmarks_ori = []
        landmarks_rep = []

        for i in tqdm(range(0, T, 10), 'audio2exp:'):  # every 10 frames

            current_mel_input = mel_input[:, i:i+10]

            # ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch['ref'][:, :, :64][:, i:i+10]
            ratio = batch['ratio_gt'][:, i:i+10]  # bs T

            # bs*T 1 80 16
            audiox = current_mel_input.view(-1, 1, 80, 16)
            curr_exp_coeff_pred = self.netG(
                audiox, ref, ratio)         # bs T 64

            exp_coeff_pred += [curr_exp_coeff_pred]

            # wav2lip
            curr_first_frame_img = first_frame_img.repeat(
                audiox.shape[0], axis=0)  # sample every 10 frames
            img_with_lip = self.wav2lip(
                audiox, curr_first_frame_img)  # T, 3, 96, 96
            full_coeff = self.coeff_enc(img_with_lip)
            coeffs = split_coeff(full_coeff)
            exp_coeffs = coeffs['exp']
            wav2lip_coeff += [exp_coeffs]

            # reconstruct coeffs
            landmarks = self.bfm.compute_for_render_landmarks(coeffs)
            landmarks_ori.append(landmarks)

            coeffs['exp'] = curr_exp_coeff_pred.squeeze(0)
            landmarks_new = self.bfm.compute_for_render_landmarks(coeffs)
            landmarks_rep.append(landmarks_new)

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': exp_coeff_pred,
            'wav2lip_coef': wav2lip_coeff,
            'landmarks_ori': landmarks_ori,
            'landmarks_rep': landmarks_rep,
            'ratio_gt': ratio,
        }

        return results_dict
