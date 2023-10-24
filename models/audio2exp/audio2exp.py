from tqdm import tqdm
import cv2
import copy
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from utils.preprocess import split_coeff
from models.face3d.bfm import ParametricFaceModel


class Audio2Exp(nn.Cell):
    """ ExpNet implementation (training)
    """

    def __init__(self, netG, cfg, wav2lip=None, coeff_enc=None, lipreading=None, is_train=False):
        super(Audio2Exp, self).__init__()
        self.cfg = cfg
        self.netG = netG

        self.is_train = is_train
        self.wav2lip = wav2lip
        self.coeff_enc = coeff_enc

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

    def construct(self, batch):
        current_mel_input = batch['indiv_mels']
        curr_ref = batch['ref']
        ratio = batch['ratio_gt']
        first_frame_img = batch['first_frame_img']

        # bs*T 1 80 16
        audiox = current_mel_input.view(-1, 1, 80, 16)
        curr_exp_coeff_pred = self.netG(
            audiox, curr_ref, ratio)         # bs T 64

        # wav2lip
        img_with_lip = self.wav2lip(
            audiox, first_frame_img)  # bs*T, 3, 96, 96
        full_coeff = self.coeff_enc(img_with_lip)

        return (curr_exp_coeff_pred, full_coeff, ratio)

    def construct_old(self, batch):

        indiv_mels = batch['indiv_mels']
        ref = batch['ref']
        ratio_gt = batch['ratio_gt']
        first_frame_img = batch['first_frame_img']
        T = batch['num_frames'].asnumpy()

        exp_coeff_pred = []
        wav2lip_coeff = []
        landmarks_ori = []
        landmarks_rep = []
        ratio = []

        for i in tqdm(range(0, T, 10), 'audio2exp:'):  # every 10 frames

            current_mel_input = indiv_mels[:, i:i+10]

            if current_mel_input.shape[1] == 10:
                # ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
                curr_ref = ref[:, :, :64][:, i:i+10]
                ratio = ratio_gt[:, i:i+10]  # bs T

                # bs*T 1 80 16
                audiox = current_mel_input.view(-1, 1, 80, 16)
                curr_exp_coeff_pred = self.netG(
                    audiox, curr_ref, ratio)         # bs T 64

                exp_coeff_pred += [curr_exp_coeff_pred]

                # wav2lip
                img_with_lip = self.wav2lip(
                    audiox, first_frame_img)  # T, 3, 96, 96
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
        results = (exp_coeff_pred, wav2lip_coeff,
                   landmarks_ori, landmarks_rep, ratio)

        return results
