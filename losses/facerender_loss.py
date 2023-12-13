import numpy as np
import mindspore as ms
from mindspore import nn, ops

from utils.preprocess import split_coeff
from models.face3d.bfm import ParametricFaceModel
from models.lipreading import get_lipreading_model
from models.external.face3d.face_renderer import renderer


class FRLoss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.loss_fn = nn.MSELoss()

    def construct(self, data_batch):

        # (id_coeffs, exp_coeffs, tex_coeffs, angles, gammas, translations)
        coeffs = split_coeff(wav2lip_coeff)
        exp_coeff_wav2lip = coeffs[1]

        new_coeffs = (
            coeffs[0],
            exp_coeff_pred.view(-1, 64),
            coeffs[2],
            coeffs[3],
            coeffs[4],
            coeffs[5]
        )

        # distill loss (lip-only coefficients, MSE)
        loss_distill = self.distill_loss(
            exp_coeff_pred.view(-1, 64), exp_coeff_wav2lip)

        # landmarks loss (eyes)
        render_results_1 = self.bfm1.compute_for_render_new(coeffs)  # bs*T, 68, 2
        landmarks_w2l = render_results_1[-1]

        render_results_2 = self.bfm2.compute_for_render_new(new_coeffs)
        landmarks_rep = render_results_2[-1]

        face_vertex = render_results_2[0]
        face_texture = render_results_2[1]
        face_color = render_results_2[2]
        face_proj = render_results_2[3]

        # landmarks_w2l = ms.Tensor(np.load("landmarks_w2l.npy"), ms.float32)
        # landmarks_rep = ms.Tensor(np.load("landmarks_rep.npy"), ms.float32)
        loss_lks = self.lks_loss(landmarks_w2l, landmarks_rep, ratio_gt)

        # lip-reading loss (cross-entropy)
        loss_read = self.lread_loss(audio_wav, face_vertex, face_color, self.bfm1.triangle, face_proj, landmarks_rep)

        loss = 2.0 * loss_distill + 0.01 * loss_lks + 0.01 * loss_read

        return loss