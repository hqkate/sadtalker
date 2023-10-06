import numpy as np
import mindspore as ms
from mindspore import nn, ops
from utils.preprocess import split_coeff
from models.face3d.bfm import ParametricFaceModel


class LandmarksLoss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.cast = ops.Cast()

    def get_eye_ratio(self, points):
        # (B, 68, 2)

        width = (ops.dist(points[:, 39, :], points[:, 36, :]) +
                 ops.dist(points[:, 45, :], points[:, 42, :])) / 2.0

        height = (ops.dist(points[:, 37, :], points[:, 40, :]) +
                  ops.dist(points[:, 38, :], points[:, 41, :]) +
                  ops.dist(points[:, 43, :], points[:, 46, :]) +
                  ops.dist(points[:, 44, :], points[:, 47, :])) / 4.0

        ratio = ops.div(height, width)
        return ratio  # [B]

    def get_eye_loss(self, lks, z_blink):
        eye_ratio = self.get_eye_ratio(lks)
        loss = ops.dist(eye_ratio, z_blink.view(-1), p=1)
        return loss

    def construct(self, landmarks_ori, landmarks_rep, ratio_gt):
        """
        args:
            landmarks_ori: T, 68, 2
            landmarks_rep: T, 68, 2
            ratio_gt: bs, T, 1
        """
        loss_eye = self.get_eye_loss(landmarks_rep, ratio_gt)
        loss_point = ops.mean(ops.dist(landmarks_ori, landmarks_rep, 2))

        loss = 200.0 * loss_eye + loss_point

        return loss


class LipReadingLoss(nn.LossBase):
    pass


class ExpNetLoss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.distill_loss = nn.MSELoss()
        self.lks_loss = LandmarksLoss()
        # self.lread_loss = LipReadingLoss()

        self.bfm1 = ParametricFaceModel(bfm_folder="checkpoints/BFM_Fitting")
        self.bfm2 = ParametricFaceModel(bfm_folder="checkpoints/BFM_Fitting")

        self.cast = ops.Cast()

    def construct(self, exp_coeff_pred, wav2lip_coeff, ratio_gt
                  ):

        # (id_coeffs, exp_coeffs, tex_coeffs, angles, gammas, translations)
        coeffs = split_coeff(wav2lip_coeff)
        exp_coeff_wav2lip = coeffs[1]

        # reconstruct coeffs
        landmarks_ori = self.bfm1.compute_for_render_landmarks(
            coeffs)  # bs*T, 68, 2

        new_coeffs = (
            coeffs[0],
            exp_coeff_pred.view(-1, 64),
            coeffs[2],
            coeffs[3],
            coeffs[4],
            coeffs[5]
        )

        landmarks_rep = self.bfm2.compute_for_render_landmarks(new_coeffs)

        # distill loss (lip-only coefficients, MSE)
        loss_distill = self.distill_loss(
            exp_coeff_pred.view(-1, 64), exp_coeff_wav2lip)

        # landmarks loss (eyes)
        loss_lks = self.lks_loss(landmarks_ori, landmarks_rep, ratio_gt)

        # lip-reading loss (cross-entropy)
        # loss_read = self.lread_loss(logits, labels)
        loss_read = 0.0

        loss = 2.0 * loss_distill + 0.01 * loss_lks + 0.01 * loss_read

        return loss


class DebugLoss(nn.LossBase):
    def __init__(self):
        super().__init__()
        self.distill_loss = nn.MSELoss()
        self.bfm1 = ParametricFaceModel(bfm_folder="checkpoints/BFM_Fitting")
        self.bfm2 = ParametricFaceModel(bfm_folder="checkpoints/BFM_Fitting")

    def construct(self, exp_coeff_pred, wav2lip_coeff, ratio_gt
                  ):

        # (id_coeffs, exp_coeffs, tex_coeffs, angles, gammas, translations)
        coeffs = split_coeff(wav2lip_coeff)
        exp_coeff_wav2lip = coeffs[1]
        exp_coeff_pred = exp_coeff_pred.view(-1, 64)

        # reconstruct coeffs
        landmarks_ori = self.bfm1.compute_for_render_landmarks(
            coeffs)  # bs*T, 68, 2

        new_coeffs = (
            coeffs[0],
            exp_coeff_pred,
            coeffs[2],
            coeffs[3],
            coeffs[4],
            coeffs[5]
        )

        landmarks_rep = self.bfm2.compute_for_render_landmarks(new_coeffs)

        # distill loss (lip-only coefficients, MSE)
        loss_distill = self.distill_loss(
            exp_coeff_pred, exp_coeff_wav2lip)

        return loss_distill
