from mindspore import nn, ops


class LandmarksLoss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.mse = nn.MSELoss()

    def get_eye_ratio(self, points):
        # (B, 68, 2)
        width = (ops.square(points[:, 39, 0] - points[:, 36, 0]) +
                 ops.square(points[:, 45, 0] - points[:, 42, 0])) / 2.0
        height = (ops.square(points[:, 37, 0] + points[:, 38, 0] - points[:, 40, 0] - points[:, 41, 0]) +
                  ops.square(points[:, 43, 0] + points[:, 44, 0] - points[:, 46, 0] - points[:, 47, 0])) / 2.0
        ratio = height / width
        return ratio  # [B]

    def get_eye_loss(self, lks, z_blink):
        eye_ratio = self.get_eye_ratio(lks)
        loss = eye_ratio - z_blink
        return loss

    def construct(self, landmarks_ori, landmarks_rep, ratio_gt):

        loss_eyes = []
        loss_points = []

        for lks_ori, lks_rep in zip(landmarks_ori, landmarks_rep):
            loss_eye = self.get_eye_loss(lks_rep, ratio_gt)
            loss_point = self.mse(lks_ori, lks_rep)

            loss_eyes.append(loss_eye)
            loss_points.append(loss_point)

        loss = 200 * ops.sum(loss_eyes) + ops.mean(loss_points)

        return loss


class LipReadingLoss(nn.LossBase):
    pass


class ExpNetLoss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.distill_loss = nn.MSELoss()
        self.lks_loss = LandmarksLoss()
        self.lread_loss = LipReadingLoss()

    def construct(self, exp_coeff_pred, wav2lip_coeff, landmarks_ori, landmarks_rep, ratio_gt):

        # distill loss (lip-only coefficients, MSE)
        loss_distill = ops.mean([self.distill_loss(
            pred, logit) for pred, logit in zip(exp_coeff_pred, wav2lip_coeff)])

        # landmarks loss (eyes)
        loss_lks = self.lks_loss(landmarks_ori, landmarks_rep, ratio_gt)

        # lip-reading loss (cross-entropy)
        loss_read = self.lread_loss(logits, labels)

        loss = 2 * loss_distill + 0.01 * loss_lks + 0.01 * loss_read

        return loss
