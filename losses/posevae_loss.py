import mindspore as ms
from mindspore import nn, ops


class VAEDiscriminatorLoss(nn.Cell):
    """Combined loss"""

    def __init__(self, criterion=nn.BCELoss(reduction="mean")):
        super(VAEDiscriminatorLoss, self).__init__()
        self.criterion = criterion
        self.real_target = ms.Tensor(1.0, ms.float32)
        self.fake_target = ms.Tensor(0.0, ms.float32)

    def construct(self, net_output, ground_truth, mask, edge, gray_image):
        real_pred, real_pred_edge = net_output.get("ground_truth")
        fake_pred, fake_pred_edge = net_output.get("fake")

        real_target = self.real_target.expand_as(real_pred)
        fake_target = self.fake_target.expand_as(fake_pred)

        loss_adversarial = (
            self.criterion(real_pred, real_target)
            + self.criterion(fake_pred, fake_target)
            + self.criterion(real_pred_edge, edge)
            + self.criterion(fake_pred_edge, edge)
        )
        return loss_adversarial


class VAEGeneratorLoss(nn.Cell):
    """Generator Loss"""

    def __init__(
        self,
        hole_loss_w,
        valid_loss_w,
        perceptual_loss_w,
        style_loss_w,
        adversarial_loss_w,
        intermediate_loss_w,
        criterion=nn.BCELoss(reduction="mean"),
        **kwargs
    ):
        super(VAEGeneratorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.criterion = criterion
        if get_current_device() == "ascend":
            self.gram_matrix = GramMat().to_float(dtype.float16)
        else:
            self.gram_matrix = gram_matrix
        self.hole_loss_w = hole_loss_w
        self.valid_loss_w = valid_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.style_loss_w = style_loss_w
        self.adversarial_loss_w = adversarial_loss_w
        self.intermediate_loss_w = intermediate_loss_w

    def construct(self, net_output, ground_truth, mask, edge, gray_image):
        output, projected_image, projected_edge = net_output.get("generator")
        output_pred, output_edge = net_output.get("discriminator")
        vgg_comp, vgg_output, vgg_ground_truth = net_output.get("vgg_feat_extractor")

        loss_hole = self.l1((1 - mask) * output, (1 - mask) * ground_truth)

        loss_valid = self.l1(mask * output, mask * ground_truth)
        loss_perceptual = 0.0
        for i in range(3):
            loss_perceptual += self.l1(vgg_output[i], vgg_ground_truth[i])
            loss_perceptual += self.l1(vgg_comp[i], vgg_ground_truth[i])

        loss_style = 0.0
        for i in range(3):
            mats = ops.Concat(axis=0)((vgg_ground_truth[i], vgg_output[i], vgg_comp[i]))
            gram = self.gram_matrix(mats)
            gram_gt, gram_out, gram_comp = ops.Split(axis=0, output_num=3)(gram)
            loss_style += self.l1(gram_out, gram_gt)
            loss_style += self.l1(gram_comp, gram_gt)

        real_target = self.real_target.expand_as(output_pred)
        loss_adversarial = self.criterion(output_pred, real_target) + self.criterion(output_edge, edge)

        loss_intermediate = self.criterion(projected_edge, edge) + self.l1(projected_image, ground_truth)

        loss_g = (
            loss_hole.mean() * self.hole_loss_w
            + loss_valid.mean() * self.valid_loss_w
            + loss_perceptual.mean() * self.perceptual_loss_w
            + loss_style.mean() * self.style_loss_w
            + loss_adversarial.mean() * self.adversarial_loss_w
            + loss_intermediate.mean() * self.intermediate_loss_w
        )

        result = ops.stop_gradient(output)
        return loss_g, result


class GANLoss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)

    def construct(self, pose_motion_pred, pose_motion_gt):
        pass


class PoseVAELoss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        self.adversarial_loss = nn.BCELoss(reduction='mean')

    def construct(self, batch):

        logits = batch['pose_motion_pred']
        labels = batch['pose_motion_gt']

        mu = batch['mu']
        logvar = batch['logvar']
        std = ops.Exp()(0.5 * logvar)

        loss_mse = self.mse(logits, labels)
        loss_kl = ops.sum(mu ** 2 + std ** 2 - ops.log(std) - 0.5)
        loss_gan = self.adversarial_loss(logits, labels)

        loss = loss_mse + loss_kl + 0.7 * loss_gan

        return loss