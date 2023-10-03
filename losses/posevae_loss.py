from mindspore import nn


class PoseVAELoss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        self.adversarial_loss = nn.BCELoss(reduction='mean')

    def construct(self, logits, labels):

        loss_mse = self.mse(logits, labels)
        loss_kl = self.kl(logits, labels)
        loss_gan = self.adversarial_loss(logits, labels)

        loss = loss_mse + loss_kl + 0.7 * loss_gan

        return loss