from mindspore import nn
from models.lipreading.networks.resnet1d import Swish


class SELayer(nn.Cell):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.SequentialCell(
            nn.Dense(channel, channel // reduction, has_bias=False),
            Swish(),
            nn.Dense(channel // reduction, channel, has_bias=False),
            nn.Sigmoid()
        )

    def construct(self, x):
        b, c, T = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
