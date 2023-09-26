from mindspore import nn, ops
import mindspore as ms
import numpy as np
import logging
from models.audio2exp.conv import Conv2d


class ExpNet(nn.Cell):
    """ ExpNet implementation (inference)
    """

    def __init__(self, wav2lip=None):
        super().__init__()
        self.audio_encoder = nn.SequentialCell(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1,
                 padding=1, use_residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1,
                 padding=1, use_residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1,
                 padding=1, use_residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1,
                 padding=1, use_residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1,
                 padding=1, use_residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1,
                 padding=1, use_residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1,
                 padding=1, use_residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.wav2lip = wav2lip
        self.mapping1 = nn.Dense(512+64+1, 64, bias_init="zeros")

    def construct(self, x, ref, ratio):

        x = self.audio_encoder(x).view(x.shape[0], -1)
        ref_reshape = ref.reshape(x.shape[0], -1)
        ratio = ratio.reshape(x.shape[0], -1)

        y = self.mapping1(ops.cat([x, ref_reshape, ratio], axis=1))
        out = y.reshape(ref.shape[0], ref.shape[1], -1)  # + ref # resudial
        return out
