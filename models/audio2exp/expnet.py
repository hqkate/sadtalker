from mindspore import nn, ops
from mindspore.common.initializer import TruncatedNormal


class Conv(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, use_bn=False, use_residual=False, pad_mode='pad', padding=0):
        super(Conv, self).__init__()

        self.Relu = nn.ReLU()
        self.conv = nn.set_train(False)(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, pad_mode=pad_mode,
                              has_bias=True, bias_init=TruncatedNormal(), weight_init=TruncatedNormal(0.02))

        if use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channel, eps=1e-4, momentum=0.9,
                                     gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

        self.use_bn = use_bn
        self.use_residual = use_residual

    def construct(self, x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        if self.use_residual:
            out += x
        out = self.Relu(out)
        return out


class ExpNet(nn.Cell):
    """ ExpNet implementation (inference)
    """
    def __init__(self, wav2lip=None):
        super().__init__()
        self.audio_encoder = nn.SequentialCell(
            Conv(1, 32, kernel_size=3, stride=1, padding=1),
            Conv(32, 32, kernel_size=3, stride=1, padding=1, use_residual=True),
            Conv(32, 32, kernel_size=3, stride=1, padding=1, use_residual=True),

            Conv(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv(64, 64, kernel_size=3, stride=1, padding=1, use_residual=True),
            Conv(64, 64, kernel_size=3, stride=1, padding=1, use_residual=True),

            Conv(64, 128, kernel_size=3, stride=3, padding=1),
            Conv(128, 128, kernel_size=3, stride=1, padding=1, use_residual=True),
            Conv(128, 128, kernel_size=3, stride=1, padding=1, use_residual=True),

            Conv(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv(256, 256, kernel_size=3, stride=1, padding=1, use_residual=True),

            Conv(256, 512, kernel_size=3, stride=1, padding=0),
            Conv(512, 512, kernel_size=1, stride=1, padding=0),
            )

        self.wav2lip = wav2lip
        self.mapping1 = nn.Dense(512+64+1, 64, bias_init="zeros")
        # nn.init.constant_(self.mapping1.bias, 0.)

    def construct(self, x, ref, ratio):
        x = self.audio_encoder(x).view(x.shape[0], -1)
        ref_reshape = ref.reshape(x.shape[0], -1)
        ratio = ratio.reshape(x.shape[0], -1)

        y = self.mapping1(ops.cat([x, ref_reshape, ratio], axis=1))
        out = y.reshape(ref.shape[0], ref.shape[1], -1) #+ ref # resudial
        return out
