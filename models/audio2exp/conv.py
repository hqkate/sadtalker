from mindspore import nn


class Conv2d(nn.Cell):
    def __init__(self, cin, cout, kernel_size, stride, padding, use_residual=False, use_act=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if padding == 0:
            pad_mode = 'valid'
        else:
            pad_mode = 'pad'

        self.conv_block = nn.SequentialCell(
            nn.Conv2d(cin, cout, kernel_size, stride, pad_mode=pad_mode,
                      padding=padding, has_bias=True),
            nn.BatchNorm2d(cout, momentum=0.9)
        )
        self.act = nn.ReLU()
        self.use_residual = use_residual
        self.use_act = use_act

    def construct(self, x):
        out = self.conv_block(x)
        if self.use_residual:
            out += x

        if self.use_act:
            return self.act(out)
        else:
            return out


class nonorm_Conv2d(nn.Cell):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.SequentialCell(
            nn.Conv2d(cin, cout, kernel_size, stride,
                      padding=padding, has_bias=True),
        )
        self.act = nn.LeakyReLU(0.01)

    def construct(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dTranspose(nn.Cell):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if padding == 0:
            pad_mode = 'valid'
        else:
            pad_mode = 'pad'

        self.conv_block = nn.SequentialCell(
            nn.Conv2dTranspose(
                cin, cout, kernel_size, stride,
                pad_mode=pad_mode, padding=padding, output_padding=output_padding, has_bias=True
            ),
            nn.BatchNorm2d(cout)
        )

        self.act = nn.ReLU()

    def construct(self, x):
        out = self.conv_block(x)
        return self.act(out)
