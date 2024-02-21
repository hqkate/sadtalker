from mindspore import nn


def _deconv_output_length(
    is_valid,
    is_same,
    is_pad,
    input_length,
    filter_size,
    stride_size,
    dilation_size,
    padding,
):
    """Calculate the width and height of output."""
    length = 0
    filter_size = filter_size + (filter_size - 1) * (dilation_size - 1)
    if is_valid:
        if filter_size - stride_size > 0:
            length = input_length * stride_size + filter_size - stride_size
        else:
            length = input_length * stride_size
    elif is_same:
        length = input_length * stride_size
    elif is_pad:
        length = input_length * stride_size - padding + filter_size - stride_size

    return length


class Conv2d(nn.Cell):
    def __init__(
        self,
        cin,
        cout,
        kernel_size,
        stride,
        padding,
        use_residual=False,
        use_act=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if padding == 0:
            pad_mode = "valid"
        else:
            pad_mode = "pad"

        self.conv_block = nn.SequentialCell(
            nn.Conv2d(
                cin,
                cout,
                kernel_size,
                stride,
                pad_mode=pad_mode,
                padding=padding,
                has_bias=True,
            ),
            nn.BatchNorm2d(cout, momentum=0.9),
        )
        self.act = nn.ReLU()
        self.use_residual = use_residual
        self.use_act = use_act

    def construct(self, x):
        out = self.conv_block(x)
        if self.use_residual:
            out = out + x

        if self.use_act:
            return self.act(out)
        else:
            return out


class nonorm_Conv2d(nn.Cell):
    def __init__(
        self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.SequentialCell(
            nn.Conv2d(cin, cout, kernel_size, stride, padding=padding, has_bias=True),
        )
        self.act = nn.LeakyReLU(0.01)

    def construct(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dTransposeTorch(nn.Conv2dTranspose):
    """Conv2dTransposeTorch

    Align the output_padding with pyTorch.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        pad_mode="same",
        padding=0,
        output_padding=0,
        dilation=1,
        group=1,
        has_bias=False,
        weight_init=None,
        bias_init=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            group=group,
            has_bias=has_bias,
            weight_init=weight_init,
            bias_init=bias_init,
        )

    def construct(self, x):
        n, _, h, w = self.shape(x)
        h_out = _deconv_output_length(
            self.is_valid,
            self.is_same,
            self.is_pad,
            h,
            self.kernel_size[0],
            self.stride[0],
            self.dilation[0],
            self.padding_top + self.padding_bottom,
        )
        w_out = _deconv_output_length(
            self.is_valid,
            self.is_same,
            self.is_pad,
            w,
            self.kernel_size[1],
            self.stride[1],
            self.dilation[1],
            self.padding_left + self.padding_right,
        )

        if isinstance(self.output_padding, tuple):
            if self.output_padding[0] < 0 or self.output_padding[0] >= max(
                self.dilation[0], self.stride[0]
            ):
                raise ValueError(
                    "output_padding[0] must be in range of [0, max(stride_h, dilation_h))."
                )
            if self.output_padding[1] < 0 or self.output_padding[1] >= max(
                self.dilation[1], self.stride[1]
            ):
                raise ValueError(
                    "output_padding[1] must be in range of [0, max(stride_w, dilation_w))."
                )
            if not self.is_pad and (
                self.output_padding[0] > 0 or self.output_padding[1] > 0
            ):
                raise ValueError(
                    "when output_padding is not zero, pad_mode must be 'pad'"
                )

            h_out += self.output_padding[0]
            w_out += self.output_padding[1]

        if not isinstance(self.output_padding, tuple) and self.output_padding != 0:
            h_out += self.output_padding
            w_out += self.output_padding

        conv2d_trans_ret = self.conv2d_transpose(
            x, self.weight, (n, self.out_channels, h_out, w_out)
        )
        if self.has_bias:
            conv2d_trans_ret = self.bias_add(conv2d_trans_ret, self.bias)

        return conv2d_trans_ret


class Conv2dTranspose(nn.Cell):
    def __init__(
        self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if padding == 0:
            pad_mode = "valid"
        else:
            pad_mode = "pad"

        self.conv_block = nn.SequentialCell(
            nn.Conv2dTranspose(
                cin,
                cout,
                kernel_size,
                stride,
                pad_mode=pad_mode,
                padding=padding,
                output_padding=output_padding,
                has_bias=True,
            ),
            nn.BatchNorm2d(cout, momentum=0.9),
        )

        self.act = nn.ReLU()

    def construct(self, x):
        out = self.conv_block(x)
        return self.act(out)
