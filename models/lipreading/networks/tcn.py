from mindspore import nn, ops
from models.lipreading.networks.resnet1d import Swish


class Chomp1d(nn.Cell):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"

    def construct(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2]
        else:
            return x[:, :, :-self.chomp_size]


class ConvBatchChompRelu(nn.Cell):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type, dwpw=False):
        super(ConvBatchChompRelu, self).__init__()
        self.dwpw = dwpw
        if dwpw:
            self.conv = nn.SequentialCell(
                # -- dw
                nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=stride, pad_mode='pad',
                          padding=padding, dilation=dilation, groups=n_inputs, has_bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(channel=n_inputs) if relu_type == 'prelu' else nn.ReLU(),
                # -- pw
                nn.Conv1d(n_inputs, n_outputs, 1, 1, 0, has_bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(channel=n_outputs) if relu_type == 'prelu' else nn.ReLU()
            )
        else:
            self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                  stride=stride, pad_mode='pad', padding=padding, dilation=dilation)
            self.batchnorm = nn.BatchNorm1d(n_outputs)
            self.chomp = Chomp1d(padding, True)
            self.non_lin = nn.PReLU(channel=n_outputs) if relu_type == 'prelu' else Swish(
            ) if relu_type == 'swish' else nn.ReLU()

    def construct(self, x):
        if self.dwpw:
            return self.conv(x)
        else:
            out = self.conv(x)
            out = self.batchnorm(out)
            out = self.chomp(out)
            return self.non_lin(out)


class MultibranchTemporalBlock(nn.Cell):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2,
                 relu_type='relu', dwpw=False):
        super(MultibranchTemporalBlock, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = ConvBatchChompRelu(n_inputs, self.n_outputs_branch,
                                      k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw)
            setattr(self, 'cbcr0_{}'.format(k_idx), cbcr)
        self.dropout0 = nn.Dropout(p=dropout)

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = ConvBatchChompRelu(n_outputs, self.n_outputs_branch,
                                      k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw)
            setattr(self, 'cbcr1_{}'.format(k_idx), cbcr)
        self.dropout1 = nn.Dropout(p=dropout)

        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (
            n_inputs//self.num_kernels) != n_outputs else None

        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(channel=n_outputs)
        elif relu_type == 'swish':
            self.relu_final = Swish()

    def construct(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr0_{}'.format(k_idx))
            outputs.append(branch_convs(x))
        out0 = ops.cat(outputs, 1)
        out0 = self.dropout0(out0)

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr1_{}'.format(k_idx))
            outputs.append(branch_convs(out0))
        out1 = ops.cat(outputs, 1)
        out1 = self.dropout1(out1)

        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_final(out1 + res)


class MultibranchTemporalConvNet(nn.Cell):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(MultibranchTemporalConvNet, self).__init__()

        self.ksizes = tcn_options['kernel_size']

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            padding = [(s-1)*dilation_size for s in self.ksizes]
            layers.append(MultibranchTemporalBlock(in_channels, out_channels, self.ksizes,
                                                   stride=1, dilation=dilation_size, padding=padding, dropout=dropout, relu_type=relu_type,
                                                   dwpw=dwpw))

        self.network = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.network(x)
# --------------------------------


# --------------- STANDARD VERSION (SINGLE BRANCH) ------------------------
class TemporalBlock(nn.Cell):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,
                 symm_chomp=False, no_padding=False, relu_type='relu', dwpw=False):
        super(TemporalBlock, self).__init__()

        self.no_padding = no_padding
        if self.no_padding:
            downsample_chomp_size = 2*padding-4
            padding = 1  # hack-ish thing so that we can use 3 layers

        if dwpw:
            self.net = nn.SequentialCell(
                # -- first conv set within block
                # -- dw
                nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=stride, pad_mode='pad',
                          padding=padding, dilation=dilation, groups=n_inputs, has_bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(channel=n_inputs) if relu_type == 'prelu' else Swish(
                ) if relu_type == 'swish' else nn.ReLU(),
                # -- pw
                nn.Conv1d(n_inputs, n_outputs, 1, 1, 0, has_bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(channel=n_outputs) if relu_type == 'prelu' else Swish(
                ) if relu_type == 'swish' else nn.ReLU(),
                nn.Dropout(p=dropout),
                # -- second conv set within block
                # -- dw
                nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, pad_mode='pad',
                          padding=padding, dilation=dilation, groups=n_outputs, has_bias=False),
                nn.BatchNorm1d(n_outputs),
                Chomp1d(padding, True),
                nn.PReLU(channel=n_outputs) if relu_type == 'prelu' else Swish(
                ) if relu_type == 'swish' else nn.ReLU(),
                # -- pw
                nn.Conv1d(n_outputs, n_outputs, 1, 1, 0, has_bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(channel=n_outputs) if relu_type == 'prelu' else Swish(
                ) if relu_type == 'swish' else nn.ReLU(),
                nn.Dropout(p=dropout),
            )
        else:
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                   stride=stride, pad_mode='pad', padding=padding, dilation=dilation)
            self.batchnorm1 = nn.BatchNorm1d(n_outputs)
            self.chomp1 = Chomp1d(
                padding, symm_chomp) if not self.no_padding else None
            if relu_type == 'relu':
                self.relu1 = nn.ReLU()
            elif relu_type == 'prelu':
                self.relu1 = nn.PReLU(channel=n_outputs)
            elif relu_type == 'swish':
                self.relu1 = Swish()
            self.dropout1 = nn.Dropout(p=dropout)

            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, pad_mode='pad',
                                   stride=stride, padding=padding, dilation=dilation)
            self.batchnorm2 = nn.BatchNorm1d(n_outputs)
            self.chomp2 = Chomp1d(
                padding, symm_chomp) if not self.no_padding else None
            if relu_type == 'relu':
                self.relu2 = nn.ReLU()
            elif relu_type == 'prelu':
                self.relu2 = nn.PReLU(channel=n_outputs)
            elif relu_type == 'swish':
                self.relu2 = Swish()
            self.dropout2 = nn.Dropout(p=dropout)

            if self.no_padding:
                self.net = nn.SequentialCell(self.conv1, self.batchnorm1, self.relu1, self.dropout1,
                                             self.conv2, self.batchnorm2, self.relu2, self.dropout2)
            else:
                self.net = nn.SequentialCell(self.conv1, self.batchnorm1, self.chomp1, self.relu1, self.dropout1,
                                             self.conv2, self.batchnorm2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if self.no_padding:
            self.downsample_chomp = Chomp1d(downsample_chomp_size, True)
        if relu_type == 'relu':
            self.relu = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu = nn.PReLU(channel=n_outputs)
        elif relu_type == 'swish':
            self.relu = Swish()

    def construct(self, x):
        out = self.net(x)
        if self.no_padding:
            x = self.downsample_chomp(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Cell):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(TemporalConvNet, self).__init__()
        self.ksize = tcn_options['kernel_size'][0] if isinstance(
            tcn_options['kernel_size'], list) else tcn_options['kernel_size']
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, self.ksize, stride=1, dilation=dilation_size,
                                        padding=(self.ksize-1) * dilation_size, dropout=dropout, symm_chomp=True,
                                        no_padding=False, relu_type=relu_type, dwpw=dwpw))

        self.network = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.network(x)
# --------------------------------
