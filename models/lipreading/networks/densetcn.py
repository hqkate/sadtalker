import mindspore as ms
from mindspore import nn, ops
from collections import OrderedDict
from models.lipreading.networks.resnet1d import Swish
import numpy as np


class Chomp1d(nn.Cell):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"

    def construct(self, x):
        if self.chomp_size == 0:
            result = x
        if self.symm_chomp:
            result = x[:, :, self.chomp_size//2:-self.chomp_size//2]
        else:
            result = x[:, :, :-self.chomp_size]
        return result


class TemporalConvLayer(nn.Cell):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type):
        super(TemporalConvLayer, self).__init__()
        self.net = nn.SequentialCell(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, pad_mode='pad',
                      stride=stride, padding=padding, dilation=dilation, has_bias=True),
            nn.BatchNorm1d(n_outputs),
            Chomp1d(padding, True),
            nn.PReLU(channel=n_outputs) if relu_type == 'prelu' else Swish() if relu_type == 'swish' else nn.ReLU(),)

    def construct(self, x):
        return self.net(x)


class _ConvBatchChompRelu(nn.Cell):
    def __init__(self, n_inputs, n_outputs, kernel_size_set, stride, dilation, dropout, relu_type, se_module=False):
        super(_ConvBatchChompRelu, self).__init__()

        self.num_kernels = len(kernel_size_set)
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"

        for k_idx, k in enumerate(kernel_size_set):
            if se_module:
                from models.lipreading.networks.se_module import SELayer
                setattr(self, 'cbcr0_se_{}'.format(k_idx),
                        SELayer(n_inputs, reduction=16))
            cbcr = TemporalConvLayer(
                n_inputs, self.n_outputs_branch, k, stride, dilation, (k-1)*dilation, relu_type)
            setattr(self, 'cbcr0_{}'.format(k_idx), cbcr)
        self.dropout0 = nn.Dropout(p=dropout)
        for k_idx, k in enumerate(kernel_size_set):
            cbcr = TemporalConvLayer(
                n_outputs, self.n_outputs_branch, k, stride, dilation, (k-1)*dilation, relu_type)
            setattr(self, 'cbcr1_{}'.format(k_idx), cbcr)
        self.dropout1 = nn.Dropout(p=dropout)

        self.se_module = se_module
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1, has_bias=True) if n_inputs != n_outputs else None

        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(channel=n_outputs)
        elif relu_type == 'swish':
            self.relu_final = Swish()

    def bn_function(self, inputs):
        x = ops.cat(inputs, 1)
        outputs = []
        branch_se = getattr(self, 'cbcr0_se_0')
        for k_idx in range(self.num_kernels):
            if self.se_module:
                branch_se = getattr(self, 'cbcr0_se_{}'.format(k_idx))
            branch_convs = getattr(self, 'cbcr0_{}'.format(k_idx))
            if self.se_module:
                outputs.append(branch_convs(branch_se(x)))
            else:
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

    def construct(self, input):
        if isinstance(input, ms.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        bottleneck_output = self.bn_function(prev_features)
        return bottleneck_output


class _DenseBlock(nn.Cell):
    _version = 2

    def __init__(self, num_layers, num_input_features, growth_rate,
                 kernel_size_set, dilation_size_set,
                 dropout, relu_type, squeeze_excitation,
                 ):
        super(_DenseBlock, self).__init__()
        self.num_layers = num_layers

        for i in range(num_layers):
            dilation_size = dilation_size_set[i % len(dilation_size_set)]
            layer = _ConvBatchChompRelu(
                n_inputs=num_input_features + i * growth_rate,
                n_outputs=growth_rate,
                kernel_size_set=kernel_size_set,
                stride=1,
                dilation=dilation_size,
                dropout=dropout,
                relu_type=relu_type,
                se_module=squeeze_excitation,
            )

            setattr(self, 'denselayer%d' % (i + 1), layer)
            # self.add_module('denselayer%d' % (i + 1), layer)

    def construct(self, init_features):
        features = [init_features]

        layer1 = getattr(self, 'denselayer1')
        new_features = layer1(features)
        features.append(new_features)

        layer2 = getattr(self, 'denselayer2')
        new_features = layer2(features)
        features.append(new_features)

        layer3 = getattr(self, 'denselayer3')
        new_features = layer3(features)
        features.append(new_features)

        # for i in range(self.num_layers):
        #     layer = getattr(self, 'denselayer%d' % (i + 1))
        #     new_features = layer(features)
        #     features.append(new_features)

        output = ops.cat(features, 1).astype(ms.float32)
        return output


class _Transition(nn.SequentialCell):
    def __init__(self, num_input_features, num_output_features, relu_type):
        super(_Transition, self).__init__()
        self.append(nn.Conv1d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, has_bias=False))
        self.append(nn.BatchNorm1d(num_output_features))
        if relu_type == 'relu':
            self.append(nn.ReLU())
        elif relu_type == 'prelu':
            self.append(nn.PReLU(
                channel=num_output_features))
        elif relu_type == 'swish':
            self.append(Swish())


class DenseTemporalConvNet(nn.Cell):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size,
                 kernel_size_set, dilation_size_set,
                 dropout=0.2, relu_type='prelu',
                 squeeze_excitation=False,
                 ):
        super(DenseTemporalConvNet, self).__init__()
        features = nn.SequentialCell()

        trans = _Transition(num_input_features=input_size,
                            num_output_features=reduced_size,
                            relu_type='prelu')
        # self.features.add_module('transition%d' % (0), trans)
        features.append(trans)
        num_features = reduced_size

        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate_set[i],
                kernel_size_set=kernel_size_set,
                dilation_size_set=dilation_size_set,
                dropout=dropout,
                relu_type=relu_type,
                squeeze_excitation=squeeze_excitation,
            )
            # self.features.add_module('denseblock%d' % (i + 1), block)
            features.append(block)
            num_features = num_features + num_layers * growth_rate_set[i]

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=reduced_size,
                                    relu_type=relu_type)
                # self.features.add_module('transition%d' % (i + 1), trans)
                features.append(trans)
                num_features = reduced_size

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        features.append(nn.BatchNorm1d(num_features))
        self.features = features

    def construct(self, x):
        features = self.features(x)
        return features
