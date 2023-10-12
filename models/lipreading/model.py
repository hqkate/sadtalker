#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import mindspore as ms
from mindspore import nn, ops

from models.lipreading.networks.resnet import ResNet, BasicBlock
from models.lipreading.networks.resnet1d import ResNet1D, BasicBlock1D, Swish
# from models.lipreading.networks.shufflenetv2 import ShuffleNetV2
from models.lipreading.networks.tcn import MultibranchTemporalConvNet, TemporalConvNet
from models.lipreading.networks.densetcn import DenseTemporalConvNet


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = ops.transpose(x, (0, 2, 1, 3, 4))
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    return ops.stack([ops.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)


class MultiscaleMultibranchTCN(nn.Cell):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len(self.kernel_sizes)

        self.mb_ms_tcn = MultibranchTemporalConvNet(
            input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Dense(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

    def construct(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = ops.transpose(x, (0, 2, 1))
        out = self.mb_ms_tcn(xtrans)
        out = self.consensus_func(out, lengths)
        return self.tcn_output(out)


class TCN(nn.Cell):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(
            input_size, num_channels, dropout=dropout, tcn_options=tcn_options, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Dense(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

        self.has_aux_losses = False

    def construct(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(ops.transpose(x, (0, 2, 1)))
        x = self.consensus_func(x, lengths, B)
        return self.tcn_output(x)


class DenseTCN(nn.Cell):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
                 kernel_size_set, dilation_size_set,
                 dropout, relu_type,
                 squeeze_excitation=False,
                 ):
        super(DenseTCN, self).__init__()

        num_features = reduced_size + block_config[-1]*growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet(block_config, growth_rate_set, input_size, reduced_size,
                                              kernel_size_set, dilation_size_set,
                                              dropout=dropout, relu_type=relu_type,
                                              squeeze_excitation=squeeze_excitation,
                                              )
        self.tcn_output = nn.Dense(num_features, num_classes)
        self.consensus_func = _average_batch

    def construct(self, x, lengths, B):
        x = self.tcn_trunk(ops.transpose(x, (0, 2, 1)))
        x = self.consensus_func(x, lengths, B)
        return self.tcn_output(x)


class Lipreading(nn.Cell):
    def __init__(self, modality='video', hidden_dim=256, backbone_type='resnet', num_classes=500,
                 relu_type='prelu', tcn_options={}, densetcn_options={}, width_mult=1.0,
                 use_boundary=False, extract_feats=False):
        super(Lipreading, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.modality = modality
        self.use_boundary = use_boundary

        if self.modality == 'audio':
            self.frontend_nout = 1
            self.backend_out = 512
            self.trunk = ResNet1D(
                BasicBlock1D, [2, 2, 2, 2], relu_type=relu_type)
        elif self.modality == 'video':
            if self.backbone_type == 'resnet':
                self.frontend_nout = 64
                self.backend_out = 512
                self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
            else:
                raise NotImplementedError(
                    f"unsupported backbone type {self.backbone_type}.")
            # elif self.backbone_type == 'shufflenet':
            #     assert width_mult in [0.5, 1.0, 1.5,
            #                           2.0], "Width multiplier not correct"
            #     shufflenet = ShuffleNetV2(input_size=96, width_mult=width_mult)
            #     self.trunk = nn.SequentialCell(
            #         shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
            #     self.frontend_nout = 24
            #     self.backend_out = 1024 if width_mult != 2.0 else 2048
            #     self.stage_out_channels = shufflenet.stage_out_channels[-1]

            # -- frontend3D
            if relu_type == 'relu':
                frontend_relu = nn.ReLU()
            elif relu_type == 'prelu':
                frontend_relu = nn.PReLU(self.frontend_nout)
            elif relu_type == 'swish':
                frontend_relu = Swish()

            self.frontend3D = nn.SequentialCell(
                nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7),
                          stride=(1, 2, 2), pad_mode='pad', padding=(2, 2, 3, 3, 3, 3), has_bias=False),
                nn.BatchNorm3d(self.frontend_nout, eps=1e-5),
                frontend_relu,
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), pad_mode='pad', padding=(0, 1, 1)))
        else:
            raise NotImplementedError

        if tcn_options:
            tcn_class = TCN if len(
                tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
            self.tcn = tcn_class(input_size=self.backend_out,
                                 num_channels=[
                                     hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers'],
                                 num_classes=num_classes,
                                 tcn_options=tcn_options,
                                 dropout=tcn_options['dropout'],
                                 relu_type=relu_type,
                                 dwpw=tcn_options['dwpw'],
                                 )
        elif densetcn_options:
            self.tcn = DenseTCN(block_config=densetcn_options['block_config'],
                                growth_rate_set=densetcn_options['growth_rate_set'],
                                input_size=self.backend_out if not self.use_boundary else self.backend_out+1,
                                reduced_size=densetcn_options['reduced_size'],
                                num_classes=num_classes,
                                kernel_size_set=densetcn_options['kernel_size_set'],
                                dilation_size_set=densetcn_options['dilation_size_set'],
                                dropout=densetcn_options['dropout'],
                                relu_type=relu_type,
                                squeeze_excitation=densetcn_options['squeeze_excitation'],
                                )
        else:
            raise NotImplementedError

    def construct(self, x, lengths, boundaries=None):
        B = 1
        if self.modality == 'video':
            B, C, T, H, W = x.shape
            x = self.frontend3D(x)
            Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
            x = threeD_to_2D_tensor(x)
            x = self.trunk(x)

            if self.backbone_type == 'shufflenet':
                x = x.view(-1, self.stage_out_channels)
            x = x.view(B, Tnew, x.shape[1])
        elif self.modality == 'audio':
            B, C, T = x.shape
            x = self.trunk(x)
            x = ops.transpose(x, (0, 2, 1))
            lengths = [_//640 for _ in lengths]

        # -- duration
        if self.use_boundary:
            x = ops.cat([x, boundaries], axis=-1)

        return x if self.extract_feats else self.tcn(x, lengths, B)
