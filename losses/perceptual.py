# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md

import mindspore as ms
from mindspore import nn, ops
from models.facerender.vgg import vgg19


def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input back to [0, 1]
    normalized_input = (input + 1) / 2
    # normalize the input using the ImageNet mean and std
    mean = ms.Tensor([0.485, 0.456, 0.406], dtype=normalized_input.dtype).view(1, 3, 1, 1)
    std = ms.Tensor([0.229, 0.224, 0.225], dtype=normalized_input.dtype).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output


class PerceptualLoss(nn.Cell):
    r"""Perceptual loss initialization.

    Args:
       network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
       layers (str or list of str) : The layers used to compute the loss.
       weights (float or list of float : The loss weights of each layer.
       criterion (str): The type of distance function: 'l1' | 'l2'.
       resize (bool) : If ``True``, resize the input images to 224x224.
       resize_mode (str): Algorithm used for resizing.
       num_scales (int): The loss will be evaluated at original size and
        this many times downsampled sizes.
       per_sample_weight (bool): Output loss for individual samples in the
        batch instead of mean loss.
    """

    def __init__(self, network='vgg19', layers='relu_4_1', weights=None,
                 criterion='l1', resize=False, resize_mode='bilinear',
                 num_scales=1, per_sample_weight=False):
        super().__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]

        assert len(layers) == len(weights), \
            'The number of layers (%s) must be equal to ' \
            'the number of weights (%s).' % (len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)

        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        reduction = 'mean' if not per_sample_weight else 'none'
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        print('Perceptual loss:')
        print('\tMode: {}'.format(network))

    def construct(self, inp, target, per_sample_weights=None):
        r"""Perceptual loss forward.

        Args:
           inp (4D tensor) : Input tensor.
           target (4D tensor) : Ground truth tensor, same shape as the input.
           per_sample_weight (bool): Output loss for individual samples in the
            batch instead of mean loss.
        Returns:
           (scalar tensor) : The perceptual loss.
        """
        # Perceptual loss should operate in eval mode by default.
        inp, target = apply_imagenet_normalization(inp), apply_imagenet_normalization(target)
        if self.resize:
            inp = ops.interpolate(inp, mode=self.resize_mode, size=(224, 224), align_corners=False)
            target = ops.interpolate(target, mode=self.resize_mode, size=(224, 224), align_corners=False)

        # Evaluate perceptual loss at each scale.
        loss = 0
        for scale in range(self.num_scales):

            input_features, target_features = self.model(inp), self.model(target)

            for i, weight in zip(range(len(self.layers)), self.weights):
                # Example per-layer VGG19 loss values after applying
                # [0.03125, 0.0625, 0.125, 0.25, 1.0] weighting.
                # relu_1_1, 0.014698
                # relu_2_1, 0.085817
                # relu_3_1, 0.349977
                # relu_4_1, 0.544188
                # relu_5_1, 0.906261
                # print('%s, %f' % (
                #     layer,
                #     weight * self.criterion(
                #                  input_features[layer],
                #                  target_features[
                #                  layer].detach()).item()))
                l_tmp = self.criterion(input_features[i], target_features[i])
                if per_sample_weights is not None:
                    l_tmp = l_tmp.mean(1).mean(1).mean(1)
                loss += weight * l_tmp
            # Downsample the input and target.
            if scale != self.num_scales - 1:
                inp = ops.interpolate(
                    inp, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)
                target = ops.interpolate(
                    target, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)

        return loss.float()


class _PerceptualNetwork(nn.Cell):
    r"""The network that extracts features to compute the perceptual loss.

    Args:
        network (nn.SequentialCell) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, nn.SequentialCell), \
            'The network needs to be of type "nn.SequentialCell".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.get_parameters():
            param.requires_grad = False

    def construct(self, x):
        r"""Extract perceptual features."""
        output = []
        for i, layer in enumerate(self.network.cell_list):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                # If the current layer is used by the perceptual loss.
                output.append(x)
            if len(output) == len(self.layers):
                break
        return output


def _vgg19(layers):
    r"""Get vgg19 layers"""
    vgg = vgg19(phase="test", pretrained=True)
    # network = nn.SequentialCell(*(list(vgg.layers) + [vgg.avgpool] + [nn.Flatten()] + list(vgg.classifier)))
    network = nn.SequentialCell(*(list(vgg.layers) + [vgg.flatten] + list(vgg.classifier)))
    layer_name_mapping = {2: 'relu_1_1',
                          5: 'relu_1_2',
                          9: 'relu_2_1',
                          12: 'relu_2_2',
                          16: 'relu_3_1',
                          19: 'relu_3_2',
                          22: 'relu_3_3',
                          25: 'relu_3_4',
                          29: 'relu_4_1',
                          32: 'relu_4_2',
                          35: 'relu_4_3',
                          38: 'relu_4_4',
                          42: 'relu_5_1',
                          45: 'relu_5_2',
                          48: 'relu_5_3',
                          51: 'relu_5_4',
                          52: 'pool_5',
                          60: 'fc_2'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


if __name__=="__main__":
    layers = [
        "relu_1_1",
        "relu_2_1",
        "relu_3_1",
        "relu_4_1",
        "relu_5_1"
    ]
    weights = [
        0.03125, 0.0625, 0.125, 0.25, 1.0
    ]
    ptloss = PerceptualLoss(layers=layers, num_scales=3, weights=weights)
