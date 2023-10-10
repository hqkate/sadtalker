from mindspore import nn
from models.face3d.facexlib.resnet import ResNet, BasicBlock


def downsample_basic_block(inplanes, outplanes, stride):
    return nn.SequentialCell(
        nn.Conv2d(inplanes, outplanes, kernel_size=1,
                  stride=stride, has_bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_basic_block_v2(inplanes, outplanes, stride):
    return nn.SequentialCell(
        nn.AvgPool2d(kernel_size=stride, stride=stride,
                     ceil_mode=True, count_include_pad=False),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, has_bias=False),
        nn.BatchNorm2d(outplanes),
    )


def get_lipreading_resnet(block, layers, relu_type='relu', avg_pool_downsample=False):
    assert relu_type in ['relu', 'prelu', 'swish']
    downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block
    model = ResNet(block, layers, relu_type=relu_type, downsample_block=downsample_block)
    return model
