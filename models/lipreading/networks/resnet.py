from mindspore import Tensor, nn
from models.lipreading.networks.resnet1d import Swish


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     pad_mode='pad', padding=1, has_bias=False)


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
        nn.Conv2d(inplanes, outplanes, kernel_size=1,
                  stride=1, has_bias=False),
        nn.BatchNorm2d(outplanes),
    )


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type = 'prelu' ):
        super(BasicBlock, self).__init__()

        assert relu_type in ['relu','prelu', 'swish']

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-5)

        # type of ReLU is an input option
        if relu_type == 'relu':
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(channel=planes)
            self.relu2 = nn.PReLU(channel=planes)
        elif relu_type == 'swish':
            self.relu1 = Swish()
            self.relu2 = Swish()
        else:
            raise Exception('relu type not implemented')
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)

        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu2(out)

        return out


class ResNet(nn.Cell):

    def __init__(self, block, layers, num_classes=1000, relu_type = 'relu', gamma_zero = False, avg_pool_downsample = False):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, stride=1):


        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block( inplanes = self.inplanes,
                                                 outplanes = planes * block.expansion,
                                                 stride = stride )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type = self.relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type = self.relu_type))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x
