import numpy as np
from mindspore import nn
from mindspore.common.initializer import Initializer, Constant


class MultiscaleDiscriminator(nn.Cell):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        num_D=3,
        getIntermFeat=False,
    ):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64

        for i in range(num_D):
            netD = NLayerDiscriminator(
                input_nc,
                min(ndf_max, ndf * (2 ** (num_D - 1 - i))),
                n_layers,
                norm_layer,
                getIntermFeat,
            )
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(
                        self,
                        "scale" + str(i) + "_layer" + str(j),
                        getattr(netD, "model" + str(j)),
                    )
            else:
                setattr(self, "layer" + str(i), netD.model)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False, pad_mode="pad"
        )

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def construct(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [
                    getattr(self, "scale" + str(num_D - 1 - i) + "_layer" + str(j))
                    for j in range(self.n_layers + 2)
                ]
            else:
                model = getattr(self, "layer" + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Cell):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
    ):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv2d(
                    input_nc,
                    ndf,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    pad_mode="pad",
                ),
                nn.LeakyReLU(0.2),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv2d(
                        nf_prev,
                        nf,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        pad_mode="pad",
                    ),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv2d(
                    nf_prev, nf, kernel_size=kw, stride=1, padding=padw, pad_mode="pad"
                ),
                norm_layer(nf),
                nn.LeakyReLU(0.2),
            ]
        ]

        sequence += [
            [nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw, pad_mode="pad")]
        ]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.SequentialCell(*sequence_stream)

    def construct(self, input):
        return self.model(input)


class Hopenet(nn.Cell):
    """Fine-Grained Head Pose Estimation Without Keypoints
    # https://github.com/natanielruiz/deep-head-pose
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    """

    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, pad_mode="pad", has_bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, pad_mode="pad")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Dense(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Dense(512 * block.expansion, num_bins)
        self.fc_roll = nn.Dense(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Dense(512 * block.expansion + 3, 3)

        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.weight_init = Initializer(
                    init="HeNormal", mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight_init = Initializer(init=Constant(1))
                m.bias_init = Initializer(init=Constant(0))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    has_bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll
