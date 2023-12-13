import numpy as np
from mindspore import nn


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
            3, stride=2, padding=[1, 1], count_include_pad=False, pad_mode='pad'
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
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, pad_mode='pad'),
                nn.LeakyReLU(0.2),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, pad_mode='pad'),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw, pad_mode='pad'),
                norm_layer(nf),
                nn.LeakyReLU(0.2),
            ]
        ]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw, pad_mode='pad')]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.SequentialCell(*sequence_stream)

    def construct(self, input):
        return self.model(input)
