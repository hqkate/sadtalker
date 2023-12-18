# Copyright © 2023 Huawei Technologies Co, Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""VGG16FeatureExtractor"""

from mindspore import load_checkpoint, load_param_into_net, nn


class VGG19FeatureExtractor(nn.Cell):
    """VGG19FeatureExtractor"""

    def __init__(self):
        super().__init__()
        self.enc_1 = nn.SequentialCell(
            [
                nn.Conv2d(3, 64, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        self.enc_2 = nn.SequentialCell(
            [
                nn.Conv2d(64, 128, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        self.enc_3 = nn.SequentialCell(
            [
                nn.Conv2d(128, 256, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        self.enc_4 = nn.SequentialCell(
            [
                nn.Conv2d(256, 512, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        self.enc_5 = nn.SequentialCell(
            [
                nn.Conv2d(512, 512, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )

    def construct(self, x):
        """construct"""
        enc_1_output = self.enc_1(x)
        enc_2_output = self.enc_2(enc_1_output)
        enc_3_output = self.enc_3(enc_2_output)
        enc_4_output = self.enc_4(enc_3_output)
        enc_5_output = self.enc_5(enc_4_output)
        return enc_1_output, enc_2_output, enc_3_output, enc_4_output, enc_5_output


def get_feature_extractor(cfg):
    """get feature extractor"""
    vgg_feat_extractor = VGG19FeatureExtractor()
    if cfg.train.pretrained_vgg:
        load_param_into_net(
            vgg_feat_extractor, load_checkpoint(cfg.train.pretrained_vgg)
        )
    return vgg_feat_extractor
