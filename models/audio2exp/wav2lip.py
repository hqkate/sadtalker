from mindspore import nn, ops
from models.audio2exp.conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class Wav2Lip(nn.Cell):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.CellList([
            nn.SequentialCell(Conv2d(6, 16, kernel_size=7,
                                     stride=1, padding=3)),  # 96,96

            nn.SequentialCell(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
                              Conv2d(32, 32, kernel_size=3, stride=1,
                                     padding=1, use_residual=True),
                              Conv2d(32, 32, kernel_size=3, stride=1, padding=1, use_residual=True)),

            nn.SequentialCell(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 24,24
                              Conv2d(64, 64, kernel_size=3, stride=1,
                                     padding=1, use_residual=True),
                              Conv2d(64, 64, kernel_size=3, stride=1,
                                     padding=1, use_residual=True),
                              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, use_residual=True)),

            nn.SequentialCell(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 12,12
                              Conv2d(128, 128, kernel_size=3, stride=1,
                                     padding=1, use_residual=True),
                              Conv2d(128, 128, kernel_size=3, stride=1, padding=1, use_residual=True)),

            nn.SequentialCell(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 6,6
                              Conv2d(256, 256, kernel_size=3, stride=1,
                                     padding=1, use_residual=True),
                              Conv2d(256, 256, kernel_size=3, stride=1, padding=1, use_residual=True)),

            nn.SequentialCell(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 3,3
                              Conv2d(512, 512, kernel_size=3, stride=1, padding=1, use_residual=True),),

            nn.SequentialCell(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
                              Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.audio_encoder = nn.SequentialCell(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, use_residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, use_residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, use_residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, use_residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, use_residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, use_residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, use_residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.CellList([
            nn.SequentialCell(
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.SequentialCell(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3,3
                              Conv2d(512, 512, kernel_size=3, stride=1, padding=1, use_residual=True),),

            nn.SequentialCell(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                              Conv2d(512, 512, kernel_size=3, stride=1,
                                     padding=1, use_residual=True),
                              Conv2d(512, 512, kernel_size=3, stride=1, padding=1, use_residual=True),),  # 6, 6

            nn.SequentialCell(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                              Conv2d(384, 384, kernel_size=3, stride=1,
                                     padding=1, use_residual=True),
                              Conv2d(384, 384, kernel_size=3, stride=1, padding=1, use_residual=True),),  # 12, 12

            nn.SequentialCell(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                              Conv2d(256, 256, kernel_size=3, stride=1,
                                     padding=1, use_residual=True),
                              Conv2d(256, 256, kernel_size=3, stride=1, padding=1, use_residual=True),),  # 24, 24

            nn.SequentialCell(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                              Conv2d(128, 128, kernel_size=3, stride=1,
                                     padding=1, use_residual=True),
                              Conv2d(128, 128, kernel_size=3, stride=1, padding=1, use_residual=True),),  # 48, 48

            nn.SequentialCell(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                              Conv2d(64, 64, kernel_size=3, stride=1,
                                     padding=1, use_residual=True),
                              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, use_residual=True),),])  # 96,96

        self.output_block = nn.SequentialCell(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                              nn.Conv2d(32, 3, kernel_size=1,
                                                        stride=1, padding=0, has_bias=True),
                                              nn.Sigmoid())

    def construct(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.shape[0]

        input_dim_size = len(face_sequences.shape)
        if input_dim_size > 4:
            audio_sequences = ops.cat(
                [audio_sequences[:, i] for i in range(audio_sequences.shape[1])], axis=0)
            face_sequences = ops.cat(
                [face_sequences[:, :, i] for i in range(face_sequences.shape[2])], axis=0)

        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = ops.cat((x, feats[-1]), axis=1)
            except Exception as e:
                print(x.shape)
                print(feats[-1].shape)
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = ops.split(x, B, axis=0)  # [(B, C, H, W)]
            outputs = ops.stack(x, axis=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs


class Wav2Lip_disc_qual(nn.Cell):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        self.face_encoder_blocks = nn.CellList([
            nn.SequentialCell(nonorm_Conv2d(3, 32, kernel_size=7,
                                            stride=1, padding=3)),  # 48,96

            nn.SequentialCell(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 48,48
                              nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.SequentialCell(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 24,24
                              nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.SequentialCell(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),   # 12,12
                              nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.SequentialCell(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),       # 6,6
                              nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.SequentialCell(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),     # 3,3
                              nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),

            nn.SequentialCell(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
                              nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.binary_pred = nn.SequentialCell(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.shape[2]//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.shape[0]
        face_sequences = ops.cat([face_sequences[:, :, i]
                                  for i in range(face_sequences.shape[2])], axis=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = ops.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
                                                   ops.ones((len(false_feats), 1)))

        return false_pred_loss

    def construct(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
