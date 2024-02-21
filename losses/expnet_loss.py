import numpy as np
import mindspore as ms
from mindspore import nn, ops
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms import Compose


class NormalizeUtterance:
    """Normalize per raw audio by removing the mean and divided by the standard deviation"""

    def __call__(self, signal):
        signal_std = signal.std()
        signal_mean = signal.mean()
        return (signal - signal_mean) / signal_std


class LandmarksLoss(nn.LossBase):
    def __init__(self, reduction="mean"):
        super().__init__(reduction)
        self.cast = ops.Cast()

    def get_eye_ratio(self, points):
        # (B, 68, 2)

        width = (
            ops.dist(points[:, 39, :], points[:, 36, :])
            + ops.dist(points[:, 45, :], points[:, 42, :])
        ) / 2.0

        height = (
            ops.dist(points[:, 37, :], points[:, 40, :])
            + ops.dist(points[:, 38, :], points[:, 41, :])
            + ops.dist(points[:, 43, :], points[:, 46, :])
            + ops.dist(points[:, 44, :], points[:, 47, :])
        ) / 4.0

        ratio = height.astype(ms.float32)  / width.astype(ms.float32)
        return ratio.astype(ms.float32)  # [B]

    def get_eye_loss(self, lks, z_blink):
        eye_ratio = self.get_eye_ratio(lks)
        loss = ops.dist(eye_ratio, z_blink.view(-1), p=1)
        return loss

    def construct(self, landmarks_ori, landmarks_rep, ratio_gt):
        """
        args:
            landmarks_ori: T, 68, 2
            landmarks_rep: T, 68, 2
            ratio_gt: bs, T, 1
        """
        loss_eye = self.get_eye_loss(landmarks_rep, ratio_gt)
        loss_point = ops.mean(ops.dist(landmarks_ori, landmarks_rep, 2))

        loss = 200.0 * loss_eye + loss_point

        return loss


class LipReadingLoss(nn.LossBase):
    def __init__(self, lipreading_video, lipreading_audio, renderer, reduction="mean", batch_size=1):
        super().__init__(reduction)
        self.lipreading_video = lipreading_video
        self.lipreading_audio = lipreading_audio
        self.renderer = renderer
        self.celoss = nn.CrossEntropyLoss()

        self._crop_width = 96
        self._crop_height = 96
        self._window_margin = 12
        self._start_idx = 48
        self._stop_idx = 68
        self._bs = batch_size

        crop_size = (88, 88)
        mean = 0.421
        std = 0.165

        # ---- transform mouths before going into the lipread network for loss ---- #
        self.mouth_transform = Compose(
            [
                vision.Normalize([0.0], [1.0]),
                vision.CenterCrop(crop_size),
                vision.Normalize([mean], [std]),
            ]
        )

    def cut_mouth(self, images, landmarks, convert_grayscale=True):
        """function adapted from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages"""

        mouth_sequence = []
        # landmarks = landmarks * 112 + 112
        converter = vision.ConvertColor(
            vision.ConvertMode.COLOR_RGB2GRAY
        )  # TODO: support CPU only

        for frame_idx, frame in enumerate(images):
            window_margin = min(
                self._window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx
            )
            smoothed_landmarks = landmarks[
                frame_idx - window_margin : frame_idx + window_margin + 1
            ].mean(axis=0)
            smoothed_landmarks += landmarks[frame_idx].mean(
                axis=0
            ) - smoothed_landmarks.mean(axis=0)

            center_x, center_y = ops.mean(
                smoothed_landmarks[self._start_idx : self._stop_idx], axis=0
            )

            center_x = center_x.round()
            center_y = center_y.round()

            height = self._crop_height // 2
            width = self._crop_width // 2

            threshold = 5

            if convert_grayscale:
                # img = F_v.rgb_to_grayscale(frame).squeeze()
                img = converter(frame.asnumpy()).squeeze()
            else:
                img = frame.asnumpy()

            if center_y - height < 0:
                center_y = height
            if center_y - height < 0 - threshold:
                raise Exception("too much bias in height")
            if center_x - width < 0:
                center_x = width
            if center_x - width < 0 - threshold:
                raise Exception("too much bias in width")

            if center_y + height > img.shape[-2]:
                center_y = img.shape[-2] - height
            if center_y + height > img.shape[-2] + threshold:
                raise Exception("too much bias in height")
            if center_x + width > img.shape[-1]:
                center_x = img.shape[-1] - width
            if center_x + width > img.shape[-1] + threshold:
                raise Exception("too much bias in width")

            mouth = img[
                ...,
                int(center_y - height) : int(center_y + height),
                int(center_x - width) : int(center_x + round(width)),
            ]

            mouth_sequence.append(ms.Tensor(mouth, ms.float32))

        mouth_sequence = ops.stack(mouth_sequence, axis=0)
        return mouth_sequence

    def preprocess_audio(self, audio_wav):
        audio_wav = NormalizeUtterance()(audio_wav)
        audio_wav = audio_wav.unsqueeze(1)
        return audio_wav

    def preprocess_images(self, pred_faces, landmarks):
        # codes borrowed from https://github.com/filby89/spectre/blob/master/src/trainer_spectre.py
        # ---- initialize values for cropping the face around the mouth for lipread loss ---- #

        """ lipread loss - first crop the mouths of the input and rendered faces
        and then calculate the cosine distance of features
        """

        mouths_pred = self.cut_mouth(pred_faces, landmarks).unsqueeze(
            -1
        )  # (84, 96, 96, 1)
        mouths_pred = self.mouth_transform(mouths_pred.asnumpy())  # (84, 88, 88, 1)

        # ---- resize back to Bx1xKxHxW (grayscale input for lipread net) ---- #
        # (bs, color channel-grey scale, seq-length, width, height)
        mouths_pred = ms.Tensor(mouths_pred.transpose(0, 3, 1, 2)).unsqueeze(2)

        return mouths_pred

    def construct(
        self, audio_wav, face_vertex, face_color, triangle_coeffs, face_proj, landmarks
    ):
        # face rendering
        pred_faces = []
        for i in range(self._bs):
            pred_face = self.renderer(
                face_vertex[i, :, :].asnumpy(),
                face_color[i, :, :].asnumpy(),
                triangle_coeffs.asnumpy(),
            )
            pred_face = ms.Tensor(pred_face, ms.float32)
            pred_faces.append(pred_face.unsqueeze(0))

        pred_faces = ops.cat(pred_faces)  # (84, 256, 256, 3)

        input_tensor = self.preprocess_images(pred_faces, landmarks)
        input_audio = self.preprocess_audio(audio_wav)

        c_p = self.lipreading_audio(input_audio, [input_audio.shape[-1]] * len(input_audio))  # (B, 500)
        c_gt = self.lipreading_video(
            input_tensor, [input_tensor.shape[-1]] * len(input_tensor)
        )

        loss = self.celoss(ops.log_softmax(c_p), ops.log_softmax(c_gt))
        return loss


"""
class ExpNetLoss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.distill_loss = nn.MSELoss()
        self.lks_loss = LandmarksLoss()
        self.lread_loss = LipReadingLoss()

        self.bfm1 = ParametricFaceModel(bfm_folder="checkpoints/BFM_Fitting")
        self.bfm2 = ParametricFaceModel(bfm_folder="checkpoints/BFM_Fitting")

        self.cast = ops.Cast()

    def construct(self, exp_coeff_pred, wav2lip_coeff, ratio_gt, audio_wav
                  ):

        # (id_coeffs, exp_coeffs, tex_coeffs, angles, gammas, translations)
        coeffs = split_coeff(wav2lip_coeff)
        exp_coeff_wav2lip = coeffs[1]

        new_coeffs = (
            coeffs[0],
            exp_coeff_pred.view(-1, 64),
            coeffs[2],
            coeffs[3],
            coeffs[4],
            coeffs[5]
        )

        # distill loss (lip-only coefficients, MSE)
        loss_distill = self.distill_loss(
            exp_coeff_pred.view(-1, 64), exp_coeff_wav2lip)

        # landmarks loss (eyes)
        render_results_1 = self.bfm1.compute_for_render_new(coeffs)  # bs*T, 68, 2
        landmarks_w2l = render_results_1[-1]

        render_results_2 = self.bfm2.compute_for_render_new(new_coeffs)
        landmarks_rep = render_results_2[-1]

        face_vertex = render_results_2[0]
        face_texture = render_results_2[1]
        face_color = render_results_2[2]
        face_proj = render_results_2[3]

        # landmarks_w2l = ms.Tensor(np.load("landmarks_w2l.npy"), ms.float32)
        # landmarks_rep = ms.Tensor(np.load("landmarks_rep.npy"), ms.float32)
        loss_lks = self.lks_loss(landmarks_w2l, landmarks_rep, ratio_gt)

        # lip-reading loss (cross-entropy)
        loss_read = self.lread_loss(audio_wav, face_vertex, face_color, self.bfm1.triangle, face_proj, landmarks_rep)

        loss = 2.0 * loss_distill + 0.01 * loss_lks + 0.01 * loss_read

        return loss
"""
