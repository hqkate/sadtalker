import os
import numpy as np
import mindspore as ms
from mindspore import ops, nn

from utils.preprocess import split_coeff
from losses.expnet_loss import LandmarksLoss, LipReadingLoss

# from models.lipreading import get_lipreading_model
# from models.external.face3d.face_renderer import renderer
from models.audio2exp.utils import get_recon_model, get_wav2lip_model


class ExpNetWithLossCell(nn.Cell):
    """Generator with loss cell"""

    def __init__(self, expnet, bfm, cfg, args):
        super().__init__()
        self.expnet = expnet
        self.bfm = bfm
        self.cfg = cfg

        # lipreading_video = get_lipreading_model("video")
        # lipreading_audio = get_lipreading_model("audio")
        # lipreading_renderer = renderer

        # distill loss (lip-only coefficients, MSE)
        self.distill_loss = nn.MSELoss()

        # landmarks loss (eyes)
        self.lks_loss = LandmarksLoss()

        # lip-reading loss (cross-entropy)
        # self.lread_loss = LipReadingLoss(
        #     lipreading_video,
        #     lipreading_audio,
        #     lipreading_renderer,
        #     batch_size=args.batch_size
        # )

    def construct(
        self,
        current_mel_input,
        curr_ref,
        ratio_gt,
        coeffs
    ):
        # expnet
        exp_coeff_pred = self.expnet(current_mel_input, curr_ref, ratio_gt)

        # # replace coeffs
        exp_coeff_wav2lip = coeffs[1]

        new_coeffs = (
            coeffs[0],
            exp_coeff_pred.view(-1, 64),
            coeffs[2],
            coeffs[3],
            coeffs[4],
            coeffs[5],
        )

        # distill loss (lip-only coefficients, MSE)
        loss_distill = self.distill_loss(exp_coeff_pred.view(-1, 64), exp_coeff_wav2lip)

        # landmarks loss (eyes)
        landmarks_w2l = self.bfm.compute_for_render_landmarks(coeffs)  # bs*T, 68, 2
        # landmarks_w2l = render_results_1[-1]

        landmarks_rep = self.bfm.compute_for_render_landmarks(new_coeffs)
        # landmarks_rep = render_results_2[-1]

        # face_vertex = render_results_2[0]
        # face_texture = render_results_2[1]
        # face_color = render_results_2[2]
        # face_proj = render_results_2[3]

        loss_lks = self.lks_loss(landmarks_w2l, landmarks_rep, ratio_gt)
        # loss_lks = 0.0

        # # lip-reading loss (cross-entropy)
        # loss_read = self.lread_loss(
        #     audio_wav,
        #     face_vertex,
        #     face_color,
        #     self.bfm1.triangle,
        #     face_proj,
        #     landmarks_rep,
        # )
        # loss = 2.0 * loss_distill + 0.01 * loss_lks + 0.01 * loss_read

        loss = 2.0 * loss_distill + 0.01 * loss_lks

        return loss


class ExpNetTrainer:
    """ExpNetTrainer"""

    def __init__(self, train_one_step, cfg):
        super(ExpNetTrainer, self).__init__()
        self.train_one_step = train_one_step
        self.cfg = cfg

        # load wav2lip model
        wav2lip = get_wav2lip_model(cfg)
        coeff_enc = get_recon_model(cfg)

        self.wav2lip = wav2lip
        self.coeff_enc = coeff_enc

    def run(self, data_batch):
        current_mel_input = data_batch["indiv_mels"]
        curr_ref = data_batch["ref"][:, :, :64]
        ratio_gt = data_batch["ratio_gt"]
        masked_src_img = data_batch["masked_src_img"]

        # get wav2lip coeffs
        audiox = current_mel_input.view(-1, 1, 80, 16)
        first_frame_img = masked_src_img.view(-1, 6, 96, 96)
        img_with_lip = self.wav2lip(audiox, first_frame_img)  # bs*T, 3, 96, 96
        wav2lip_coeff = self.coeff_enc(img_with_lip)

        # replace coeffs
        coeffs = split_coeff(wav2lip_coeff)

        print("running train_one_step ...")
        self.train_one_step.set_train(True)
        loss_g = self.train_one_step(
            current_mel_input, curr_ref, ratio_gt, coeffs
        )
        return loss_g

    def train(self, total_steps, dataset, callbacks, save_ckpt_logs=True, **kwargs):
        print(f"Start training for {total_steps} iterations")

        dataset_size = dataset.get_dataset_size()
        repeats_num = (total_steps + dataset_size - 1) // dataset_size
        dataset = dataset.repeat(repeats_num)
        dataloader = dataset.create_dict_iterator()
        for num_batch, databatch in enumerate(dataloader):

            if num_batch >= total_steps:
                print("Reached the target number of iterations")
                break
            loss = self.run(databatch)

            print(loss)

            # if save_ckpt_logs:
            #     callbacks([loss_g.asnumpy().mean(), loss_d.asnumpy().mean()])

        print("Training completed")
