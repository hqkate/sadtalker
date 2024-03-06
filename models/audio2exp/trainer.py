import os
from mindspore import ops, nn
from mindspore.nn import TrainOneStepCell

from utils.preprocess import split_coeff
from losses.expnet_loss import LandmarksLoss, LipReadingLoss
from models.face3d.bfm import ParametricFaceModel
from models.lipreading import get_lipreading_model
from models.external.face3d.face_renderer import renderer
from models.audio2exp.utils import get_recon_model, get_wav2lip_model


class ExpNetWithLossCell(nn.Cell):
    """Generator with loss cell"""

    def __init__(self, expnet, cfg, args):
        super().__init__()
        self.expnet = expnet
        self.cfg = cfg

        wav2lip = get_wav2lip_model(cfg)
        coeff_enc = get_recon_model(cfg)

        self.wav2lip = wav2lip
        self.coeff_enc = coeff_enc

        self.bfm1 = ParametricFaceModel(bfm_folder="checkpoints/BFM_Fitting")
        self.bfm2 = ParametricFaceModel(bfm_folder="checkpoints/BFM_Fitting")

        # self.bfm = ParametricFaceModel(bfm_folder="checkpoints/BFM_Fitting")

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
        first_frame_img,
        # audio_wav,
    ):
        audiox = current_mel_input.view(-1, 1, 80, 16)
        first_frame_img = first_frame_img.view(-1, 6, 96, 96)

        # expnet
        exp_coeff_pred = self.expnet(current_mel_input, curr_ref, ratio_gt)

        # wav2lip
        img_with_lip = self.wav2lip(audiox, first_frame_img)  # bs*T, 3, 96, 96

        wav2lip_coeff = self.coeff_enc(img_with_lip)

        # replace coeffs
        coeffs = split_coeff(wav2lip_coeff)
        exp_coeff_wav2lip = coeffs[1]

        new_coeffs = (
            coeffs[0].copy(),
            exp_coeff_pred.view(-1, 64),
            coeffs[2].copy(),
            coeffs[3].copy(),
            coeffs[4].copy(),
            coeffs[5].copy(),
        )

        # distill loss (lip-only coefficients, MSE)
        loss_distill = self.distill_loss(exp_coeff_pred.view(-1, 64), exp_coeff_wav2lip)

        # landmarks loss (eyes)
        render_results_1 = self.bfm1.compute_for_render_new(coeffs)  # bs*T, 68, 2
        landmarks_w2l = render_results_1[-1]

        render_results_2 = self.bfm2.compute_for_render_new(new_coeffs)
        landmarks_rep = render_results_2[-1]

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

    def run(self, data_batch):
        current_mel_input = data_batch["indiv_mels"]
        curr_ref = data_batch["ref"][:, :, :64]
        ratio_gt = data_batch["ratio_gt"]
        first_frame_img = data_batch["masked_src_img"]
        # audio_wav = data_batch["audio_wav"]

        print("running train_one_step ...")
        self.train_one_step.set_train(True)
        loss_g = self.train_one_step(
            current_mel_input, curr_ref, ratio_gt, first_frame_img
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
