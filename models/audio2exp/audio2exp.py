from tqdm import tqdm
import cv2
import mindspore as ms
from mindspore import nn, ops
from models.utils.load_models import load_wav2lip, load_net_recon, load_facerender


class Audio2Exp(nn.Cell):
    """ ExpNet implementation (training)
    """

    def __init__(self, netG, cfg, wav2lip=None, is_train=False):
        super(Audio2Exp, self).__init__()
        self.cfg = cfg
        self.netG = netG

        self.is_train = is_train
        self.wav2lip = wav2lip

    def test(self, batch):

        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]

        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10), 'audio2exp:'):  # every 10 frames

            current_mel_input = mel_input[:, i:i+10]

            # ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch['ref'][:, :, :64][:, i:i+10]
            ratio = batch['ratio_gt'][:, i:i+10]  # bs T

            # bs*T 1 80 16
            audiox = current_mel_input.view(-1, 1, 80, 16)
            curr_exp_coeff_pred = self.netG(
                audiox, ref, ratio)         # bs T 64

            exp_coeff_pred += [curr_exp_coeff_pred]

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': ops.cat(exp_coeff_pred, axis=1)
        }
        return results_dict

    def getloss(self, batch):

        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        pic_name = batch['pic_name']
        bs = mel_input.shape[0]
        T = mel_input.shape[1]

        import pdb; pdb.set_trace()
        first_frame_img = ms.Tensor(cv2.imread(pic_name), dtype=ms.float32).unsqueeze(0).repeat(T, axis=0)

        exp_coeff_pred = []
        wav2lip_coeff = []

        for i in tqdm(range(0, T, 10), 'audio2exp:'):  # every 10 frames

            current_mel_input = mel_input[:, i:i+10]
            first_mel_input = mel_input[:, i]

            # ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch['ref'][:, :, :64][:, i:i+10]
            ratio = batch['ratio_gt'][:, i:i+10]  # bs T

            # bs*T 1 80 16
            audiox = current_mel_input.view(-1, 1, 80, 16)
            curr_exp_coeff_pred = self.netG(
                audiox, ref, ratio)         # bs T 64

            exp_coeff_pred += [curr_exp_coeff_pred]

            # wav2lip

            img_with_lip = self.wav2lip(audiox, first_frame_img)
            full_coeff = self.net_recon(self.wav2lip(audiox, first_mel_input))
            wav2lip_coeff += [full_coeff]

            # reconstruct coeffs

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': ops.cat(exp_coeff_pred, axis=1),
            'wav2lip_coef': ops.cat(wav2lip_coeff, axis=1),
        }

        return results_dict
