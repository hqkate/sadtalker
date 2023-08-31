import os
import numpy as np
import mindspore as ms
from mindspore import ops
from scipy.io import savemat, loadmat
from yacs.config import CfgNode as CN
from scipy.signal import savgol_filter

from models.audio2pose.audio2pose import Audio2Pose
from models.audio2exp.audio2exp import Audio2Exp
from models.audio2exp.expnet import ExpNet


def load_cpk(checkpoint_path, model=None, optimizer=None):
    checkpoint = ms.load_checkpoint(checkpoint_path)
    if model is not None:
        ms.load_param_into_net(model, checkpoint['model'])
    if optimizer is not None:
        ms.load_param_into_net(optimizer, checkpoint['optimizer'])

    return checkpoint['epoch']


class Audio2Coeff():

    def __init__(self, sadtalker_path, device):
        # load config
        fcfg_pose = open(sadtalker_path['audio2pose_yaml_path'])
        cfg_pose = CN.load_cfg(fcfg_pose)
        cfg_pose.freeze()
        fcfg_exp = open(sadtalker_path['audio2exp_yaml_path'])
        cfg_exp = CN.load_cfg(fcfg_exp)
        cfg_exp.freeze()

        # load audio2pose_model
        self.audio2pose_model = Audio2Pose(cfg_pose, None)
        self.audio2pose_model = self.audio2pose_model
        self.audio2pose_model.set_train(False)
        for param in self.audio2pose_model.get_parameters():
            param.requires_grad = False

            load_cpk(sadtalker_path['audio2pose_checkpoint'],
                     model=self.audio2pose_model)

        # load audio2exp_model
        netG = ExpNet()
        for param in netG.get_parameters():
            netG.requires_grad = False
        netG.set_train(False)

        load_cpk(
            sadtalker_path['audio2exp_checkpoint'], model=netG)

        self.audio2exp_model = Audio2Exp(
            netG, cfg_exp, device=device, prepare_training_loss=False)
        self.audio2exp_model = self.audio2exp_model
        for param in self.audio2exp_model.get_parameters():
            param.requires_grad = False
        self.audio2exp_model.set_train(False)

    def generate(self, batch, coeff_save_dir, pose_style, ref_pose_coeff_path=None):

        # test
        results_dict_exp = self.audio2exp_model.test(batch)
        exp_pred = results_dict_exp['exp_coeff_pred']  # bs T 64

        # for class_id in  range(1):
        # class_id = 0#(i+10)%45
        # class_id = random.randint(0,46)                                   #46 styles can be selected
        batch['class'] = ms.Tensor([pose_style], dtype=ms.int32)
        results_dict_pose = self.audio2pose_model.test(batch)
        pose_pred = results_dict_pose['pose_pred']  # bs T 6

        pose_len = pose_pred.shape[1]
        if pose_len < 13:
            pose_len = int((pose_len-1)/2)*2+1
            pose_pred = ms.Tensor(savgol_filter(
                np.array(pose_pred), pose_len, 2, axis=1))
        else:
            pose_pred = ms.Tensor(savgol_filter(
                np.array(pose_pred), 13, 2, axis=1))

        coeffs_pred = ops.cat((exp_pred, pose_pred), axis=-1)  # bs T 70

        coeffs_pred_numpy = coeffs_pred[0].clone().numpy()

        if ref_pose_coeff_path is not None:
            coeffs_pred_numpy = self.using_refpose(
                coeffs_pred_numpy, ref_pose_coeff_path)

        savemat(os.path.join(coeff_save_dir, '%s##%s.mat' % (batch['pic_name'], batch['audio_name'])),
                {'coeff_3dmm': coeffs_pred_numpy})

        return os.path.join(coeff_save_dir, '%s##%s.mat' % (batch['pic_name'], batch['audio_name']))

    def using_refpose(self, coeffs_pred_numpy, ref_pose_coeff_path):
        num_frames = coeffs_pred_numpy.shape[0]
        refpose_coeff_dict = loadmat(ref_pose_coeff_path)
        refpose_coeff = refpose_coeff_dict['coeff_3dmm'][:, 64:70]
        refpose_num_frames = refpose_coeff.shape[0]
        if refpose_num_frames < num_frames:
            div = num_frames//refpose_num_frames
            re = num_frames % refpose_num_frames
            refpose_coeff_list = [refpose_coeff for i in range(div)]
            refpose_coeff_list.append(refpose_coeff[:re, :])
            refpose_coeff = np.concatenate(refpose_coeff_list, axis=0)

        # relative head pose
        coeffs_pred_numpy[:, 64:70] = coeffs_pred_numpy[:, 64:70] + \
            (refpose_coeff[:num_frames, :] - refpose_coeff[0:1, :])
        return coeffs_pred_numpy
