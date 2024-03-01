import os
import mindspore as ms
from models.face3d.networks import ReconNetWrapper
from models.wav2lip.wav2lip import Wav2Lip


def get_wav2lip_model(config):
    # load wav2lip model
    wav2lip = Wav2Lip()
    checkpoint_dir = config.audio2exp.path.checkpoint_dir
    path_wav2lip = os.path.join(
        checkpoint_dir, config.audio2exp.path.wav2lip_checkpoint
    )
    param_dict = ms.load_checkpoint(path_wav2lip)
    ms.load_param_into_net(wav2lip, param_dict)
    wav2lip.set_train(False)
    return wav2lip


def get_recon_model(config):
    # load 3DMM Encoder
    coeff_enc = ReconNetWrapper("resnet50", use_last_fc=False, init_path="")
    checkpoint_dir = config.preprocess.path.checkpoint_dir
    path_net_recon = os.path.join(
        checkpoint_dir, config.preprocess.path.path_of_net_recon_model
    )
    param_dict = ms.load_checkpoint(path_net_recon)
    ms.load_param_into_net(coeff_enc, param_dict)
    coeff_enc.set_train(False)
    return coeff_enc
