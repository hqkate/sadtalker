import os
from utils.preprocess import CropAndExtract

checkpoint_dir = "./checkpoints/"
config_dir = "./config/"

sadtalker_paths = {
    'wav2lip_checkpoint': os.path.join(checkpoint_dir, 'wav2lip.pth'),
    'audio2pose_checkpoint': os.path.join(checkpoint_dir, 'auido2pose_00140-model.pth'),
    'audio2exp_checkpoint': os.path.join(checkpoint_dir, 'auido2exp_00300-model.pth'),
    'free_view_checkpoint': os.path.join(checkpoint_dir, 'facevid2vid_00189-model.pth.tar'),
    'path_of_net_recon_model': os.path.join(checkpoint_dir, 'epoch_20.pth')
}
sadtalker_paths['dir_of_BFM_fitting'] = os.path.join(
    config_dir)  # , 'BFM_Fitting'
sadtalker_paths['audio2pose_yaml_path'] = os.path.join(
    config_dir, 'auido2pose.yaml')
sadtalker_paths['audio2exp_yaml_path'] = os.path.join(
    config_dir, 'auido2exp.yaml')
# os.path.join(config_dir, 'auido2exp.yaml')
sadtalker_paths['use_safetensor'] = False

extractor = CropAndExtract(sadtalker_paths)
