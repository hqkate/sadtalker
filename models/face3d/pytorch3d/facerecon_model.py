"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""
import argparse
import numpy as np
import trimesh
from scipy.io import savemat
from mindspore import nn
from models.face3d.bfm import ParametricFaceModel
from models.face3d.pytorch3d.mesh_renderer import MeshRenderer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class FaceReconModel(nn.Cell):

    @staticmethod
    def modify_commandline_options(parser):
        """  Configures options specific for CUT model
        """
        # net structure and parameters
        parser.add_argument('--net_recon', type=str, default='resnet50',
                            choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
        parser.add_argument('--init_path', type=str,
                            default='./checkpoints/init_model/resnet50-0676ba61.pth')
        parser.add_argument('--use_last_fc', type=str2bool, nargs='?',
                            const=True, default=False, help='zero initialize the last fc')
        parser.add_argument('--bfm_folder', type=str,
                            default='./checkpoints/BFM_Fitting/')
        parser.add_argument('--bfm_model', type=str,
                            default='BFM_model_front.mat', help='bfm model')

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)

        opt, _ = parser.parse_known_args()
        parser.set_defaults(
            focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
        )
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        super().__init__()  # call the initialization method of BaseModel

        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=False, default_name=opt.bfm_model
        )

        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(
                2 * opt.center)
        )

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs']
        self.atten_mask = input['msks'] if 'msks' in input else None
        self.gt_lm = input['lms'] if 'lms' in input else None
        self.trans_m = input['M'] if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None

    def construct(self, output_coeff):
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(output_coeff)
        self.pred_mask, _, self.pred_face = self.renderer.forward_rendering(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)

        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)

    def save_mesh(self, name):

        recon_shape = self.pred_vertex  # get reconstructed shape
        # from camera space to world space
        recon_shape[..., -1] = 10 - recon_shape[..., -1]
        recon_shape = recon_shape.asnumpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.asnumpy()[0]
        tri = self.facemodel.face_buf.asnumpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(
            255. * recon_color, 0, 255).astype(np.uint8))
        mesh.export(name)

    def save_coeff(self, name):

        pred_coeffs = {key: self.pred_coeffs_dict[key].asnumpy() for key in self.pred_coeffs_dict}
        pred_lm = self.pred_lm.asnumpy()
        # transfer to image coordinate
        pred_lm = np.stack(
            [pred_lm[:, :, 0], self.input_img.shape[2]-1-pred_lm[:, :, 1]], axis=2)
        pred_coeffs['lm68'] = pred_lm
        savemat(name, pred_coeffs)
