import pickle
import mindspore as ms
from argparse import ArgumentParser
from models.face3d.pytorch3d.facerecon_model import FaceReconModel
from mindspore import context

context.set_context(mode=context.PYNATIVE_MODE,
                    device_target="CPU")

parser = ArgumentParser()
parser = FaceReconModel.modify_commandline_options(parser)
args = parser.parse_args()
recon_model = FaceReconModel(args)

with open("coeffs.pkl", "rb") as f:
    coeffs = pickle.load(f)

coeffs = [ms.Tensor(coeff) for coeff in coeffs]
pred_vertex, pred_tex, pred_color, _, pred_lm = recon_model.facemodel.compute_for_render_new(
    coeffs)
pred_mask, _, pred_face = recon_model.renderer.forward_rendering(
    pred_vertex, ms.Tensor(recon_model.facemodel.face_buf), feat=pred_color)
