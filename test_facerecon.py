from argparse import ArgumentParser
from models.face3d.pytorch3d.facerecon_model import FaceReconModel
from mindspore import context

context.set_context(mode=context.GRAPH_MODE,
                    device_target="CPU")

parser = ArgumentParser()
parser = FaceReconModel.modify_commandline_options(parser)
args = parser.parse_args()
recon_model = FaceReconModel(args)