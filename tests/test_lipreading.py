import os
import json
import argparse
import numpy as np
import mindspore as ms
from mindspore import ops, context
from models.lipreading import get_model_from_json
from models.lipreading.model import Lipreading
from tools.save_ms_params import save_params


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description="Pytorch Lipreading ")
    # -- dataset config
    parser.add_argument("--dataset", default="lrw", help="dataset selection")
    parser.add_argument(
        "--num-classes", type=int, default=500, help="Number of classes"
    )
    parser.add_argument(
        "--modality",
        default="video",
        choices=["video", "audio"],
        help="choose the modality",
    )
    # -- directory
    parser.add_argument(
        "--data-dir",
        default="./datasets/LRW_h96w96_mouth_crop_gray",
        help="Loaded data directory",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        default="./labels/500WordsSortedList.txt",
        help="Path to txt file with labels",
    )
    parser.add_argument(
        "--annonation-direc", default=None, help="Loaded data directory"
    )
    # -- model config
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="resnet",
        choices=["resnet", "shufflenet"],
        help="Architecture used for backbone",
    )
    parser.add_argument(
        "--relu-type",
        type=str,
        default="relu",
        choices=["relu", "prelu"],
        help="what relu to use",
    )
    parser.add_argument(
        "--width-mult",
        type=float,
        default=1.0,
        help="Width multiplier for mobilenets and shufflenets",
    )
    # -- TCN config
    parser.add_argument(
        "--tcn-kernel-size",
        type=int,
        nargs="+",
        help="Kernel to be used for the TCN module",
    )
    parser.add_argument(
        "--tcn-num-layers",
        type=int,
        default=4,
        help="Number of layers on the TCN module",
    )
    parser.add_argument(
        "--tcn-dropout",
        type=float,
        default=0.2,
        help="Dropout value for the TCN module",
    )
    parser.add_argument(
        "--tcn-dwpw",
        default=False,
        action="store_true",
        help="If True, use the depthwise seperable convolution in TCN architecture",
    )
    parser.add_argument(
        "--tcn-width-mult", type=int, default=1, help="TCN width multiplier"
    )
    # -- DenseTCN config
    parser.add_argument(
        "--densetcn-block-config",
        type=int,
        nargs="+",
        help="number of denselayer for each denseTCN block",
    )
    parser.add_argument(
        "--densetcn-kernel-size-set",
        type=int,
        nargs="+",
        help="kernel size set for each denseTCN block",
    )
    parser.add_argument(
        "--densetcn-dilation-size-set",
        type=int,
        nargs="+",
        help="dilation size set for each denseTCN block",
    )
    parser.add_argument(
        "--densetcn-growth-rate-set",
        type=int,
        nargs="+",
        help="growth rate for DenseTCN",
    )
    parser.add_argument(
        "--densetcn-dropout", default=0.2, type=float, help="Dropout value for DenseTCN"
    )
    parser.add_argument(
        "--densetcn-reduced-size",
        default=256,
        type=int,
        help="the feature dim for the output of reduce layer",
    )
    parser.add_argument(
        "--densetcn-se",
        default=False,
        action="store_true",
        help="If True, enable SE in DenseTCN",
    )
    parser.add_argument(
        "--densetcn-condense",
        default=False,
        action="store_true",
        help="If True, enable condenseTCN",
    )
    # -- train
    parser.add_argument("--training-mode", default="tcn", help="tcn")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adam", "sgd", "adamw"]
    )
    parser.add_argument("--lr", default=3e-4, type=float, help="initial learning rate")
    parser.add_argument("--init-epoch", default=0, type=int, help="epoch to start at")
    parser.add_argument("--epochs", default=80, type=int, help="number of epochs")
    parser.add_argument(
        "--test", default=False, action="store_true", help="training mode"
    )
    # -- mixup
    parser.add_argument(
        "--alpha",
        default=0.4,
        type=float,
        help="interpolation strength (uniform=1., ERM=0.)",
    )
    # -- test
    parser.add_argument(
        "--model-path", type=str, default=None, help="Pretrained model pathname"
    )
    parser.add_argument(
        "--allow-size-mismatch",
        default=False,
        action="store_true",
        help="If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes",
    )
    # -- feature extractor
    parser.add_argument(
        "--extract-feats", default=False, action="store_true", help="Feature extractor"
    )
    parser.add_argument(
        "--mouth-patch-path",
        type=str,
        default=None,
        help="Path to the mouth ROIs, assuming the file is saved as numpy.array",
    )
    parser.add_argument(
        "--mouth-embedding-out-path",
        type=str,
        default=None,
        help="Save mouth embeddings to a specificed path",
    )
    # -- json pathname
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Model configuration with json format",
    )
    # -- other vars
    parser.add_argument("--interval", default=50, type=int, help="display interval")
    parser.add_argument(
        "--workers", default=8, type=int, help="number of data loading workers"
    )
    # paths
    parser.add_argument(
        "--logging-dir",
        type=str,
        default="./train_logs",
        help="path to the directory in which to save the log file",
    )
    # use boundaries
    parser.add_argument(
        "--use-boundary",
        default=False,
        action="store_true",
        help="include hard border at the testing stage.",
    )

    args = parser.parse_args()
    return args


def calculateNorm2(model):
    para_norm = 0.0
    for p in model.get_parameters():
        para_norm += p.value().norm(ord=2)
    print("2-norm of the neural network: {:.4f}".format(para_norm**0.5))


def main():

    context.set_context(mode=context.GRAPH_MODE,
                        device_target="CPU", device_id=2)

    args = load_args()
    model = get_model_from_json(args)

    if args.modality == "video":
        model_path = "checkpoints/lipreading/ms_resnet18_dctcn_video.ckpt"
    else:
        model_path = "checkpoints/lipreading/ms_resnet18_dctcn_audio.ckpt"

    param_dict = ms.load_checkpoint(model_path)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)

    B = 84
    # input_np = np.random.rand(B, 1, 29, 88, 88)
    input_np = np.random.rand(B, 1, 640)
    # np.save("input_np.npy", input_np)
    # input_np = np.load("input_np.npy")
    input_tensor = ms.Tensor(input_np, ms.float32) # (bs, color channel-grey scale, seq-elngth, width, height)
    logits = model(input_tensor, [640] * B)

    # _, preds = ops.max(ops.softmax(logits, axis=1), axis=1)
    # sentence = labels.view_as(preds)

    print(logits)
    # save_params(model, "tools/parameters/lipreading_params.txt")


if __name__=="__main__":
    main()
