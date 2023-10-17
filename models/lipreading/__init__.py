import os
import json
import mindspore as ms
from models.lipreading.model import Lipreading


def load_json(json_fp):
    assert os.path.isfile(
        json_fp
    ), "Error loading JSON. File provided does not exist, cannot read: {}".format(
        json_fp
    )
    with open(json_fp, "r") as f:
        json_content = json.load(f)
    return json_content


def get_model_from_json(args):
    assert args.config_path.endswith(".json") and os.path.isfile(
        args.config_path
    ), f"'.json' config path does not exist. Path input: {args.config_path}"
    args_loaded = load_json(args.config_path)
    args.backbone_type = args_loaded["backbone_type"]
    args.width_mult = args_loaded["width_mult"]
    args.relu_type = args_loaded["relu_type"]
    args.use_boundary = args_loaded.get("use_boundary", False)

    if args_loaded.get("tcn_num_layers", ""):
        tcn_options = {
            "num_layers": args_loaded["tcn_num_layers"],
            "kernel_size": args_loaded["tcn_kernel_size"],
            "dropout": args_loaded["tcn_dropout"],
            "dwpw": args_loaded["tcn_dwpw"],
            "width_mult": args_loaded["tcn_width_mult"],
        }
    else:
        tcn_options = {}
    if args_loaded.get("densetcn_block_config", ""):
        densetcn_options = {
            "block_config": args_loaded["densetcn_block_config"],
            "growth_rate_set": args_loaded["densetcn_growth_rate_set"],
            "reduced_size": args_loaded["densetcn_reduced_size"],
            "kernel_size_set": args_loaded["densetcn_kernel_size_set"],
            "dilation_size_set": args_loaded["densetcn_dilation_size_set"],
            "squeeze_excitation": args_loaded["densetcn_se"],
            "dropout": args_loaded["densetcn_dropout"],
        }
    else:
        densetcn_options = {}

    model = Lipreading(
        modality=args.modality,
        num_classes=args.num_classes,
        tcn_options=tcn_options,
        densetcn_options=densetcn_options,
        backbone_type=args.backbone_type,
        relu_type=args.relu_type,
        width_mult=args.width_mult,
        use_boundary=args.use_boundary,
        extract_feats=args.extract_feats,
    )
    return model


def get_lipreading_model(modality):
    args = {
        "modality": modality,
        "config_path": "config/lipreading/lrw_resnet18_dctcn.json",
        "num_classes": 500,
        "extract_feats": False,
    }
    args = AttributeDict(args)
    model = get_model_from_json(args)

    if args.modality == "video":
        model_path = "checkpoints/lipreading/ms_resnet18_dctcn_video.ckpt"
    else:
        model_path = "checkpoints/lipreading/ms_resnet18_dctcn_audio.ckpt"

    param_dict = ms.load_checkpoint(model_path)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)
    return model


class AttributeDict(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
