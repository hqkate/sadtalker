import pickle
import mindspore as ms
from mindspore import context


def param_convert(ms_params, pt_params, ckpt_path, extra_dict=None):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}

    if extra_dict:
        bn_ms2pt.update(extra_dict)

    new_params_list = []
    for ms_param in ms_params:
        # 在参数列表中，只有包含bn和downsample.1的参数是BatchNorm算子的参数

        # import pdb; pdb.set_trace()

        # if "conv_block.1." in ms_param.name:
        if any(x in ms_param.name for x in bn_ms2pt.keys()):
            # ms_param_item = ms_param.name.split(".")
            # pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            # pt_param = ".".join(pt_param_item)

            param_name = ms_param.name
            for k, v in bn_ms2pt.items():
                param_name = param_name.replace(k, v)
            pt_param = param_name

            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_param.data.shape:
                ms_value = pt_params[pt_param]
                new_params_list.append(
                    {"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32)})
            else:
                print(ms_param.name, "not match in pt_params")
        # 其他参数
        else:
            # 如找到参数对应且shape一致，加入到参数列表
            if ms_param.name in pt_params and tuple(pt_params[ms_param.name].shape) == tuple(ms_param.data.shape):
                ms_value = pt_params[ms_param.name]
                new_params_list.append(
                    {"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32)})
            elif "netD_motion.seq" in ms_param.name:
                ms_value = pt_params[ms_param.name]
                new_params_list.append(
                    {"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32).unsqueeze(2)})
            else:
                print(ms_param.name, "not match in pt_params")
    # 保存成MindSpore的checkpoint
    ms.save_checkpoint(new_params_list, ckpt_path)


def convert_audio2exp():
    from models.audio2exp.expnet import ExpNet
    from models.audio2exp.audio2exp import Audio2Exp
    from yacs.config import CfgNode as CN

    fcfg_exp = open("config/audio2exp.yaml")
    cfg_exp = CN.load_cfg(fcfg_exp)
    cfg_exp.freeze()
    fcfg_exp.close()

    netG = ExpNet()
    audio2exp_model = Audio2Exp(netG, cfg_exp, prepare_training_loss=False)
    ms_params = audio2exp_model.get_parameters()

    with open("../SadTalker/pt_weights_audio2exp.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict, "checkpoints/ms/ms_audio2exp.ckpt")


def convert_audio2pose():

    from models.audio2pose.audio2pose import Audio2Pose
    from yacs.config import CfgNode as CN

    fcfg_pose = open("config/audio2pose.yaml")
    cfg_pose = CN.load_cfg(fcfg_pose)
    cfg_pose.freeze()
    fcfg_pose.close()

    extra_dict = {
        "mlp.0": "MLP.L0",
        "mlp.2": "MLP.L1",
    }

    audio2exp_model = Audio2Pose(cfg_pose)
    ms_params = audio2exp_model.get_parameters()

    with open("../SadTalker/pt_weights_audio2pose.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/ms/ms_audio2pose.ckpt", extra_dict)


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", device_id=6)
    convert_audio2pose()
