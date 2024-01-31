import mindspore as ms
from mindspore import context
from models.wav2lip.wav2lip import Wav2Lip


def save_params(model, out_path):
    param_list = []
    for param in model.get_parameters():
        line = param.name + "#" + \
            str(param.dtype) + "#" + str(tuple(param.shape))
        param_list.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(param_list))


context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend", device_id=7)


model = Wav2Lip()
save_params(model, "tools/parameters/wav2lip_params.txt")
