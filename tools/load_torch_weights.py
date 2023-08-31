import torch
import mindspore as ms


model_path = "gfpgan/weights/alignment_WFLW_4HG.pth"
model = torch.load(model_path, map_location="cpu")

param_list = []
for p_name, value in model["state_dict"].items():
    datatype = value.dtype
    line = p_name + "#" + str(datatype)
    param_list.append(line)

with open("alignment_WFLW_4HG_params.txt", "w") as f:
    f.write("\n".join(param_list))

# ms_param_dict = {}
# for p_name, value in model["state_dict"].items():
#     weights = value.numpy()
#     weights_tensor = ms.Tensor(weights, dtype=ms.float32)
#     ms_param_dict[p_name] = weights_tensor
