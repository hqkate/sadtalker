import mindspore.ops as ops
from mindspore import nn
from tqdm import tqdm
from mindspore.common import dtype as mstype


class NetWithLossWrapper(nn.Cell):
    """
    A universal wrapper for any network with any loss.

    Args:
        net (nn.Cell): network
        loss_fn: loss function
        input_indices: The indices of the data tuples which will be fed into the network.
            If it is None, then the first item will be fed only.
        label_indices: The indices of the data tuples which will be fed into the loss function.
            If it is None, then the remaining items will be fed.
    """

    def __init__(self, net, loss_fn):
        super().__init__(auto_prefix=False)
        self._net = net
        self._loss_fn = loss_fn

    def construct(self, *args):
        """
        Args:
            args (Tuple): contains network inputs, labels (given by data loader)
        Returns:
            loss_val (Tensor): loss value
        """
        outputs = self._net(args[0])

        loss_val = self._loss_fn(*outputs)

        return loss_val


def select_inputs_by_indices(inputs, indices):
    new_inputs = list()
    for x in indices:
        new_inputs.append(inputs[x])
    return new_inputs
