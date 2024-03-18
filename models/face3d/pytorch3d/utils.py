import copy
import inspect
import warnings
from mindspore import nn, ops
import mindspore as ms
import numpy as np
from typing import Any, List, Optional, Tuple, TypeVar, Union


def format_tensor(
    input,
    dtype: ms.dtype = ms.float32,
) -> ms.Tensor:
    """
    Helper function for converting a scalar value to a tensor.

    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
        dtype: data type for the input
        device: Device (as str or torch.device) on which the tensor should be placed.

    Returns:
        input_vec: torch tensor with optional added batch dimension.
    """
    if not ops.is_tensor(input):
        input = ops.Tensor(input, dtype=dtype)

    if input.dim() == 0:
        input = input.view(1)
    return input


def convert_to_tensors_and_broadcast(
    *args,
    dtype: ms.dtype = ms.float32,
):
    """
    Helper function to handle parsing an arbitrary number of inputs (*args)
    which all need to have the same batch dimension.
    The output is a list of tensors.

    Args:
        *args: an arbitrary number of inputs
            Each of the values in `args` can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N, K_i) or (1, K_i) where K_i are
                  an arbitrary number of dimensions which can vary for each
                  value in args. In this case each input is broadcast to a
                  tensor of shape (N, K_i)
        dtype: data type to use when creating new tensors.
        device: torch device on which the tensors should be placed.

    Output:
        args: A list of tensors of shape (N, K_i)
    """
    # Convert all inputs to tensors with a batch dimension
    args_1d = [format_tensor(c, dtype) for c in args]

    # Find broadcast size
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)

    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r" % sizes
            raise ValueError(msg)

        # Expand broadcast dim and keep non broadcast dims the same size
        expand_sizes = (N,) + (-1,) * len(c.shape[1:])
        args_Nd.append(ops.broadcast_to(c, expand_sizes))
        # args_Nd.append(c.expand(size=ms.Tensor(expand_sizes, ms.int32)))

    return args_Nd


class TensorAccessor(nn.Cell):
    """
    A helper class to be used with the __getitem__ method. This can be used for
    getting/setting the values for an attribute of a class at one particular
    index.  This is useful when the attributes of a class are batched tensors
    and one element in the batch needs to be modified.
    """

    def __init__(self, class_object, index: Union[int, slice]) -> None:
        """
        Args:
            class_object: this should be an instance of a class which has
                attributes which are tensors representing a batch of
                values.
            index: int/slice, an index indicating the position in the batch.
                In __setattr__ and __getattr__ only the value of class
                attributes at this index will be accessed.
        """
        self.__dict__["class_object"] = class_object
        self.__dict__["index"] = index

    def __setattr__(self, name: str, value: Any):
        """
        Update the attribute given by `name` to the value given by `value`
        at the index specified by `self.index`.

        Args:
            name: str, name of the attribute.
            value: value to set the attribute to.
        """
        v = getattr(self.class_object, name)
        if not ops.is_tensor(v):
            msg = "Can only set values on attributes which are tensors; got %r"
            raise AttributeError(msg % type(v))

        # Convert the attribute to a tensor if it is not a tensor.
        if not ops.is_tensor(value):
            value = ms.Tensor(
                value, dtype=v.dtype, requires_grad=v.requires_grad
            )

        # Check the shapes match the existing shape and the shape of the index.
        if v.dim() > 1 and value.dim() > 1 and value.shape[1:] != v.shape[1:]:
            msg = "Expected value to have shape %r; got %r"
            raise ValueError(msg % (v.shape, value.shape))
        if (
            v.dim() == 0
            and isinstance(self.index, slice)
            and len(value) != len(self.index)
        ):
            msg = "Expected value to have len %r; got %r"
            raise ValueError(msg % (len(self.index), len(value)))
        self.class_object.__dict__[name][self.index] = value

    def __getattr__(self, name: str):
        """
        Return the value of the attribute given by "name" on self.class_object
        at the index specified in self.index.

        Args:
            name: string of the attribute name
        """
        if hasattr(self.class_object, name):
            return self.class_object.__dict__[name][self.index]
        else:
            msg = "Attribute %s not found on %r"
            return AttributeError(msg % (name, self.class_object.__name__))


BROADCAST_TYPES = (float, int, list, tuple, ms.Tensor, np.ndarray)


class TensorProperties(nn.Cell):
    """
    A mix-in class for storing tensors as properties with helper methods.
    """

    def __init__(
        self,
        dtype: ms.dtype = ms.float32,
        **kwargs,
    ) -> None:
        """
        Args:
            dtype: data type to set for the inputs
            device: Device (as str or torch.device)
            kwargs: any number of keyword arguments. Any arguments which are
                of type (float/int/list/tuple/tensor/array) are broadcasted and
                other keyword arguments are set as attributes.
        """
        super().__init__()
        self._N = 0
        if kwargs is not None:

            # broadcast all inputs which are float/int/list/tuple/tensor/array
            # set as attributes anything else e.g. strings, bools
            args_to_broadcast = {}
            for k, v in kwargs.items():
                if v is None or isinstance(v, (str, bool)):
                    setattr(self, k, v)
                elif isinstance(v, BROADCAST_TYPES):
                    args_to_broadcast[k] = v
                else:
                    msg = "Arg %s with type %r is not broadcastable"
                    warnings.warn(msg % (k, type(v)))

            names = args_to_broadcast.keys()
            # convert from type dict.values to tuple
            values = tuple(v for v in args_to_broadcast.values())

            if len(values) > 0:
                broadcasted_values = convert_to_tensors_and_broadcast(
                    *values
                )

                # Set broadcasted values as attributes on self.
                for i, n in enumerate(names):
                    setattr(self, n, broadcasted_values[i])
                    if self._N == 0:
                        self._N = broadcasted_values[i].shape[0]

    def __len__(self) -> int:
        return self._N

    def isempty(self) -> bool:
        return self._N == 0

    def __getitem__(self, index: Union[int, slice]):
        """

        Args:
            index: an int or slice used to index all the fields.

        Returns:
            if `index` is an index int/slice return a TensorAccessor class
            with getattribute/setattribute methods which return/update the value
            at the index in the original class.
        """
        if isinstance(index, (int, slice)):
            return TensorAccessor(class_object=self, index=index)

        msg = "Expected index of type int or slice; got %r"
        raise ValueError(msg % type(index))

    def clone(self, other) -> "TensorProperties":
        """
        Update the tensor properties of other with the cloned properties of self.
        """
        for k in dir(self):
            v = getattr(self, k)
            if inspect.ismethod(v) or k.startswith("__") or type(v) is TypeVar:
                continue
            if ops.is_tensor(v):
                v_clone = v.copy()
            else:
                v_clone = copy.deepcopy(v)
            setattr(other, k, v_clone)
        return other

    def gather_props(self, batch_idx) -> "TensorProperties":
        """
        This is an in place operation to reformat all tensor class attributes
        based on a set of given indices using torch.gather. This is useful when
        attributes which are batched tensors e.g. shape (N, 3) need to be
        multiplied with another tensor which has a different first dimension
        e.g. packed vertices of shape (V, 3).

        Example

        .. code-block:: python

            self.specular_color = (N, 3) tensor of specular colors for each mesh

        A lighting calculation may use

        .. code-block:: python

            verts_packed = meshes.verts_packed()  # (V, 3)

        To multiply these two tensors the batch dimension needs to be the same.
        To achieve this we can do

        .. code-block:: python

            batch_idx = meshes.verts_packed_to_mesh_idx()  # (V)

        This gives index of the mesh for each vertex in verts_packed.

        .. code-block:: python

            self.gather_props(batch_idx)
            self.specular_color = (V, 3) tensor with the specular color for
                                     each packed vertex.

        torch.gather requires the index tensor to have the same shape as the
        input tensor so this method takes care of the reshaping of the index
        tensor to use with class attributes with arbitrary dimensions.

        Args:
            batch_idx: shape (B, ...) where `...` represents an arbitrary
                number of dimensions

        Returns:
            self with all properties reshaped. e.g. a property with shape (N, 3)
            is transformed to shape (B, 3).
        """
        # Iterate through the attributes of the class which are tensors.
        for k in dir(self):
            v = getattr(self, k)
            if ops.is_tensor(v):
                if v.shape[0] > 1:
                    # There are different values for each batch element
                    # so gather these using the batch_idx.
                    # First clone the input batch_idx tensor before
                    # modifying it.
                    _batch_idx = batch_idx.copy()
                    idx_dims = _batch_idx.shape
                    tensor_dims = v.shape
                    if len(idx_dims) > len(tensor_dims):
                        msg = "batch_idx cannot have more dimensions than %s. "
                        msg += "got shape %r and %s has shape %r"
                        raise ValueError(msg % (k, idx_dims, k, tensor_dims))
                    if idx_dims != tensor_dims:
                        # To use torch.gather the index tensor (_batch_idx) has
                        # to have the same shape as the input tensor.
                        new_dims = len(tensor_dims) - len(idx_dims)
                        new_shape = idx_dims + (1,) * new_dims
                        expand_dims = (-1,) + tensor_dims[1:]
                        _batch_idx = _batch_idx.view(*new_shape)
                        _batch_idx = _batch_idx.expand(*expand_dims)

                    v = v.gather(0, _batch_idx)
                    setattr(self, k, v)
        return self


def parse_image_size(
    image_size: Union[List[int], Tuple[int, int], int]
) -> Tuple[int, int]:
    """
    Args:
        image_size: A single int (for square images) or a tuple/list of two ints.

    Returns:
        A tuple of two ints.

    Throws:
        ValueError if got more than two ints, any negative numbers or non-ints.
    """
    if not isinstance(image_size, (tuple, list)):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("Image size can only be a tuple/list of (H, W)")
    if not all(i > 0 for i in image_size):
        raise ValueError(
            "Image sizes must be greater than 0; got %d, %d" % image_size)
    if not all(isinstance(type(i), int) for i in image_size):
        raise ValueError(
            "Image sizes must be integers; got %f, %f" % image_size)
    return tuple(image_size)


def padded_to_list(
    x: ms.Tensor,
    split_size=None,
):
    r"""
    Transforms a padded tensor of shape (N, S_1, S_2, ..., S_D) into a list
    of N tensors of shape:
    - (Si_1, Si_2, ..., Si_D) where (Si_1, Si_2, ..., Si_D) is specified in split_size(i)
    - or (S_1, S_2, ..., S_D) if split_size is None
    - or (Si_1, S_2, ..., S_D) if split_size(i) is an integer.

    Args:
      x: tensor
      split_size: optional 1D or 2D list/tuple of ints defining the number of
        items for each tensor.

    Returns:
      x_list: a list of tensors sharing the memory with the input.
    """
    x_list = list(ops.unbind(x, 0))

    if split_size is None:
        return x_list

    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError(
            "Split size must be of same length as inputs first dimension")

    for i in range(N):
        if isinstance(split_size[i], list):
            slices = tuple(slice(0, s) for s in split_size[i])  # pyre-ignore
            x_list[i] = x_list[i][slices]
        else:
            x_list[i] = x_list[i][: split_size[i]]

    return x_list


def list_to_packed(x: List[ms.Tensor]):
    r"""
    Transforms a list of N tensors each of shape (Mi, K, ...) into a single
    tensor of shape (sum(Mi), K, ...).

    Args:
      x: list of tensors.

    Returns:
        4-element tuple containing

        - **x_packed**: tensor consisting of packed input tensors along the
          1st dimension.
        - **num_items**: tensor of shape N containing Mi for each element in x.
        - **item_packed_first_idx**: tensor of shape N indicating the index of
          the first item belonging to the same element in the original list.
        - **item_packed_to_list_idx**: tensor of shape sum(Mi) containing the
          index of the element in the list the item belongs to.
    """
    N = len(x)
    num_items = ops.zeros(N, dtype=ms.int64)
    item_packed_first_idx = ops.zeros(N, dtype=ms.int64)
    item_packed_to_list_idx = []
    cur = 0
    for i, y in enumerate(x):
        num = len(y)
        num_items[i] = num
        item_packed_first_idx[i] = cur
        item_packed_to_list_idx.append(
            ops.full((num,), i, dtype=ms.int64)
        )
        cur += num

    x_packed = ops.cat(x, axis=0)
    item_packed_to_list_idx = ops.cat(item_packed_to_list_idx, axis=0)

    return x_packed, num_items, item_packed_first_idx, item_packed_to_list_idx


def packed_to_list(x: ms.Tensor, split_size: Union[list, int]):
    r"""
    Transforms a tensor of shape (sum(Mi), K, L, ...) to N set of tensors of
    shape (Mi, K, L, ...) where Mi's are defined in split_size

    Args:
      x: tensor
      split_size: list, tuple or int defining the number of items for each tensor
        in the output list.

    Returns:
      x_list: A list of Tensors
    """
    return x.split(split_size, axis=0)


def padded_to_packed(
    x: ms.Tensor,
    split_size: Union[list, tuple, None] = None,
    pad_value: Union[float, int, None] = None,
):
    r"""
    Transforms a padded tensor of shape (N, M, K) into a packed tensor
    of shape:
     - (sum(Mi), K) where (Mi, K) are the dimensions of
        each of the tensors in the batch and Mi is specified by split_size(i)
     - (N*M, K) if split_size is None

    Support only for 3-dimensional input tensor and 1-dimensional split size.

    Args:
      x: tensor
      split_size: list, tuple or int defining the number of items for each tensor
        in the output list.
      pad_value: optional value to use to filter the padded values in the input
        tensor.

    Only one of split_size or pad_value should be provided, or both can be None.

    Returns:
      x_packed: a packed tensor.
    """
    if x.ndim != 3:
        raise ValueError("Supports only 3-dimensional input tensors")

    N, M, D = x.shape

    if split_size is not None and pad_value is not None:
        raise ValueError(
            "Only one of split_size or pad_value should be provided.")

    x_packed = x.reshape(-1, D)  # flatten padded

    if pad_value is None and split_size is None:
        return x_packed

    # Convert to packed using pad value
    if pad_value is not None:
        mask = x_packed.ne(pad_value).any(-1)
        x_packed = x_packed[mask]
        return x_packed

    # Convert to packed using split sizes
    # pyre-fixme[6]: Expected `Sized` for 1st param but got `Union[None,
    #  List[typing.Any], typing.Tuple[typing.Any, ...]]`.
    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError(
            "Split size must be of same length as inputs first dimension")

    # pyre-fixme[16]: `None` has no attribute `__iter__`.
    if not all(isinstance(i, int) for i in split_size):
        raise ValueError(
            "Support only 1-dimensional unbinded tensor. \
                Split size for more dimensions provided"
        )

    padded_to_packed_idx = ops.cat(
        [
            ops.arange(v, dtype=ms.int64) + i * M
            # pyre-fixme[6]: Expected `Iterable[Variable[_T]]` for 1st param but got
            #  `Union[None, List[typing.Any], typing.Tuple[typing.Any, ...]]`.
            for (i, v) in enumerate(split_size)
        ],
        axis=0,
    )

    return x_packed[padded_to_packed_idx]
