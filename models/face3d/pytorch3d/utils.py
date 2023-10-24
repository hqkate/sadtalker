import copy
import inspect
import warnings
from mindspore import nn, ops
import mindspore as ms
from typing import Any, List, Optional, Tuple, TypeVar, Union


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
                v_clone = v.clone()
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
                    _batch_idx = batch_idx.clone()
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
    if not all(type(i) == int for i in image_size):
        raise ValueError(
            "Image sizes must be integers; got %f, %f" % image_size)
    return tuple(image_size)
