import os
import math
import warnings
import numpy as np
import mindspore as ms
from mindspore import ops
from typing import List, Optional, Union


def _broadcast_bmm(a, b) -> ms.Tensor:
    """
    Batch multiply two matrices and broadcast if necessary.

    Args:
        a: torch tensor of shape (P, K) or (M, P, K)
        b: torch tensor of shape (N, K, K)

    Returns:
        a and b broadcast multiplied. The output batch dimension is max(N, M).

    To broadcast transforms across a batch dimension if M != N then
    expect that either M = 1 or N = 1. The tensor with batch dimension 1 is
    expanded to have shape N or M.
    """
    if a.dim() == 2:
        a = a[None]
    if len(a) != len(b):
        if not ((len(a) == 1) or (len(b) == 1)):
            msg = "Expected batch dim for bmm to be equal or 1; got %r, %r"
            raise ValueError(msg % (a.shape, b.shape))
        if len(a) == 1:
            a = ops.broadcast_to(a, (len(b), -1, -1))
        if len(b) == 1:
            b = ops.broadcast_to(b, (len(a), -1, -1))
    return a.bmm(b)


class Transform3d:
    """
    A Transform3d object encapsulates a batch of N 3D transformations, and knows
    how to transform points and normal vectors. Suppose that t is a Transform3d;
    then we can do the following:

    .. code-block:: python

        N = len(t)
        points = torch.randn(N, P, 3)
        normals = torch.randn(N, P, 3)
        points_transformed = t.transform_points(points)    # => (N, P, 3)
        normals_transformed = t.transform_normals(normals)  # => (N, P, 3)


    BROADCASTING
    Transform3d objects supports broadcasting. Suppose that t1 and tN are
    Transform3d objects with len(t1) == 1 and len(tN) == N respectively. Then we
    can broadcast transforms like this:

    .. code-block:: python

        t1.transform_points(torch.randn(P, 3))     # => (P, 3)
        t1.transform_points(torch.randn(1, P, 3))  # => (1, P, 3)
        t1.transform_points(torch.randn(M, P, 3))  # => (M, P, 3)
        tN.transform_points(torch.randn(P, 3))     # => (N, P, 3)
        tN.transform_points(torch.randn(1, P, 3))  # => (N, P, 3)


    COMBINING TRANSFORMS
    Transform3d objects can be combined in two ways: composing and stacking.
    Composing is function composition. Given Transform3d objects t1, t2, t3,
    the following all compute the same thing:

    .. code-block:: python

        y1 = t3.transform_points(t2.transform_points(t1.transform_points(x)))
        y2 = t1.compose(t2).compose(t3).transform_points(x)
        y3 = t1.compose(t2, t3).transform_points(x)


    Composing transforms should broadcast.

    .. code-block:: python

        if len(t1) == 1 and len(t2) == N, then len(t1.compose(t2)) == N.

    We can also stack a sequence of Transform3d objects, which represents
    composition along the batch dimension; then the following should compute the
    same thing.

    .. code-block:: python

        N, M = len(tN), len(tM)
        xN = torch.randn(N, P, 3)
        xM = torch.randn(M, P, 3)
        y1 = torch.cat([tN.transform_points(xN), tM.transform_points(xM)], dim=0)
        y2 = tN.stack(tM).transform_points(torch.cat([xN, xM], dim=0))

    BUILDING TRANSFORMS
    We provide convenience methods for easily building Transform3d objects
    as compositions of basic transforms.

    .. code-block:: python

        # Scale by 0.5, then translate by (1, 2, 3)
        t1 = Transform3d().scale(0.5).translate(1, 2, 3)

        # Scale each axis by a different amount, then translate, then scale
        t2 = Transform3d().scale(1, 3, 3).translate(2, 3, 1).scale(2.0)

        t3 = t1.compose(t2)
        tN = t1.stack(t3, t3)


    BACKPROP THROUGH TRANSFORMS
    When building transforms, we can also parameterize them by Torch tensors;
    in this case we can backprop through the construction and application of
    Transform objects, so they could be learned via gradient descent or
    predicted by a neural network.

    .. code-block:: python

        s1_params = torch.randn(N, requires_grad=True)
        t_params = torch.randn(N, 3, requires_grad=True)
        s2_params = torch.randn(N, 3, requires_grad=True)

        t = Transform3d().scale(s1_params).translate(t_params).scale(s2_params)
        x = torch.randn(N, 3)
        y = t.transform_points(x)
        loss = compute_loss(y)
        loss.backward()

        with torch.no_grad():
            s1_params -= lr * s1_params.grad
            t_params -= lr * t_params.grad
            s2_params -= lr * s2_params.grad

    CONVENTIONS
    We adopt a right-hand coordinate system, meaning that rotation about an axis
    with a positive angle results in a counter clockwise rotation.

    This class assumes that transformations are applied on inputs which
    are row vectors. The internal representation of the Nx4x4 transformation
    matrix is of the form:

    .. code-block:: python

        M = [
                [Rxx, Ryx, Rzx, 0],
                [Rxy, Ryy, Rzy, 0],
                [Rxz, Ryz, Rzz, 0],
                [Tx,  Ty,  Tz,  1],
            ]

    To apply the transformation to points, which are row vectors, the latter are
    converted to homogeneous (4D) coordinates and right-multiplied by the M matrix:

    .. code-block:: python

        points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
        [transformed_points, 1] ∝ [points, 1] @ M

    """

    def __init__(
        self,
        dtype: ms.dtype = ms.float32,
        matrix: Optional[ms.Tensor] = None,
    ) -> None:
        """
        Args:
            dtype: The data type of the transformation matrix.
                to be used if `matrix = None`.
            device: The device for storing the implemented transformation.
                If `matrix != None`, uses the device of input `matrix`.
            matrix: A tensor of shape (4, 4) or of shape (minibatch, 4, 4)
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
                the specified `device` and `dtype`.
        """
        if matrix is None:
            self._matrix = ops.eye(4, dtype=dtype).view(1, 4, 4)
        else:
            if matrix.ndim not in (2, 3):
                raise ValueError(
                    '"matrix" has to be a 2- or a 3-dimensional tensor.')
            if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
                raise ValueError(
                    '"matrix" has to be a tensor of shape (minibatch, 4, 4) or (4, 4).'
                )
            # set dtype and device from matrix
            dtype = matrix.dtype
            self._matrix = matrix.view(-1, 4, 4)

        self._transforms = []  # store transforms to compose
        self._lu = None
        self.dtype = dtype

    def __len__(self) -> int:
        return self.get_matrix().shape[0]

    def __getitem__(
        self, index: Union[int, List[int], slice, ms.Tensor]
    ) -> "Transform3d":
        """
        Args:
            index: Specifying the index of the transform to retrieve.
                Can be an int, slice, list of ints, boolean, long tensor.
                Supports negative indices.

        Returns:
            Transform3d object with selected transforms. The tensors are not cloned.
        """
        if isinstance(index, int):
            index = [index]
        return self.__class__(matrix=self.get_matrix()[index])

    def compose(self, *others: "Transform3d") -> "Transform3d":
        """
        Return a new Transform3d representing the composition of self with the
        given other transforms, which will be stored as an internal list.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d with the stored transforms
        """
        out = Transform3d(dtype=self.dtype)
        out._matrix = self._matrix.copy()
        for other in others:
            if not isinstance(other, Transform3d):
                msg = "Only possible to compose Transform3d objects; got %s"
                raise ValueError(msg % type(other))
        out._transforms = self._transforms + list(others)
        return out

    def get_matrix(self) -> ms.Tensor:
        """
        Returns a 4×4 matrix corresponding to each transform in the batch.

        If the transform was composed from others, the matrix for the composite
        transform will be returned.
        For example, if self.transforms contains transforms t1, t2, and t3, and
        given a set of points x, the following should be true:

        .. code-block:: python

            y1 = t1.compose(t2, t3).transform(x)
            y2 = t3.transform(t2.transform(t1.transform(x)))
            y1.get_matrix() == y2.get_matrix()

        Where necessary, those transforms are broadcast against each other.

        Returns:
            A (N, 4, 4) batch of transformation matrices representing
                the stored transforms. See the class documentation for the conventions.
        """
        composed_matrix = self._matrix.copy()
        if len(self._transforms) > 0:
            for other in self._transforms:
                other_matrix = other.get_matrix()
                composed_matrix = _broadcast_bmm(composed_matrix, other_matrix)
        return composed_matrix

    def get_se3_log(self, eps: float = 1e-4, cos_bound: float = 1e-4) -> ms.Tensor:
        """
        Returns a 6D SE(3) log vector corresponding to each transform in the batch.

        In the SE(3) logarithmic representation SE(3) matrices are
        represented as 6-dimensional vectors `[log_translation | log_rotation]`,
        i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

        The conversion from the 4x4 SE(3) matrix `transform` to the
        6D representation `log_transform = [log_translation | log_rotation]`
        is done as follows::

            log_transform = log(transform.get_matrix())
            log_translation = log_transform[3, :3]
            log_rotation = inv_hat(log_transform[:3, :3])

        where `log` is the matrix logarithm
        and `inv_hat` is the inverse of the Hat operator [2].

        See the docstring for `se3.se3_log_map` and [1], Sec 9.4.2. for more
        detailed description.

        Args:
            eps: A threshold for clipping the squared norm of the rotation logarithm
                to avoid division by zero in the singular case.
            cos_bound: Clamps the cosine of the rotation angle to
                [-1 + cos_bound, 3 - cos_bound] to avoid non-finite outputs.
                The non-finite outputs can be caused by passing small rotation angles
                to the `acos` function in `so3_rotation_angle` of `so3_log_map`.

        Returns:
            A (N, 6) tensor, rows of which represent the individual transforms
            stored in the object as SE(3) logarithms.

        Raises:
            ValueError if the stored transform is not Euclidean (e.g. R is not a rotation
                matrix or the last column has non-zeros in the first three places).

        [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
        [2] https://en.wikipedia.org/wiki/Hat_operator
        """
        return se3_log_map(self.get_matrix(), eps, cos_bound)

    def _get_matrix_inverse(self) -> ms.Tensor:
        """
        Return the inverse of self._matrix.
        """
        return ops.inverse(self._matrix)

    def inverse(self, invert_composed: bool = False) -> "Transform3d":
        """
        Returns a new Transform3d object that represents an inverse of the
        current transformation.

        Args:
            invert_composed:
                - True: First compose the list of stored transformations
                  and then apply inverse to the result. This is
                  potentially slower for classes of transformations
                  with inverses that can be computed efficiently
                  (e.g. rotations and translations).
                - False: Invert the individual stored transformations
                  independently without composing them.

        Returns:
            A new Transform3d object containing the inverse of the original
            transformation.
        """

        tinv = Transform3d(dtype=self.dtype)

        if invert_composed:
            # first compose then invert
            tinv._matrix = ops.inverse(self.get_matrix())
        else:
            # self._get_matrix_inverse() implements efficient inverse
            # of self._matrix
            i_matrix = self._get_matrix_inverse()

            # 2 cases:
            if len(self._transforms) > 0:
                # a) Either we have a non-empty list of transforms:
                # Here we take self._matrix and append its inverse at the
                # end of the reverted _transforms list. After composing
                # the transformations with get_matrix(), this correctly
                # right-multiplies by the inverse of self._matrix
                # at the end of the composition.
                tinv._transforms = [t.inverse()
                                    for t in reversed(self._transforms)]
                last = Transform3d(dtype=self.dtype)
                last._matrix = i_matrix
                tinv._transforms.append(last)
            else:
                # b) Or there are no stored transformations
                # we just set inverted matrix
                tinv._matrix = i_matrix

        return tinv

    def stack(self, *others: "Transform3d") -> "Transform3d":
        """
        Return a new batched Transform3d representing the batch elements from
        self and all the given other transforms all batched together.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d.
        """
        transforms = [self] + list(others)
        matrix = ops.cat([t.get_matrix() for t in transforms], axis=0)
        out = Transform3d(dtype=self.dtype)
        out._matrix = matrix
        return out

    def transform_points(self, points, eps: Optional[float] = None) -> ms.Tensor:
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.

        Args:
            points: Tensor of shape (P, 3) or (N, P, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before performing the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, P, 3) or (P, 3) depending
            on the dimensions of the transform
        """
        points_batch = points.copy()
        if points_batch.dim() == 2:
            points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        if points_batch.dim() != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % repr(points.shape))

        N, P, _3 = points_batch.shape
        ones = ops.ones((N, P, 1), dtype=points.dtype)
        points_batch = ops.cat([points_batch, ones], axis=2)

        composed_matrix = self.get_matrix()
        points_out = _broadcast_bmm(points_batch, composed_matrix)
        denom = points_out[..., 3:]  # denominator
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0).astype(denom)
            denom = denom_sign * ops.clamp(denom.abs(), eps)
        points_out = points_out[..., :3] / denom

        # When transform is (1, 4, 4) and points is (P, 3) return
        # points_out of shape (P, 3)
        if points_out.shape[0] == 1 and points.dim() == 2:
            points_out = points_out.reshape(points.shape)

        return points_out

    def transform_normals(self, normals) -> ms.Tensor:
        """
        Use this transform to transform a set of normal vectors.

        Args:
            normals: Tensor of shape (P, 3) or (N, P, 3)

        Returns:
            normals_out: Tensor of shape (P, 3) or (N, P, 3) depending
            on the dimensions of the transform
        """
        if normals.dim() not in [2, 3]:
            msg = "Expected normals to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % (normals.shape,))
        composed_matrix = self.get_matrix()

        # TODO: inverse is bad! Solve a linear system instead
        mat = composed_matrix[:, :3, :3]
        normals_out = _broadcast_bmm(normals, mat.transpose(1, 2).inverse())

        # This doesn't pass unit tests. TODO investigate further
        # if self._lu is None:
        #     self._lu = self._matrix[:, :3, :3].transpose(1, 2).lu()
        # normals_out = normals.lu_solve(*self._lu)

        # When transform is (1, 4, 4) and normals is (P, 3) return
        # normals_out of shape (P, 3)
        if normals_out.shape[0] == 1 and normals.dim() == 2:
            normals_out = normals_out.reshape(normals.shape)

        return normals_out

    def translate(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            Translate(dtype=self.dtype, *args, **kwargs)
        )

    def scale(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            Scale(dtype=self.dtype, *args, **kwargs)
        )

    def rotate(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            Rotate(dtype=self.dtype, *args, **kwargs)
        )

    def rotate_axis_angle(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            RotateAxisAngle(dtype=self.dtype, *args, **kwargs)
        )

    def clone(self) -> "Transform3d":
        """
        Deep copy of Transforms object. All internal tensors are cloned
        individually.

        Returns:
            new Transforms object.
        """
        other = Transform3d(dtype=self.dtype)
        if self._lu is not None:
            other._lu = [elem.copy() for elem in self._lu]
        other._matrix = self._matrix.copy()
        other._transforms = [t.copy() for t in self._transforms]
        return other


class Translate(Transform3d):
    def __init__(
        self,
        x,
        y=None,
        z=None,
        dtype: ms.dtype = ms.float32,
    ) -> None:
        """
        Create a new Transform3d representing 3D translations.

        Option I: Translate(xyz, dtype=torch.float32, device='cpu')
            xyz should be a tensor of shape (N, 3)

        Option II: Translate(x, y, z, dtype=torch.float32, device='cpu')
            Here x, y, and z will be broadcast against each other and
            concatenated to form the translation. Each can be:
                - A python scalar
                - A torch scalar
                - A 1D torch tensor
        """
        xyz = _handle_input(x, y, z, dtype, "Translate")
        super().__init__(dtype=dtype)
        N = xyz.shape[0]

        mat = ops.eye(4, dtype=dtype)
        mat = mat.view(1, 4, 4).repeat(N, axis=1)
        # mat[:, 3, :3] = xyz # TODO!!!
        self._matrix = mat

    def _get_matrix_inverse(self) -> ms.Tensor:
        """
        Return the inverse of self._matrix.
        """
        inv_mask = self._matrix.new_ones([1, 4, 4])
        inv_mask[0, 3, :3] = -1.0
        i_matrix = self._matrix * inv_mask
        return i_matrix


class Scale(Transform3d):
    def __init__(
        self,
        x,
        y=None,
        z=None,
        dtype: ms.dtype = ms.float32,
    ) -> None:
        """
        A Transform3d representing a scaling operation, with different scale
        factors along each coordinate axis.

        Option I: Scale(s, dtype=torch.float32, device='cpu')
            s can be one of
                - Python scalar or torch scalar: Single uniform scale
                - 1D torch tensor of shape (N,): A batch of uniform scale
                - 2D torch tensor of shape (N, 3): Scale differently along each axis

        Option II: Scale(x, y, z, dtype=torch.float32, device='cpu')
            Each of x, y, and z can be one of
                - python scalar
                - torch scalar
                - 1D torch tensor
        """
        xyz = _handle_input(x, y, z, dtype, "scale", allow_singleton=True)
        super().__init__(dtype=dtype)
        N = xyz.shape[0]

        # TODO: Can we do this all in one go somehow?
        mat = ops.eye(4, dtype=dtype)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, 0, 0] = xyz[:, 0]
        mat[:, 1, 1] = xyz[:, 1]
        mat[:, 2, 2] = xyz[:, 2]
        self._matrix = mat

    def _get_matrix_inverse(self) -> ms.Tensor:
        """
        Return the inverse of self._matrix.
        """
        xyz = ops.stack([self._matrix[:, i, i] for i in range(4)], axis=1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        ixyz = 1.0 / xyz
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `float`.
        imat = ops.diag_embed(ixyz, dim1=1, dim2=2)
        return imat


class Rotate(Transform3d):
    def __init__(
        self,
        xR: ms.Tensor,
        dtype: ms.dtype = ms.float32,
        orthogonal_tol: float = 1e-5,
    ) -> None:
        """
        Create a new Transform3d representing 3D rotation using a rotation
        matrix as the input.

        Args:
            R: a tensor of shape (3, 3) or (N, 3, 3)
            orthogonal_tol: tolerance for the test of the orthogonality of R

        """
        super().__init__(dtype=dtype)
        if xR.dim() == 2:
            xR = xR[None]
        if xR.shape[-2:] != (3, 3):
            msg = "R must have shape (3, 3) or (N, 3, 3); got %s"
            raise ValueError(msg % repr(xR.shape))
        if os.environ.get("PYTORCH3D_CHECK_ROTATION_MATRICES", "0") == "1":
            # Note: aten::all_close in the check is computationally slow, so we
            # only run the check when PYTORCH3D_CHECK_ROTATION_MATRICES is on.
            _check_valid_rotation_matrix(xR, tol=orthogonal_tol)
        N = xR.shape[0]
        mat = ops.eye(4, dtype=dtype)
        mat = mat.view(1, 4, 4).repeat(N, axis=1)
        # mat[:, :3, :3] = xR # TODO!!
        self._matrix = mat

    def _get_matrix_inverse(self) -> ms.Tensor:
        """
        Return the inverse of self._matrix.
        """
        return self._matrix.permute(0, 2, 1).contiguous()


class RotateAxisAngle(Rotate):
    def __init__(
        self,
        angle,
        axis: str = "X",
        degrees: bool = True,
        dtype: ms.dtype = ms.float32,
    ) -> None:
        """
        Create a new Transform3d representing 3D rotation about an axis
        by an angle.

        Assuming a right-hand coordinate system, positive rotation angles result
        in a counter clockwise rotation.

        Args:
            angle:
                - A torch tensor of shape (N,)
                - A python scalar
                - A torch scalar
            axis:
                string: one of ["X", "Y", "Z"] indicating the axis about which
                to rotate.
                NOTE: All batch elements are rotated about the same axis.
        """
        axis = axis.upper()
        if axis not in ["X", "Y", "Z"]:
            msg = "Expected axis to be one of ['X', 'Y', 'Z']; got %s"
            raise ValueError(msg % axis)
        angle = _handle_angle_input(angle, dtype, "RotateAxisAngle")
        angle = (angle / 180.0 * math.pi) if degrees else angle
        # We assume the points on which this transformation will be applied
        # are row vectors. The rotation matrix returned from _axis_angle_rotation
        # is for transforming column vectors. Therefore we transpose this matrix.
        # R will always be of shape (N, 3, 3)
        R = _axis_angle_rotation(axis, angle).transpose(1, 2)
        super().__init__(R=R, dtype=dtype)


def _handle_coord(c, dtype: ms.dtype) -> ms.Tensor:
    """
    Helper function for _handle_input.

    Args:
        c: Python scalar, torch scalar, or 1D torch tensor

    Returns:
        c_vec: 1D torch tensor
    """
    if not ops.is_tensor(c):
        c = ms.Tensor(c, dtype=dtype)
    if c.dim() == 0:
        c = c.view(1)
    return c.astype(dtype)


def _handle_input(
    x,
    y,
    z,
    dtype: ms.dtype,
    name: str,
    allow_singleton: bool = False,
) -> ms.Tensor:
    """
    Helper function to handle parsing logic for building transforms. The output
    is always a tensor of shape (N, 3), but there are several types of allowed
    input.

    Case I: Single Matrix
        In this case x is a tensor of shape (N, 3), and y and z are None. Here just
        return x.

    Case II: Vectors and Scalars
        In this case each of x, y, and z can be one of the following
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        In this case x, y and z are broadcast to tensors of shape (N, 1)
        and concatenated to a tensor of shape (N, 3)

    Case III: Singleton (only if allow_singleton=True)
        In this case y and z are None, and x can be one of the following:
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        Here x will be duplicated 3 times, and we return a tensor of shape (N, 3)

    Returns:
        xyz: Tensor of shape (N, 3)
    """
    # If x is actually a tensor of shape (N, 3) then just return it
    if ops.is_tensor(x) and x.dim() == 2:
        if x.shape[1] != 3:
            msg = "Expected tensor of shape (N, 3); got %r (in %s)"
            raise ValueError(msg % (x.shape, name))
        if y is not None or z is not None:
            msg = "Expected y and z to be None (in %s)" % name
            raise ValueError(msg)
        return x.astype(dtype)

    if allow_singleton and y is None and z is None:
        y = x
        z = x

    # Convert all to 1D tensors
    xyz = [_handle_coord(c, dtype) for c in [x, y, z]]

    # Broadcast and concatenate
    sizes = [c.shape[0] for c in xyz]
    N = max(sizes)
    for c in xyz:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r (in %s)" % (sizes, name)
            raise ValueError(msg)
    xyz = [c.expand(N) for c in xyz]
    xyz = ops.stack(xyz, axis=1)
    return xyz


def _handle_angle_input(
    x, dtype: ms.dtype, name: str
) -> ms.Tensor:
    """
    Helper function for building a rotation function using angles.
    The output is always of shape (N,).

    The input can be one of:
        - Torch tensor of shape (N,)
        - Python scalar
        - Torch scalar
    """
    if ops.is_tensor(x) and x.dim() > 1:
        msg = "Expected tensor of shape (N,); got %r (in %s)"
        raise ValueError(msg % (x.shape, name))
    else:
        return _handle_coord(x, dtype)


# def _broadcast_bmm(a, b) -> ms.Tensor:
#     """
#     Batch multiply two matrices and broadcast if necessary.

#     Args:
#         a: torch tensor of shape (P, K) or (M, P, K)
#         b: torch tensor of shape (N, K, K)

#     Returns:
#         a and b broadcast multiplied. The output batch dimension is max(N, M).

#     To broadcast transforms across a batch dimension if M != N then
#     expect that either M = 1 or N = 1. The tensor with batch dimension 1 is
#     expanded to have shape N or M.
#     """
#     if a.dim() == 2:
#         a = a[None]
#     if len(a) != len(b):
#         if not ((len(a) == 1) or (len(b) == 1)):
#             msg = "Expected batch dim for bmm to be equal or 1; got %r, %r"
#             raise ValueError(msg % (a.shape, b.shape))
#         if len(a) == 1:
#             a = a.expand(ms.Tensor((len(b), -1, -1), ms.int32))
#         if len(b) == 1:
#             b = b.expand(ms.Tensor((len(a), -1, -1), ms.int32))
#     return a.bmm(b)


def _check_valid_rotation_matrix(R, tol: float = 1e-7) -> None:
    """
    Determine if R is a valid rotation matrix by checking it satisfies the
    following conditions:

    ``RR^T = I and det(R) = 1``

    Args:
        R: an (N, 3, 3) matrix

    Returns:
        None

    Emits a warning if R is an invalid rotation matrix.
    """
    N = R.shape[0]
    eye = ops.eye(3, dtype=R.dtype)
    eye = eye.view(1, 3, 3).expand(N, -1, -1)
    orthogonal = np.allclose(R.bmm(R.transpose(0, 2, 1)).asnumpy(), eye.asnumpy(), atol=tol)
    det_R = _safe_det_3x3(R)
    no_distortion = np.allclose(det_R.asnumpy(), ops.ones_like(det_R.asnumpy()))
    if not (orthogonal and no_distortion):
        msg = "R is not a valid rotation matrix"
        warnings.warn(msg)
    return


def _safe_det_3x3(t: ms.Tensor):
    """
    Fast determinant calculation for a batch of 3x3 matrices.

    Note, result of this function might not be the same as `torch.det()`.
    The differences might be in the last significant digit.

    Args:
        t: Tensor of shape (N, 3, 3).

    Returns:
        Tensor of shape (N) with determinants.
    """

    det = (
        t[..., 0, 0] * (t[..., 1, 1] * t[..., 2, 2] -
                        t[..., 1, 2] * t[..., 2, 1])
        - t[..., 0, 1] * (t[..., 1, 0] * t[..., 2, 2] -
                          t[..., 2, 0] * t[..., 1, 2])
        + t[..., 0, 2] * (t[..., 1, 0] * t[..., 2, 1] -
                          t[..., 2, 0] * t[..., 1, 1])
    )

    return det


def _axis_angle_rotation(axis: str, angle: ms.Tensor) -> ms.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = ops.cos(angle)
    sin = ops.sin(angle)
    one = ops.ones_like(angle)
    zero = ops.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return ops.stack(R_flat, -1).reshape(angle.shape + (3, 3))
