from typing import Any, List, Optional, Tuple, Union, Dict
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from models.face3d.pytorch3d.utils import TensorProperties
from models.face3d.pytorch3d.transforms import Transform3d, Translate, Rotate


# Default values for rotation and translation matrices.
_R = ops.eye(3)[None]  # (1, 3, 3)
_T = ops.zeros((1, 3))  # (1, 3)


class CamerasBase(TensorProperties):
    """
    `CamerasBase` implements a base class for all cameras.

    For cameras, there are four different coordinate systems (or spaces)
    - World coordinate system: This is the system the object lives - the world.
    - Camera view coordinate system: This is the system that has its origin on
        the camera and the Z-axis perpendicular to the image plane.
        In PyTorch3D, we assume that +X points left, and +Y points up and
        +Z points out from the image plane.
        The transformation from world --> view happens after applying a rotation (R)
        and translation (T)
    - NDC coordinate system: This is the normalized coordinate system that confines
        points in a volume the rendered part of the object or scene, also known as
        view volume. For square images, given the PyTorch3D convention, (+1, +1, znear)
        is the top left near corner, and (-1, -1, zfar) is the bottom right far
        corner of the volume.
        The transformation from view --> NDC happens after applying the camera
        projection matrix (P) if defined in NDC space.
        For non square images, we scale the points such that smallest side
        has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    - Screen coordinate system: This is another representation of the view volume with
        the XY coordinates defined in image space instead of a normalized space.

    An illustration of the coordinate systems can be found in pytorch3d/docs/notes/cameras.md.

    CameraBase defines methods that are common to all camera models:
        - `get_camera_center` that returns the optical center of the camera in
            world coordinates
        - `get_world_to_view_transform` which returns a 3D transform from
            world coordinates to the camera view coordinates (R, T)
        - `get_full_projection_transform` which composes the projection
            transform (P) with the world-to-view transform (R, T)
        - `transform_points` which takes a set of input points in world coordinates and
            projects to the space the camera is defined in (NDC or screen)
        - `get_ndc_camera_transform` which defines the transform from screen/NDC to
            PyTorch3D's NDC space
        - `transform_points_ndc` which takes a set of points in world coordinates and
            projects them to PyTorch3D's NDC space
        - `transform_points_screen` which takes a set of points in world coordinates and
            projects them to screen space

    For each new camera, one should implement the `get_projection_transform`
    routine that returns the mapping from camera view coordinates to camera
    coordinates (NDC or screen).

    Another useful function that is specific to each camera model is
    `unproject_points` which sends points from camera coordinates (NDC or screen)
    back to camera view or world coordinates depending on the `world_coordinates`
    boolean argument of the function.
    """

    # Used in __getitem__ to index the relevant fields
    # When creating a new camera, this should be set in the __init__
    _FIELDS: Tuple[str, ...] = ()

    # Names of fields which are a constant property of the whole batch, rather
    # than themselves a batch of data.
    # When joining objects into a batch, they will have to agree.
    _SHARED_FIELDS: Tuple[str, ...] = ()

    def get_projection_transform(self, **kwargs):
        """
        Calculate the projective transformation matrix.

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.

        Return:
            a `Transform3d` object which represents a batch of projection
            matrices of shape (N, 3, 3)
        """
        raise NotImplementedError()

    def unproject_points(self, xy_depth: ms.Tensor, **kwargs):
        """
        Transform input points from camera coodinates (NDC or screen)
        to the world / camera coordinates.

        Each of the input points `xy_depth` of shape (..., 3) is
        a concatenation of the x, y location and its depth.

        For instance, for an input 2D tensor of shape `(num_points, 3)`
        `xy_depth` takes the following form:
            `xy_depth[i] = [x[i], y[i], depth[i]]`,
        for a each point at an index `i`.

        The following example demonstrates the relationship between
        `transform_points` and `unproject_points`:

        .. code-block:: python

            cameras = # camera object derived from CamerasBase
            xyz = # 3D points of shape (batch_size, num_points, 3)
            # transform xyz to the camera view coordinates
            xyz_cam = cameras.get_world_to_view_transform().transform_points(xyz)
            # extract the depth of each point as the 3rd coord of xyz_cam
            depth = xyz_cam[:, :, 2:]
            # project the points xyz to the camera
            xy = cameras.transform_points(xyz)[:, :, :2]
            # append depth to xy
            xy_depth = torch.cat((xy, depth), dim=2)
            # unproject to the world coordinates
            xyz_unproj_world = cameras.unproject_points(xy_depth, world_coordinates=True)
            print(torch.allclose(xyz, xyz_unproj_world)) # True
            # unproject to the camera coordinates
            xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)
            print(torch.allclose(xyz_cam, xyz_unproj)) # True

        Args:
            xy_depth: torch tensor of shape (..., 3).
            world_coordinates: If `True`, unprojects the points back to world
                coordinates using the camera extrinsics `R` and `T`.
                `False` ignores `R` and `T` and unprojects to
                the camera view coordinates.
            from_ndc: If `False` (default), assumes xy part of input is in
                NDC space if self.in_ndc(), otherwise in screen space. If
                `True`, assumes xy is in NDC space even if the camera
                is defined in screen space.

        Returns
            new_points: unprojected points with the same shape as `xy_depth`.
        """
        raise NotImplementedError()

    def get_camera_center(self, **kwargs) -> ms.Tensor:
        """
        Return the 3D location of the camera optical center
        in the world coordinates.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R or T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            C: a batch of 3D locations of shape (N, 3) denoting
            the locations of the center of each camera in the batch.
        """
        w2v_trans = self.get_world_to_view_transform(**kwargs)
        P = w2v_trans.inverse().get_matrix()
        # the camera center is the translation component (the first 3 elements
        # of the last row) of the inverted world-to-view
        # transform (4x4 RT matrix)
        C = P[:, 3, :3]
        return C

    def get_world_to_view_transform(self, **kwargs):
        """
        Return the world-to-view transform.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            A Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        R: ms.Tensor = kwargs.get("R", self.R)
        T: ms.Tensor = kwargs.get("T", self.T)
        self.R = R
        self.T = T
        world_to_view_transform = get_world_to_view_transform(R=R, T=T)
        return world_to_view_transform

    def get_full_projection_transform(self, **kwargs):
        """
        Return the full world-to-camera transform composing the
        world-to-view and view-to-camera transforms.
        If camera is defined in NDC space, the projected points are in NDC space.
        If camera is defined in screen space, the projected points are in screen space.

        Args:
            **kwargs: parameters for the projection transforms can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            a Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        self.R: ms.Tensor = kwargs.get("R", self.R)
        self.T: ms.Tensor = kwargs.get("T", self.T)
        world_to_view_transform = self.get_world_to_view_transform(
            R=self.R, T=self.T)
        view_to_proj_transform = self.get_projection_transform(**kwargs)
        return world_to_view_transform.compose(view_to_proj_transform)

    def transform_points(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> ms.Tensor:
        """
        Transform input points from world to camera space.
        If camera is defined in NDC space, the projected points are in NDC space.
        If camera is defined in screen space, the projected points are in screen space.

        For `CamerasBase.transform_points`, setting `eps > 0`
        stabilizes gradients since it leads to avoiding division
        by excessively low numbers for points close to the camera plane.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_proj_transform = self.get_full_projection_transform(**kwargs)
        return world_to_proj_transform.transform_points(points, eps=eps)

    def get_ndc_camera_transform(self, **kwargs):
        """
        Returns the transform from camera projection space (screen or NDC) to NDC space.
        For cameras that can be specified in screen space, this transform
        allows points to be converted from screen to NDC space.
        The default transform scales the points from [0, W]x[0, H]
        to [-1, 1]x[-u, u] or [-u, u]x[-1, 1] where u > 1 is the aspect ratio of the image.
        This function should be modified per camera definitions if need be,
        e.g. for Perspective/Orthographic cameras we provide a custom implementation.
        This transform assumes PyTorch3D coordinate system conventions for
        both the NDC space and the input points.

        This transform interfaces with the PyTorch3D renderer which assumes
        input points to the renderer to be in NDC space.
        """
        if self.in_ndc():
            return Transform3d(dtype=ms.float32)
        else:
            # For custom cameras which can be defined in screen space,
            # users might might have to implement the screen to NDC transform based
            # on the definition of the camera parameters.
            # See PerspectiveCameras/OrthographicCameras for an example.
            # We don't flip xy because we assume that world points are in
            # PyTorch3D coordinates, and thus conversion from screen to ndc
            # is a mere scaling from image to [-1, 1] scale.
            image_size = kwargs.get("image_size", self.get_image_size())
            return get_screen_to_ndc_transform(
                self, with_xyflip=False, image_size=image_size
            )

    def transform_points_ndc(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> ms.Tensor:
        """
        Transforms points from PyTorch3D world/camera space to NDC space.
        Input points follow the PyTorch3D coordinate system conventions: +X left, +Y up.
        Output points are in NDC space: +X left, +Y up, origin at image center.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_ndc_transform = self.get_full_projection_transform(**kwargs)
        if not self.in_ndc():
            to_ndc_transform = self.get_ndc_camera_transform(**kwargs)
            world_to_ndc_transform = world_to_ndc_transform.compose(
                to_ndc_transform)

        return world_to_ndc_transform.transform_points(points, eps=eps)

    def transform_points_screen(
        self, points, eps: Optional[float] = None, with_xyflip: bool = True, **kwargs
    ) -> ms.Tensor:
        """
        Transforms points from PyTorch3D world/camera space to screen space.
        Input points follow the PyTorch3D coordinate system conventions: +X left, +Y up.
        Output points are in screen space: +X right, +Y down, origin at top left corner.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the ndc space. Please see
                `transforms.Transform3d.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessively low numbers for points close to the
                camera plane.
            with_xyflip: If True, flip x and y directions. In world/camera/ndc coords,
                +x points to the left and +y up. If with_xyflip is true, in screen
                coords +x points right, and +y down, following the usual RGB image
                convention. Warning: do not set to False unless you know what you're
                doing!

        Returns
            new_points: transformed points with the same shape as the input.
        """
        points_ndc = self.transform_points_ndc(points, eps=eps, **kwargs)
        image_size = kwargs.get("image_size", self.get_image_size())
        return get_ndc_to_screen_transform(
            self, with_xyflip=with_xyflip, image_size=image_size
        ).transform_points(points_ndc, eps=eps)

    def clone(self):
        """
        Returns a copy of `self`.
        """
        cam_type = type(self)
        other = cam_type(device=self.device)
        return super().clone(other)

    def is_perspective(self):
        raise NotImplementedError()

    def in_ndc(self):
        """
        Specifies whether the camera is defined in NDC space
        or in screen (image) space
        """
        raise NotImplementedError()

    def get_znear(self):
        return getattr(self, "znear", None)

    def get_image_size(self):
        """
        Returns the image size, if provided, expected in the form of (height, width)
        The image size is used for conversion of projected points to screen coordinates.
        """
        return getattr(self, "image_size", None)

    def __getitem__(
        self, index: Union[int, List[int], ms.Tensor]
    ) -> "CamerasBase":
        """
        Override for the __getitem__ method in TensorProperties which needs to be
        refactored.

        Args:
            index: an integer index, list/tensor of integer indices, or tensor of boolean
                indicators used to filter all the fields in the cameras given by self._FIELDS.
        Returns:
            an instance of the current cameras class with only the values at the selected index.
        """

        kwargs = {}

        tensor_types = {
            # pyre-fixme[16]: Module `cuda` has no attribute `BoolTensor`.
            "bool": (torch.BoolTensor, torch.cuda.BoolTensor),
            # pyre-fixme[16]: Module `cuda` has no attribute `LongTensor`.
            "long": (torch.LongTensor, torch.cuda.LongTensor),
        }
        if not isinstance(
            index, (int, list, *tensor_types["bool"], *tensor_types["long"])
        ) or (
            isinstance(index, list)
            and not all(isinstance(i, int) and not isinstance(i, bool) for i in index)
        ):
            msg = (
                "Invalid index type, expected int, List[int] or Bool/LongTensor; got %r"
            )
            raise ValueError(msg % type(index))

        if isinstance(index, int):
            index = [index]

        if isinstance(index, tensor_types["bool"]):
            # pyre-fixme[16]: Item `List` of `Union[List[int], BoolTensor,
            #  LongTensor]` has no attribute `ndim`.
            # pyre-fixme[16]: Item `List` of `Union[List[int], BoolTensor,
            #  LongTensor]` has no attribute `shape`.
            if index.ndim != 1 or index.shape[0] != len(self):
                raise ValueError(
                    # pyre-fixme[16]: Item `List` of `Union[List[int], BoolTensor,
                    #  LongTensor]` has no attribute `shape`.
                    f"Boolean index of shape {index.shape} does not match cameras"
                )
        elif max(index) >= len(self):
            raise IndexError(
                f"Index {max(index)} is out of bounds for select cameras")

        for field in self._FIELDS:
            val = getattr(self, field, None)
            if val is None:
                continue

            # e.g. "in_ndc" is set as attribute "_in_ndc" on the class
            # but provided as "in_ndc" on initialization
            if field.startswith("_"):
                field = field[1:]

            if isinstance(val, (str, bool)):
                kwargs[field] = val
            elif isinstance(val, torch.Tensor):
                # In the init, all inputs will be converted to
                # tensors before setting as attributes
                kwargs[field] = val[index]
            else:
                raise ValueError(
                    f"Field {field} type is not supported for indexing")

        kwargs["device"] = self.device
        return self.__class__(**kwargs)


class FoVPerspectiveCameras(CamerasBase):
    """
    A class which stores a batch of parameters to generate a batch of
    projection matrices by specifying the field of view.
    The definitions of the parameters follow the OpenGL perspective camera.

    The extrinsics of the camera (R and T matrices) can also be set in the
    initializer or passed in to `get_full_projection_transform` to get
    the full transformation from world -> ndc.

    The `transform_points` method calculates the full world -> ndc transform
    and then applies it to the input points.

    The transforms can also be returned separately as Transform3d objects.

    * Setting the Aspect Ratio for Non Square Images *

    If the desired output image size is non square (i.e. a tuple of (H, W) where H != W)
    the aspect ratio needs special consideration: There are two aspect ratios
    to be aware of:
        - the aspect ratio of each pixel
        - the aspect ratio of the output image
    The `aspect_ratio` setting in the FoVPerspectiveCameras sets the
    pixel aspect ratio. When using this camera with the differentiable rasterizer
    be aware that in the rasterizer we assume square pixels, but allow
    variable image aspect ratio (i.e rectangle images).

    In most cases you will want to set the camera `aspect_ratio=1.0`
    (i.e. square pixels) and only vary the output image dimensions in pixels
    for rasterization.
    """

    # For __getitem__
    _FIELDS = (
        "K",
        "znear",
        "zfar",
        "aspect_ratio",
        "fov",
        "R",
        "T",
        "degrees",
    )

    _SHARED_FIELDS = ("degrees",)

    def __init__(
        self,
        znear = 1.0,
        zfar = 100.0,
        aspect_ratio = 1.0,
        fov = 60.0,
        degrees: bool = True,
        R: ms.Tensor = _R,
        T: ms.Tensor = _T,
        K: Optional[ms.Tensor] = None,
    ) -> None:
        """

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            fov: field of view angle of the camera.
            degrees: bool, set to True if fov is specified in degrees.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need znear, zfar, fov, aspect_ratio, degrees
            device: Device (as str or torch.device)
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(
            znear=znear,
            zfar=zfar,
            aspect_ratio=aspect_ratio,
            fov=fov,
            R=R,
            T=T,
            K=K,
        )

        # No need to convert to tensor or broadcast.
        self.degrees = degrees

    def compute_projection_matrix(
        self, znear, zfar, fov, aspect_ratio, degrees: bool
    ) -> ms.Tensor:
        """
        Compute the calibration matrix K of shape (N, 4, 4)

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            fov: field of view angle of the camera.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            degrees: bool, set to True if fov is specified in degrees.

        Returns:
            torch.FloatTensor of the calibration matrix with shape (N, 4, 4)
        """
        K = ops.zeros((self._N, 4, 4),
                        dtype=ms.float32)
        ones = ops.ones((self._N), dtype=ms.float32)
        if degrees:
            fov = (np.pi / 180) * fov

        if not ops.is_tensor(fov):
            fov = ms.Tensor(fov)
        tanHalfFov = ops.tan((fov / 2))
        max_y = tanHalfFov * znear
        min_y = -max_y
        max_x = max_y * aspect_ratio
        min_x = -max_x

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space positive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.0

        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        K[:, 0, 0] = 2.0 * znear / (max_x - min_x)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        K[:, 1, 1] = 2.0 * znear / (max_y - min_y)
        K[:, 0, 2] = (max_x + min_x) / (max_x - min_x)
        K[:, 1, 2] = (max_y + min_y) / (max_y - min_y)
        K[:, 3, 2] = z_sign * ones

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane.
        K[:, 2, 2] = z_sign * zfar / (zfar - znear)
        K[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        return K

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the perspective projection matrix with a symmetric
        viewing frustrum. Use column major order.
        The viewing frustrum will be projected into ndc, s.t.
        (max_x, max_y) -> (+1, +1)
        (min_x, min_y) -> (-1, -1)

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.

        Return:
            a Transform3d object which represents a batch of projection
            matrices of shape (N, 4, 4)

        .. code-block:: python

            h1 = (max_y + min_y)/(max_y - min_y)
            w1 = (max_x + min_x)/(max_x - min_x)
            tanhalffov = tan((fov/2))
            s1 = 1/tanhalffov
            s2 = 1/(tanhalffov * (aspect_ratio))

            # To map z to the range [0, 1] use:
            f1 =  far / (far - near)
            f2 = -(far * near) / (far - near)

            # Projection matrix
            K = [
                    [s1,   0,   w1,   0],
                    [0,   s2,   h1,   0],
                    [0,    0,   f1,  f2],
                    [0,    0,    1,   0],
            ]
        """
        K = kwargs.get("K", self.K)
        if K is not None:
            if K.shape != (self._N, 4, 4):
                msg = "Expected K to have shape of (%r, 4, 4)"
                raise ValueError(msg % (self._N))
        else:
            K = self.compute_projection_matrix(
                kwargs.get("znear", self.znear),
                kwargs.get("zfar", self.zfar),
                kwargs.get("fov", self.fov),
                kwargs.get("aspect_ratio", self.aspect_ratio),
                kwargs.get("degrees", self.degrees),
            )

        # Transpose the projection matrix as PyTorch3D transforms use row vectors.
        transform = Transform3d(
            matrix=K.transpose(0, 2, 1)
        )
        return transform

    def unproject_points(
        self,
        xy_depth: ms.Tensor,
        world_coordinates: bool = True,
        scaled_depth_input: bool = False,
        **kwargs,
    ) -> ms.Tensor:
        """>!
        FoV cameras further allow for passing depth in world units
        (`scaled_depth_input=False`) or in the [0, 1]-normalized units
        (`scaled_depth_input=True`)

        Args:
            scaled_depth_input: If `True`, assumes the input depth is in
                the [0, 1]-normalized units. If `False` the input depth is in
                the world units.
        """

        # obtain the relevant transformation to ndc
        if world_coordinates:
            to_ndc_transform = self.get_full_projection_transform()
        else:
            to_ndc_transform = self.get_projection_transform()

        if scaled_depth_input:
            # the input is scaled depth, so we don't have to do anything
            xy_sdepth = xy_depth
        else:
            # parse out important values from the projection matrix
            K_matrix = self.get_projection_transform(
                **kwargs.copy()).get_matrix()
            # parse out f1, f2 from K_matrix
            unsqueeze_shape = [1] * xy_depth.dim()
            unsqueeze_shape[0] = K_matrix.shape[0]
            f1 = K_matrix[:, 2, 2].reshape(unsqueeze_shape)
            f2 = K_matrix[:, 3, 2].reshape(unsqueeze_shape)
            # get the scaled depth
            sdepth = (f1 * xy_depth[..., 2:3] + f2) / xy_depth[..., 2:3]
            # concatenate xy + scaled depth
            xy_sdepth = ops.cat((xy_depth[..., 0:2], sdepth), axis=-1)

        # unproject with inverse of the projection
        unprojection_transform = to_ndc_transform.inverse()
        return unprojection_transform.transform_points(xy_sdepth)

    def is_perspective(self):
        return True

    def in_ndc(self):
        return True


def try_get_projection_transform(
    cameras: CamerasBase, cameras_kwargs: Dict[str, Any]
):
    """
    Try block to get projection transform from cameras and cameras_kwargs.

    Args:
        cameras: cameras instance, can be linear cameras or nonliear cameras
        cameras_kwargs: camera parameters to be passed to cameras

    Returns:
        If the camera implemented projection_transform, return the
        projection transform; Otherwise, return None
    """

    transform = None
    transform = cameras.get_projection_transform(**cameras_kwargs)

    return transform


def get_world_to_view_transform(
    R: ms.Tensor = _R, T: ms.Tensor = _T
) -> Transform3d:
    """
    This function returns a Transform3d representing the transformation
    matrix to go from world space to view space by applying a rotation and
    a translation.

    PyTorch3D uses the same convention as Hartley & Zisserman.
    I.e., for camera extrinsic parameters R (rotation) and T (translation),
    we map a 3D point `X_world` in world coordinates to
    a point `X_cam` in camera coordinates with:
    `X_cam = X_world R + T`

    Args:
        R: (N, 3, 3) matrix representing the rotation.
        T: (N, 3) matrix representing the translation.

    Returns:
        a Transform3d object which represents the composed RT transformation.

    """
    # TODO: also support the case where RT is specified as one matrix
    # of shape (N, 4, 4).
    if T.shape[0] != R.shape[0]:
        msg = "Expected R, T to have the same batch dimension; got %r, %r"
        raise ValueError(msg % (R.shape[0], T.shape[0]))
    if T.dim() != 2 or T.shape[1:] != (3,):
        msg = "Expected T to have shape (N, 3); got %r"
        raise ValueError(msg % repr(T.shape))
    if R.dim() != 3 or R.shape[1:] != (3, 3):
        msg = "Expected R to have shape (N, 3, 3); got %r"
        raise ValueError(msg % repr(R.shape))

    # Create a Transform3d object
    T_ = Translate(T)
    R_ = Rotate(R)
    return R_.compose(T_)
