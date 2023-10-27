"""This script is the differentiable renderer for Deep3DFaceRecon_pytorch
    Attention, antialiasing step is missing in current version.
"""
import numpy as np
import mindspore as ms
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
from mindspore import nn, ops
from mindspore import jit_class
from models.face3d.pytorch3d.cameras import FoVPerspectiveCameras, try_get_projection_transform
from models.face3d.pytorch3d.interp_face_attrs import interpolate_face_attributes
from models.face3d.pytorch3d.meshes import Meshes
from models.face3d.pytorch3d.rasterize_meshes_python import rasterize_meshes_python
from models.face3d.pytorch3d.clip import ClipFrustum, clip_faces, convert_clipped_rasterization_to_original_faces
from models.face3d.pytorch3d.utils import parse_image_size


# TODO make the epsilon user configurable
kEpsilon = 1e-8

# Maximum number of faces per bins for
# coarse-to-fine rasterization
kMaxFacesPerBin = 22


@dataclass(frozen=True)
class Fragments:
    """
    A dataclass representing the outputs of a rasterizer. Can be detached from the
    computational graph in order to stop the gradients from flowing through the
    rasterizer.

    Members:
        pix_to_face:
            LongTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the indices of the nearest faces at each pixel, sorted in ascending
            z-order. Concretely ``pix_to_face[n, y, x, k] = f`` means that
            ``faces_verts[f]`` is the kth closest face (in the z-direction) to pixel
            (y, x). Pixels that are hit by fewer than faces_per_pixel are padded with
            -1.

        zbuf:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the NDC z-coordinates of the nearest faces at each pixel, sorted in
            ascending z-order. Concretely, if ``pix_to_face[n, y, x, k] = f`` then
            ``zbuf[n, y, x, k] = face_verts[f, 2]``. Pixels hit by fewer than
            faces_per_pixel are padded with -1.

        bary_coords:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel, 3)
            giving the barycentric coordinates in NDC units of the nearest faces at
            each pixel, sorted in ascending z-order. Concretely, if ``pix_to_face[n,
            y, x, k] = f`` then ``[w0, w1, w2] = barycentric[n, y, x, k]`` gives the
            barycentric coords for pixel (y, x) relative to the face defined by
            ``face_verts[f]``. Pixels hit by fewer than faces_per_pixel are padded
            with -1.

        dists:
            FloatTensor of shape (N, image_size, image_size, faces_per_pixel) giving
            the signed Euclidean distance (in NDC units) in the x/y plane of each
            point closest to the pixel. Concretely if ``pix_to_face[n, y, x, k] = f``
            then ``pix_dists[n, y, x, k]`` is the squared distance between the pixel
            (y, x) and the face given by vertices ``face_verts[f]``. Pixels hit with
            fewer than ``faces_per_pixel`` are padded with -1.
    """

    pix_to_face: ms.Tensor
    zbuf: ms.Tensor
    bary_coords: ms.Tensor
    dists: Optional[ms.Tensor]

    def detach(self) -> "Fragments":
        return Fragments(
            pix_to_face=self.pix_to_face,
            zbuf=self.zbuf,
            bary_coords=self.bary_coords,
            dists=self.dists,
        )


@dataclass
class RasterizationSettings:
    """
    Class to store the mesh rasterization params with defaults

    Members:
        image_size: Either common height and width or (height, width), in pixels.
        blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
        faces_per_pixel: (int) Number of faces to keep track of per pixel.
            We return the nearest faces_per_pixel faces along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts
            to set it heuristically based on the shape of the input. This should
            not affect the output, but can affect the speed of the forward pass.
        max_faces_opengl: Max number of faces in any mesh we will rasterize. Used only by
            MeshRasterizerOpenGL to pre-allocate OpenGL memory.
        max_faces_per_bin: Only applicable when using coarse-to-fine
            rasterization (bin_size != 0); this is the maximum number of faces
            allowed within each bin. This should not affect the output values,
            but can affect the memory usage in the forward pass.
            Setting max_faces_per_bin=None attempts to set with a heuristic.
        perspective_correct: Whether to apply perspective correction when
            computing barycentric coordinates for pixels.
            None (default) means make correction if the camera uses perspective.
        clip_barycentric_coords: Whether, after any perspective correction
            is applied but before the depth is calculated (e.g. for
            z clipping), to "correct" a location outside the face (i.e. with
            a negative barycentric coordinate) to a position on the edge of the
            face. None (default) means clip if blur_radius > 0, which is a condition
            under which such outside-face-points are likely.
        cull_backfaces: Whether to only rasterize mesh faces which are
            visible to the camera.  This assumes that vertices of
            front-facing triangles are ordered in an anti-clockwise
            fashion, and triangles that face away from the camera are
            in a clockwise order relative to the current view
            direction. NOTE: This will only work if the mesh faces are
            consistently defined with counter-clockwise ordering when
            viewed from the outside.
        z_clip_value: if not None, then triangles will be clipped (and possibly
            subdivided into smaller triangles) such that z >= z_clip_value.
            This avoids camera projections that go to infinity as z->0.
            Default is None as clipping affects rasterization speed and
            should only be turned on if explicitly needed.
            See clip.py for all the extra computation that is required.
        cull_to_frustum: Whether to cull triangles outside the view frustum.
            Culling involves removing all faces which fall outside view frustum.
            Default is False for performance as often not needed.
    """

    image_size: Union[int, Tuple[int, int]] = 256
    blur_radius: float = 0.0
    faces_per_pixel: int = 1
    bin_size: Optional[int] = None
    max_faces_opengl: int = 10_000_000
    max_faces_per_bin: Optional[int] = None
    perspective_correct: Optional[bool] = None
    clip_barycentric_coords: Optional[bool] = None
    cull_backfaces: bool = False
    z_clip_value: Optional[float] = None
    cull_to_frustum: bool = False


@jit_class
class MeshRasterizer():
    """
    This class implements methods for rasterizing a batch of heterogeneous
    Meshes.
    """

    def __init__(self, cameras=None, raster_settings=None) -> None:
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-ndc transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        if raster_settings is None:
            raster_settings = RasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def transform(self, meshes_world, **kwargs) -> ms.Tensor:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                vertex coordinates in world space.

        Returns:
            meshes_proj: a Meshes object with the vertex positions projected
            in NDC space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of MeshRasterizer"
            raise ValueError(msg)

        n_cameras = len(cameras)
        if n_cameras != 1 and n_cameras != len(meshes_world):
            msg = "Wrong number (%r) of cameras for %r meshes"
            raise ValueError(msg % (n_cameras, len(meshes_world)))

        verts_world = meshes_world.verts_padded()

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        eps = kwargs.get("eps", None)
        verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            verts_world, eps=eps
        )
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = try_get_projection_transform(cameras, kwargs)
        if projection_transform is not None:
            projection_transform = projection_transform.compose(
                to_ndc_transform)
            verts_ndc = projection_transform.transform_points(
                verts_view, eps=eps)
        else:
            # Call transform_points instead of explicitly composing transforms to handle
            # the case, where camera class does not have a projection matrix form.
            verts_proj = cameras.transform_points(verts_world, eps=eps)
            verts_ndc = to_ndc_transform.transform_points(verts_proj, eps=eps)

        verts_ndc[..., 2] = verts_view[..., 2]
        meshes_ndc = meshes_world.update_padded(new_verts_padded=verts_ndc)
        return meshes_ndc

    def forward_rasterizer(self, meshes_world, **kwargs) -> Fragments:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        print("start transform ...")
        meshes_proj = self.transform(meshes_world, **kwargs)
        print("finished transform ...")
        raster_settings = kwargs.get("raster_settings", self.raster_settings)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = raster_settings.blur_radius > 0.0

        # If not specified, infer perspective_correct and z_clip_value from the camera
        cameras = kwargs.get("cameras", self.cameras)
        if raster_settings.perspective_correct is not None:
            perspective_correct = raster_settings.perspective_correct
        else:
            perspective_correct = cameras.is_perspective()
        if raster_settings.z_clip_value is not None:
            z_clip = raster_settings.z_clip_value
        else:
            znear = cameras.get_znear()
            if isinstance(znear, ms.Tensor):
                znear = znear.min()
            z_clip = None if not perspective_correct or znear is None else znear / 2

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.

        print("start rasterizidsng meshes ...")

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes_python(
            meshes_proj,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            # bin_size=raster_settings.bin_size,
            # max_faces_per_bin=raster_settings.max_faces_per_bin,
            # clip_barycentric_coords=clip_barycentric_coords,
            perspective_correct=perspective_correct,
            cull_backfaces=raster_settings.cull_backfaces,
            z_clip_value=z_clip,
            cull_to_frustum=raster_settings.cull_to_frustum,
        )

        return Fragments(
            pix_to_face=pix_to_face,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=dists,
        )


@jit_class
class MeshRenderer():
    def __init__(self,
                 rasterize_fov,
                 znear=0.1,
                 zfar=10,
                 rasterize_size=224):

        self.rasterize_size = rasterize_size
        self.fov = rasterize_fov
        self.znear = znear
        self.zfar = zfar

        self.rasterizer = MeshRasterizer()

    def forward_rendering(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- ms.Tensor, size (B, 1, H, W)
            depth              -- ms.Tensor, size (B, 1, H, W)
            features(optional) -- ms.Tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- ms.Tensor, size (B, N, 3)
            tri             -- ms.Tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- ms.Tensor, size (B, N ,C), features
        """
        rsize = int(self.rasterize_size)
        # ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = ops.cat(
                [vertex, ops.ones([vertex.shape[0], vertex.shape[1], 1])], axis=-1)
            vertex[..., 0] = -vertex[..., 0]

        # vertex_ndc = vertex @ ndc_proj.t()

        tri = tri.astype(ms.int32)

        # rasterize
        cameras = FoVPerspectiveCameras(
            fov=self.fov,
            znear=self.znear,
            zfar=self.zfar,
        )

        raster_settings = RasterizationSettings(
            image_size=rsize
        )

        verts = ms.Tensor(vertex[..., :3].asnumpy())
        faces = ms.Tensor(tri.unsqueeze(0).repeat(vertex.shape[0], axis=0).asnumpy())

        mesh = Meshes(verts, faces)

        # mesh = Meshes(ops.zeros((84, 35709, 3)), ops.zeros((84, 70789, 3)))

        fragments = self.rasterizer.forward_rasterizer(
            mesh, cameras=cameras, raster_settings=raster_settings)
        rast_out = fragments.pix_to_face.squeeze(-1)
        depth = fragments.zbuf

        # render depth
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out > 0).float().unsqueeze(1)
        depth = mask * depth

        image = None
        if feat is not None:
            attributes = feat.reshape(-1, 3)[mesh.faces_packed()]
            image = interpolate_face_attributes(fragments.pix_to_face,
                                                fragments.bary_coords,
                                                attributes)
            # print(image.shape)
            image = image.squeeze(-2).permute(0, 3, 1, 2)
            image = mask * image

        return mask, depth, image


class _RasterizeFaceVerts(nn.Cell):
    """
    Torch autograd wrapper for forward and backward pass of rasterize_meshes
    implemented in C++/CUDA.

    Args:
        face_verts: Tensor of shape (F, 3, 3) giving (packed) vertex positions
            for faces in all the meshes in the batch. Concretely,
            face_verts[f, i] = [x, y, z] gives the coordinates for the
            ith vertex of the fth face. These vertices are expected to
            be in NDC coordinates in the range [-1, 1].
        mesh_to_face_first_idx: LongTensor of shape (N) giving the index in
            faces_verts of the first face in each mesh in
            the batch.
        num_faces_per_mesh: LongTensor of shape (N) giving the number of faces
            for each mesh in the batch.
        image_size, blur_radius, faces_per_pixel: same as rasterize_meshes.
        perspective_correct: same as rasterize_meshes.
        cull_backfaces: same as rasterize_meshes.

    Returns:
        same as rasterize_meshes function.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def construct(
        ctx,
        face_verts: ms.Tensor,
        mesh_to_face_first_idx: ms.Tensor,
        num_faces_per_mesh: ms.Tensor,
        clipped_faces_neighbor_idx: ms.Tensor,
        image_size: Union[List[int], Tuple[int, int]] = (256, 256),
        blur_radius: float = 0.01,
        faces_per_pixel: int = 0,
        bin_size: int = 0,
        max_faces_per_bin: int = 0,
        perspective_correct: bool = False,
        clip_barycentric_coords: bool = False,
        cull_backfaces: bool = False,
        z_clip_value: Optional[float] = None,
        cull_to_frustum: bool = True,
    ):
        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        pix_to_face, zbuf, barycentric_coords, dists = _C.rasterize_meshes(  # TODO!!!
            face_verts,
            mesh_to_face_first_idx,
            num_faces_per_mesh,
            clipped_faces_neighbor_idx,
            image_size,
            blur_radius,
            faces_per_pixel,
            bin_size,
            max_faces_per_bin,
            perspective_correct,
            clip_barycentric_coords,
            cull_backfaces,
        )

        ctx.save_for_backward(face_verts, pix_to_face)
        ctx.mark_non_differentiable(pix_to_face)
        ctx.perspective_correct = perspective_correct
        ctx.clip_barycentric_coords = clip_barycentric_coords
        return pix_to_face, zbuf, barycentric_coords, dists


def rasterize_meshes(
    meshes,
    image_size: Union[int, List[int], Tuple[int, int]] = 256,
    blur_radius: float = 0.0,
    faces_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_faces_per_bin: Optional[int] = None,
    perspective_correct: bool = False,
    clip_barycentric_coords: bool = False,
    cull_backfaces: bool = False,
    z_clip_value: Optional[float] = None,
    cull_to_frustum: bool = False,
):
    """
    Rasterize a batch of meshes given the shape of the desired output image.
    Each mesh is rasterized onto a separate image of shape
    (H, W) if `image_size` is a tuple or (image_size, image_size) if it
    is an int.

    If the desired image size is non square (i.e. a tuple of (H, W) where H != W)
    the aspect ratio needs special consideration. There are two aspect ratios
    to be aware of:
        - the aspect ratio of each pixel
        - the aspect ratio of the output image
    The camera can be used to set the pixel aspect ratio. In the rasterizer,
    we assume square pixels, but variable image aspect ratio (i.e rectangle images).

    In most cases you will want to set the camera aspect ratio to
    1.0 (i.e. square pixels) and only vary the
    `image_size` (i.e. the output image dimensions in pixels).

    Args:
        meshes: A Meshes object representing a batch of meshes, batch size N.
        image_size: Size in pixels of the output image to be rasterized.
            Can optionally be a tuple of (H, W) in the case of non square images.
        blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
        faces_per_pixel (Optional): Number of faces to save per pixel, returning
            the nearest faces_per_pixel points along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts to
            set it heuristically based on the shape of the input. This should not
            affect the output, but can affect the speed of the forward pass.
        max_faces_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maximum number of faces allowed within each
            bin. This should not affect the output values, but can affect
            the memory usage in the forward pass.
        perspective_correct: Bool, Whether to apply perspective correction when computing
            barycentric coordinates for pixels. This should be set to True if a perspective
            camera is used.
        clip_barycentric_coords: Whether, after any perspective correction is applied
            but before the depth is calculated (e.g. for z clipping),
            to "correct" a location outside the face (i.e. with a negative
            barycentric coordinate) to a position on the edge of the face.
        cull_backfaces: Bool, Whether to only rasterize mesh faces which are
            visible to the camera.  This assumes that vertices of
            front-facing triangles are ordered in an anti-clockwise
            fashion, and triangles that face away from the camera are
            in a clockwise order relative to the current view
            direction. NOTE: This will only work if the mesh faces are
            consistently defined with counter-clockwise ordering when
            viewed from the outside.
        z_clip_value: if not None, then triangles will be clipped (and possibly
            subdivided into smaller triangles) such that z >= z_clip_value.
            This avoids camera projections that go to infinity as z->0.
            Default is None as clipping affects rasterization speed and
            should only be turned on if explicitly needed.
            See clip.py for all the extra computation that is required.
        cull_to_frustum: if True, triangles outside the view frustum will be culled.
            Culling involves removing all faces which fall outside view frustum.
            Default is False so that it is turned on only when needed.

    Returns:
        4-element tuple containing

        - **pix_to_face**: LongTensor of shape
          (N, image_size, image_size, faces_per_pixel)
          giving the indices of the nearest faces at each pixel,
          sorted in ascending z-order.
          Concretely ``pix_to_face[n, y, x, k] = f`` means that
          ``faces_verts[f]`` is the kth closest face (in the z-direction)
          to pixel (y, x). Pixels that are hit by fewer than
          faces_per_pixel are padded with -1.
        - **zbuf**: FloatTensor of shape (N, image_size, image_size, faces_per_pixel)
          giving the NDC z-coordinates of the nearest faces at each pixel,
          sorted in ascending z-order.
          Concretely, if ``pix_to_face[n, y, x, k] = f`` then
          ``zbuf[n, y, x, k] = face_verts[f, 2]``. Pixels hit by fewer than
          faces_per_pixel are padded with -1.
        - **barycentric**: FloatTensor of shape
          (N, image_size, image_size, faces_per_pixel, 3)
          giving the barycentric coordinates in NDC units of the
          nearest faces at each pixel, sorted in ascending z-order.
          Concretely, if ``pix_to_face[n, y, x, k] = f`` then
          ``[w0, w1, w2] = barycentric[n, y, x, k]`` gives
          the barycentric coords for pixel (y, x) relative to the face
          defined by ``face_verts[f]``. Pixels hit by fewer than
          faces_per_pixel are padded with -1.
        - **pix_dists**: FloatTensor of shape
          (N, image_size, image_size, faces_per_pixel)
          giving the signed Euclidean distance (in NDC units) in the
          x/y plane of each point closest to the pixel. Concretely if
          ``pix_to_face[n, y, x, k] = f`` then ``pix_dists[n, y, x, k]`` is the
          squared distance between the pixel (y, x) and the face given
          by vertices ``face_verts[f]``. Pixels hit with fewer than
          ``faces_per_pixel`` are padded with -1.

        In the case that image_size is a tuple of (H, W) then the outputs
        will be of shape `(N, H, W, ...)`.
    """
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    face_verts = verts_packed[faces_packed]
    mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()
    num_faces_per_mesh = meshes.num_faces_per_mesh()

    # In the case that H != W use the max image size to set the bin_size
    # to accommodate the num bins constraint in the coarse rasterizer.
    # If the ratio of H:W is large this might cause issues as the smaller
    # dimension will have fewer bins.
    # TODO: consider a better way of setting the bin size.
    im_size = parse_image_size(image_size)
    max_image_size = max(*im_size)

    clipped_faces_neighbor_idx = None

    if z_clip_value is not None or cull_to_frustum:
        # Cull faces outside the view frustum, and clip faces that are partially
        # behind the camera into the portion of the triangle in front of the
        # camera.  This may change the number of faces
        frustum = ClipFrustum(
            left=-1,
            right=1,
            top=-1,
            bottom=1,
            perspective_correct=perspective_correct,
            z_clip_value=z_clip_value,
            cull=cull_to_frustum,
        )
        clipped_faces = clip_faces(
            face_verts, mesh_to_face_first_idx, num_faces_per_mesh, frustum=frustum
        )
        face_verts = clipped_faces.face_verts
        mesh_to_face_first_idx = clipped_faces.mesh_to_face_first_idx
        num_faces_per_mesh = clipped_faces.num_faces_per_mesh

        # For case 4 clipped triangles (where a big triangle is split in two smaller triangles),
        # need the index of the neighboring clipped triangle as only one can be in
        # in the top K closest faces in the rasterization step.
        clipped_faces_neighbor_idx = clipped_faces.clipped_faces_neighbor_idx

    if clipped_faces_neighbor_idx is None:
        # Set to the default which is all -1s.
        clipped_faces_neighbor_idx = ops.full(
            size=(face_verts.shape[0],),
            fill_value=-1,
            dtype=ms.int64,
        )

    # TODO: Choose naive vs coarse-to-fine based on mesh size and image size.
    if bin_size is None:
        if not verts_packed.is_cuda:
            # Binned CPU rasterization is not supported.
            bin_size = 0
        else:
            # TODO better heuristics for bin size.
            if max_image_size <= 64:
                bin_size = 8
            else:
                # Heuristic based formula maps max_image_size -> bin_size as follows:
                # max_image_size < 64 -> 8
                # 16 < max_image_size < 256 -> 16
                # 256 < max_image_size < 512 -> 32
                # 512 < max_image_size < 1024 -> 64
                # 1024 < max_image_size < 2048 -> 128
                bin_size = int(
                    2 ** max(np.ceil(np.log2(max_image_size)) - 4, 4))

    if bin_size != 0:
        # There is a limit on the number of faces per bin in the cuda kernel.
        faces_per_bin = 1 + (max_image_size - 1) // bin_size
        if faces_per_bin >= kMaxFacesPerBin:
            raise ValueError(
                "bin_size too small, number of faces per bin must be less than %d; got %d"
                % (kMaxFacesPerBin, faces_per_bin)
            )

    if max_faces_per_bin is None:
        max_faces_per_bin = int(max(10000, meshes._F / 5))

    pix_to_face, zbuf, barycentric_coords, dists = _RasterizeFaceVerts.apply(
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        clipped_faces_neighbor_idx,
        im_size,
        blur_radius,
        faces_per_pixel,
        bin_size,
        max_faces_per_bin,
        perspective_correct,
        clip_barycentric_coords,
        cull_backfaces,
    )

    if z_clip_value is not None or cull_to_frustum:
        # If faces were clipped, map the rasterization result to be in terms of the
        # original unclipped faces.  This may involve converting barycentric
        # coordinates
        outputs = convert_clipped_rasterization_to_original_faces(
            pix_to_face,
            barycentric_coords,
            # pyre-fixme[61]: `clipped_faces` may not be initialized here.
            clipped_faces,
        )
        pix_to_face, barycentric_coords = outputs

    return pix_to_face, zbuf, barycentric_coords, dists
