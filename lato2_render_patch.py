# lato2_render_patch.py
# File: lato2_render_patch.py
"""
Windows replacement for LATO.2's Open3D conditioning renderer.

LATO.2 renders one white-model view of the input mesh and encodes it with DINOv2
as a conditioning signal. Upstream does that through
`open3d.visualization.rendering.OffscreenRenderer`, which cannot work on Windows:
Open3D's python binding calls `EngineInstance::EnableHeadless()` unconditionally
(cpp/pybind/visualization/rendering/rendering.cpp), and the headless path is
guarded by `#ifdef __linux__`, so every construction raises

    [Open3D Error] EGL Headless is not supported on this platform.

That is true of 0.17, 0.18 and 0.19 alike, so there is no version to fall back to.

This module provides an nvdiffrast-backed renderer with the same constructor and
`render()` contract, and monkey-patches it over the Open3D one. nvdiffrast is
already a hard dependency of Pixal3D, and its CUDA rasteriser needs no OpenGL
context, no window and no display - which suits an unattended install better than
Open3D did even on Linux.

Framing is identical to upstream: the camera placement (`_orbit_eye`) and the
crop-to-object step are reused from LATO.2's own module, so only the shading model
differs. Shading is a Lambertian approximation of Filament's sun + image-based
light rather than a pixel-exact match.
"""
import math

import numpy as np

# Linear-space shading constants, tuned to approximate Open3D's
# sun_intensity=90000 / ambient_intensity=32000 white-model look.
AMBIENT = 0.35
GAMMA = 2.2


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    fwd = center - eye
    fwd = fwd / max(np.linalg.norm(fwd), 1e-12)
    right = np.cross(fwd, up)
    n = np.linalg.norm(right)
    if n < 1e-9:
        # Degenerate when looking straight along `up`; nudge to a stable basis.
        right = np.cross(fwd, np.roll(up, 1))
        n = np.linalg.norm(right)
    right = right / max(n, 1e-12)
    true_up = np.cross(right, fwd)

    view = np.eye(4, dtype=np.float64)
    view[0, :3], view[1, :3], view[2, :3] = right, true_up, -fwd
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2.0 * far * near / (near - far)
    proj[3, 2] = -1.0
    return proj


class NvdiffrastWhiteModelRenderer:
    """Drop-in for LATO.2's WhiteModelRenderer, rasterised with nvdiffrast."""

    _glctx = None  # one CUDA raster context per process

    def __init__(
        self,
        img_res: int = 512,
        mesh_color=(0.78, 0.78, 0.82),
        bg_color=(1.0, 1.0, 1.0),
        up_axis: str = "y",
        add_ground: bool = True,
        shadow: bool = True,
        elevation_range=(15.0, 40.0),
        azimuth_range=(0.0, 360.0),
        camera_distance: float = 1.8,
        fov: float = 50.0,
        ground_color=(0.92, 0.92, 0.92),
        sun_intensity: float = 90000.0,
        ambient_intensity: float = 32000.0,
        crop_to_object: bool = False,
        crop_padding: float = 1.2,
    ):
        from dataset.mesh_render import _to_rgb01

        self.img_res = int(img_res)
        self.mesh_color = _to_rgb01(mesh_color)
        self.bg_color = _to_rgb01(bg_color)
        self.up_axis = up_axis.lower()
        # add_ground / shadow are accepted for signature parity. LATO.2 calls this
        # with add_ground=False, and with no ground plane the only shadowing left
        # is self-shadowing, which the Lambertian term already approximates.
        self.add_ground = add_ground
        self.shadow = shadow
        self.elevation_range = elevation_range
        self.azimuth_range = azimuth_range
        self.camera_distance = float(camera_distance)
        self.fov = float(fov)
        self.ground_color = _to_rgb01(ground_color)
        self.sun_intensity = float(sun_intensity)
        self.ambient_intensity = float(ambient_intensity)
        self.crop_to_object = crop_to_object
        self.crop_padding = float(crop_padding)

        self._rng = np.random.default_rng()

    @classmethod
    def _context(cls):
        if cls._glctx is None:
            import nvdiffrast.torch as dr

            # CUDA rasteriser, not the OpenGL one: no context, window or display.
            cls._glctx = dr.RasterizeCudaContext()
        return cls._glctx

    def _shade(self, vertices: np.ndarray, faces: np.ndarray, eye, center, up):
        import nvdiffrast.torch as dr
        import torch

        from dataset.mesh_render import _axis_index

        dev = torch.device("cuda")
        verts = torch.as_tensor(np.ascontiguousarray(vertices), dtype=torch.float32, device=dev)
        tris = torch.as_tensor(np.ascontiguousarray(faces), dtype=torch.int32, device=dev)

        # Smooth vertex normals, matching Open3D's compute_vertex_normals().
        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        face_n = torch.linalg.cross(v1 - v0, v2 - v0)
        normals = torch.zeros_like(verts)
        for i in range(3):
            normals.index_add_(0, tris[:, i].long(), face_n)
        normals = torch.nn.functional.normalize(normals, dim=1)

        mvp = _perspective(self.fov, 1.0, 0.05, 100.0) @ _look_at(eye, center, up)
        mvp_t = torch.as_tensor(mvp, dtype=torch.float32, device=dev)
        hom = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=1)
        clip = (hom @ mvp_t.T)[None]

        # The CUDA rasteriser wants both dimensions to be multiples of 8.
        res = (self.img_res + 7) // 8 * 8
        rast, _ = dr.rasterize(self._context(), clip, tris, resolution=[res, res])

        normal_px, _ = dr.interpolate(normals[None], rast, tris)
        normal_px = torch.nn.functional.normalize(normal_px, dim=-1)

        # Open3D points the sun along +0.35/-1/+0.35 (for up='y'); the incident
        # direction is its negation.
        sun_dir = np.full(3, 0.35)
        sun_dir[_axis_index(self.up_axis)] = -1.0
        sun_dir = sun_dir / np.linalg.norm(sun_dir)
        light = torch.as_tensor(-sun_dir, dtype=torch.float32, device=dev)

        ndl = torch.clamp((normal_px * light).sum(-1, keepdim=True), 0.0, 1.0)
        base = torch.as_tensor(self.mesh_color, dtype=torch.float32, device=dev)
        lit = base * (AMBIENT + (1.0 - AMBIENT) * ndl)

        bg = torch.as_tensor(self.bg_color, dtype=torch.float32, device=dev)
        mask = (rast[..., 3:4] > 0).float()
        color = lit * mask + bg * (1.0 - mask)
        color = dr.antialias(color.contiguous(), rast, clip, tris)

        srgb = torch.clamp(color, 0.0, 1.0) ** (1.0 / GAMMA)
        rgb = (srgb[0] * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        cov = (mask[0, ..., 0] > 0).cpu().numpy()
        # Trim the multiple-of-8 padding back to the requested resolution.
        return rgb[: self.img_res, : self.img_res], cov[: self.img_res, : self.img_res]

    def render(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        num_views: int = 1,
        mesh_color=None,
        azimuths=None,
        elevations=None,
        seed=None,
    ):
        from dataset.mesh_render import _orbit_eye, _to_rgb01

        rng = np.random.default_rng(seed) if seed is not None else self._rng
        if mesh_color is not None:
            self.mesh_color = _to_rgb01(mesh_color)

        vertices = np.asarray(vertices, dtype=np.float64)
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0

        images, params = [], []
        for v in range(num_views):
            az = float(azimuths[v]) if azimuths is not None else float(rng.uniform(*self.azimuth_range))
            el = float(elevations[v]) if elevations is not None else float(rng.uniform(*self.elevation_range))

            eye, up = _orbit_eye(center, self.camera_distance, az, el, self.up_axis)
            rgb, mask = self._shade(vertices, np.asarray(faces), eye, center, up)

            if self.crop_to_object:
                rgb = self._crop_resize_to_object(rgb, mask)

            images.append(np.ascontiguousarray(rgb))
            params.append({"azimuth": az, "elevation": el, "distance": self.camera_distance})

        return images, params


def apply():
    """Swap the nvdiffrast renderer in for LATO.2's Open3D one.

    Must run before dataset.voxel_dataset._render_image does its lazy
    `from dataset.mesh_render import WhiteModelRenderer`.
    """
    import dataset.mesh_render as mr

    # Reuse upstream's crop so framing stays bit-identical; it only touches
    # self.img_res and self.crop_padding.
    NvdiffrastWhiteModelRenderer._crop_resize_to_object = mr.WhiteModelRenderer._crop_resize_to_object
    mr.WhiteModelRenderer = NvdiffrastWhiteModelRenderer
    print("[LATO.2] Conditioning renderer: nvdiffrast (Open3D offscreen is Linux-only)")
