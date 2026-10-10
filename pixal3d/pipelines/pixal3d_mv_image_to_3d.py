from typing import *
import torch
from .pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
from . import proj_cond


class Pixal3DMVImageTo3DPipeline(Pixal3DImageTo3DPipeline):
    """
    Multi-view variant of the Pixal3D proj pipeline.

    Same cascade as the single-view pipeline (SS -> Shape 512 -> Shape 1024 -> Tex 1024);
    the only difference is the image condition: instead of one image at the canonical
    front view, it takes V views with explicit c2w matrices and lets
    DinoV3ProjMultiViewFeatureExtractor project all of them into the shared 3D grid,
    averaging the per-view features.

    Why the denoisers are unchanged: with multiview_fusion="average" the fused z_proj
    has exactly the single-view shape [B, R^3, C], so every per-block proj_linear /
    proj cross-attn weight stays compatible. View 0 is the main view and its
    calc_mat_0 == F by construction, so V=1 degenerates to the single-view path.

    This class only overrides what a condition is built from (`_cond_inputs`); the two
    builders themselves are inherited. `run()` is inherited verbatim: there, `image` and
    `camera_params` are pure pass-through to those builders, so we route the whole
    multi-view bundle through them (see `run_mv`).
    """

    def _views_of(self, image) -> dict:
        """Unwrap the views bundle that `run()` passes along as its `image` argument."""
        views = image[0] if isinstance(image, (list, tuple)) else image
        if not isinstance(views, dict) or 'images' not in views:
            raise TypeError(
                "Pixal3DMVImageTo3DPipeline expects a views dict (see run_mv), "
                f"got {type(views).__name__}. Use run_mv() instead of run().")
        return views

    def _cond_inputs(self, image_cond_model, image, camera_angle_x, distance, mesh_scale):
        """The bundle's views at this stage's size (already resized and alpha-premultiplied, unlike the
        single-view path which takes PIL and resizes itself), DINOv3's tokens of all of them, and the camera
        of each view. The angles, distances and scale come from the arguments, as `run()` forwards them; only
        the c2w matrices are read from the bundle."""
        device = self.device
        views = self._views_of(image)
        spec = proj_cond.Spec.of(image_cond_model)
        pictures = proj_cond.image_tensor(views, spec.image_size, device)
        z = proj_cond.encode(image_cond_model.model, pictures)
        transform_matrix = views['transform_matrix'].to(device)
        camera = proj_cond.Camera(
            camera_angle_x=torch.as_tensor(camera_angle_x, dtype=torch.float32, device=device),
            distance=torch.as_tensor(distance, dtype=torch.float32, device=device),
            # mesh_scale is per-object, i.e. [B]; the projection expands it over the views.
            mesh_scale=mesh_scale,
            transform_matrix=transform_matrix,
            views=transform_matrix.shape[1],
        )
        return spec, z, pictures, camera

    @torch.no_grad()
    def run_mv(self, views: dict, **kwargs):
        """
        Run the cascade on a multi-view bundle.

        Args:
            views: as built by inference_mv.load_views():
                images:           {image_size: [1, V, 3, H, W]} alpha-premultiplied
                camera_angle_x:   [1, V] horizontal fov in radians
                camera_distance:  [1, V] camera distance (norm of the c2w translation)
                transform_matrix: [1, V, 4, 4] c2w, index 0 = main view
                mesh_scale:       float
            **kwargs: forwarded to the inherited run() (seed, sampler params,
                pipeline_type, max_num_tokens, return_latent, ...).

        The bundle rides along as run()'s `image` argument and the camera tensors as its
        `camera_params`; run() only forwards both to the condition builders, which take
        them apart in `_cond_inputs`.
        """
        return super().run(
            views,
            camera_params={
                'camera_angle_x': views['camera_angle_x'],
                'distance': views['camera_distance'],
                'mesh_scale': views['mesh_scale'],
            },
            preprocess_image=False,
            **kwargs,
        )
