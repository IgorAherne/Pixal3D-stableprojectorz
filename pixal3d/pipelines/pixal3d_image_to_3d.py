from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg, decode, proj_cond
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel


class Pixal3DImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Pixal3D (proj mode) image-to-3D models.

    Based on Trellis2 pipeline, using proj mode for inference.
    Each stage (SS, Shape 512, Shape 1024, Tex 1024) has its own image_cond_model (DinoV3ProjFeatureExtractor).
    Condition building uses camera-aware projection (requires camera_angle_x, distance, mesh_scale parameters).

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model_ss (nn.Module): Proj image cond model for sparse structure stage.
        image_cond_model_shape_512 (nn.Module): Proj image cond model for shape LR (512) stage.
        image_cond_model_shape_1024 (nn.Module): Proj image cond model for shape HR (1024) stage.
        image_cond_model_tex_1024 (nn.Module): Proj image cond model for texture (1024) stage.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model_ss: nn.Module = None,
        image_cond_model_shape_512: nn.Module = None,
        image_cond_model_shape_1024: nn.Module = None,
        image_cond_model_tex_1024: nn.Module = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model_ss = image_cond_model_ss
        self.image_cond_model_shape_512 = image_cond_model_shape_512
        self.image_cond_model_shape_1024 = image_cond_model_shape_1024
        self.image_cond_model_tex_1024 = image_cond_model_tex_1024
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pixal3DImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        # Proj mode: image_cond_models need to be loaded externally, set to None here
        pipeline.image_cond_model_ss = None
        pipeline.image_cond_model_shape_512 = None
        pipeline.image_cond_model_shape_1024 = None
        pipeline.image_cond_model_tex_1024 = None

        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image, bg_color: tuple = (0, 0, 0)) -> Image.Image:
        """
        Preprocess the input image.

        Args:
            input: Input image (RGB or RGBA).
            bg_color: Background color (R, G, B) in 0~255. Default black (0,0,0).
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        rgb = output[:, :, :3]
        a = output[:, :, 3:4]
        bg = np.array(bg_color, dtype=np.float32) / 255.0
        output = rgb * a + bg * (1.0 - a)
        output = Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))
        return output

    # =========================================================================
    # Proj mode condition building
    # =========================================================================

    def _cond_inputs(self, image_cond_model: nn.Module, image, camera_angle_x, distance, mesh_scale):
        """What a conditioning is built from: this stage's settings, DINOv3's tokens of its pictures, those
        pictures again (NAF's guide) and the camera. The multi-view pipeline overrides this and nothing else."""
        device = self.device
        spec = proj_cond.Spec.of(image_cond_model)
        pictures = proj_cond.image_tensor(image, spec.image_size, device)
        z = proj_cond.encode(image_cond_model.model, pictures)
        camera = proj_cond.Camera.single(
            {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}, device)
        return spec, z, pictures, camera

    @torch.no_grad()
    def get_proj_cond_ss(
        self,
        image: list,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
    ) -> dict:
        """
        Get proj conditioning for sparse structure stage.

        Args:
            image: List of PIL images.
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': ...}
        """
        image_cond_model = self.image_cond_model_ss
        if self.low_vram:
            image_cond_model.to(self.device)
        spec, z, _, camera = self._cond_inputs(image_cond_model, image, camera_angle_x, distance, mesh_scale)
        cond = proj_cond.cond_dense(spec, z, camera)
        if self.low_vram:
            image_cond_model.cpu()
            proj_cond.clear_grids()     # the projection grids are on the card too: they go with the extractor
        return {'cond': cond, 'neg_cond': proj_cond.negative(cond)}

    @torch.no_grad()
    def get_proj_cond_shape(
        self,
        image_cond_model: nn.Module,
        image: list,
        coords: torch.Tensor,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
        grid_resolution_override: int = None,
    ) -> dict:
        """
        Get proj conditioning for shape/texture stages (sparse-token aligned).

        Args:
            image_cond_model: The proj image cond model for this stage.
            image: List of PIL images.
            coords: Sparse structure coordinates [N, 4] (batch_idx, x, y, z).
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.
            grid_resolution_override: Override the grid resolution if not None.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': SparseTensor}
        """
        if self.low_vram:
            image_cond_model.to(self.device)
        spec, z, pictures, camera = self._cond_inputs(
            image_cond_model, image, camera_angle_x, distance, mesh_scale)
        image_cond_model._load_naf()
        # The HR cascade's grid floats with the token budget, so it can differ from what this stage was
        # trained at; proj_cond keeps one projection grid per side and leaves the extractor's own alone.
        grid = image_cond_model.grid_resolution if grid_resolution_override is None else grid_resolution_override
        cond = proj_cond.cond_tokens(
            spec, z, pictures, image_cond_model.naf_model, camera, coords, grid)
        feats = cond['proj']
        if self.low_vram:
            image_cond_model.cpu()
            proj_cond.clear_grids()
        return {
            'cond': {'global': cond['global'], 'proj': SparseTensor(feats=feats, coords=coords)},
            'neg_cond': {'global': torch.zeros_like(cond['global']),
                         'proj': SparseTensor(feats=torch.zeros_like(feats), coords=coords)},
        }

    # =========================================================================
    # Sampling methods (consistent with Trellis2)
    # =========================================================================

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure (proj)",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (proj)",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with cascade (LR → HR).
        
        Args:
            lr_cond (dict): The conditioning information for LR stage.
            cond (dict): The conditioning information for HR stage.
            flow_model_lr: LR flow model.
            flow_model: HR flow model.
            lr_resolution (int): LR resolution.
            resolution (int): Target HR resolution.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
            max_num_tokens (int): Maximum number of tokens.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling LR shape SLat (proj, 512)",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        decoder = self.models['shape_slat_decoder']
        if self.low_vram:
            decoder.to(self.device)
        hr_coords = decode.upsample(decoder, slat.feats, slat.coords)
        if self.low_vram:
            decoder.cpu()
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent (HR)
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc=f"Sampling HR shape SLat (proj, {hr_resolution})",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[Mesh, List[torch.Tensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            Mesh: The decoded mesh, before its holes are closed.
            List[torch.Tensor]: The four subdivision tables the texture decoder follows.
        """
        decoder = self.models['shape_slat_decoder']
        if self.low_vram:
            decoder.to(self.device)
        ret = decode.decode_shape(decoder, slat.feats, slat.coords, resolution)
        if self.low_vram:
            decoder.cpu()
        return ret

    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample texture structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat (proj)",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.
            subs (List[torch.Tensor]): The shape decode's subdivision tables.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The texture voxels' coords [N, 3] and their attributes [N, 6].
        """
        decoder = self.models['tex_slat_decoder']
        if self.low_vram:
            decoder.to(self.device)
        ret = decode.decode_texture(decoder, slat.feats, slat.coords, subs)
        if self.low_vram:
            decoder.cpu()
        return ret

    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        mesh, subs = self.decode_shape_slat(shape_slat, resolution)
        coords, attrs = self.decode_tex_slat(tex_slat, subs)
        decode.fill_holes(mesh)
        return [
            MeshWithVoxel(
                mesh.vertices, mesh.faces,
                origin = [-0.5, -0.5, -0.5],
                voxel_size = 1 / resolution,
                coords = coords,
                attrs = attrs,
                voxel_shape = torch.Size([1, attrs.shape[1], *(coords.max(0)[0] + 1).tolist()]),
                layout=self.pbr_attr_layout
            )
        ]
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        camera_params: dict,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the Pixal3D pipeline (proj mode, cascade).

        Args:
            image (Image.Image): The image prompt.
            camera_params (dict): Camera parameters with keys:
                - camera_angle_x (float): Horizontal FOV in radians.
                - distance (float): Camera distance.
                - mesh_scale (float): Mesh scale factor.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            hr_resolution = 1024
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            hr_resolution = 1536
        else:
            raise ValueError(f"Invalid pipeline type for Pixal3D proj mode: {pipeline_type}. "
                             f"Supported: '1024_cascade', '1536_cascade'.")

        # Validate image_cond_models are set
        assert self.image_cond_model_ss is not None, "image_cond_model_ss not set."
        assert self.image_cond_model_shape_512 is not None, "image_cond_model_shape_512 not set."
        assert self.image_cond_model_shape_1024 is not None, "image_cond_model_shape_1024 not set."
        assert self.image_cond_model_tex_1024 is not None, "image_cond_model_tex_1024 not set."

        # Extract camera params
        camera_angle_x = camera_params['camera_angle_x']
        distance = camera_params['distance']
        mesh_scale = camera_params.get('mesh_scale', 1.0)
        
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)

        # ---- Stage 1: Sparse Structure (proj) ----
        cond_ss = self.get_proj_cond_ss(
            [image],
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        ss_res = 32
        coords = self.sample_sparse_structure(
            cond_ss, ss_res,
            num_samples, sparse_structure_sampler_params
        )
        del cond_ss
        torch.cuda.empty_cache()

        # ---- Stage 2: Shape LR 512 (proj) ----
        cond_shape_lr = self.get_proj_cond_shape(
            self.image_cond_model_shape_512, [image], coords,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        lr_slat = self.sample_shape_slat(
            cond_shape_lr, self.models['shape_slat_flow_model_512'],
            coords, shape_slat_sampler_params
        )
        del cond_shape_lr
        torch.cuda.empty_cache()

        # ---- Stage 3a: Upsample LR → HR ----
        decoder = self.models['shape_slat_decoder']
        if self.low_vram:
            decoder.to(self.device)
        hr_coords = decode.upsample(decoder, lr_slat.feats, lr_slat.coords)
        if self.low_vram:
            decoder.cpu()

        lr_resolution = 512
        actual_hr_resolution = hr_resolution
        while True:
            grid_res = actual_hr_resolution // 16
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
            ], dim=1)
            hr_coords_unique = quant_coords.unique(dim=0)
            num_tokens = hr_coords_unique.shape[0]
            if num_tokens < max_num_tokens or actual_hr_resolution == 1024:
                break
            actual_hr_resolution -= 128

        actual_grid_res = actual_hr_resolution // 16
        del lr_slat, hr_coords, quant_coords
        torch.cuda.empty_cache()

        # ---- Stage 3b: Shape HR (proj) ----
        cond_shape_hr = self.get_proj_cond_shape(
            self.image_cond_model_shape_1024, [image], hr_coords_unique,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            grid_resolution_override=actual_grid_res,
        )
        noise_hr = SparseTensor(
            feats=torch.randn(hr_coords_unique.shape[0], self.models['shape_slat_flow_model_1024'].in_channels).to(self.device),
            coords=hr_coords_unique,
        )
        sampler_params_hr = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}
        flow_model_hr = self.models['shape_slat_flow_model_1024']
        if self.low_vram:
            flow_model_hr.to(self.device)
        hr_slat = self.shape_slat_sampler.sample(
            flow_model_hr,
            noise_hr,
            **cond_shape_hr,
            **sampler_params_hr,
            verbose=True,
            tqdm_desc=f"Sampling HR shape SLat (proj, {actual_hr_resolution})",
        ).samples
        if self.low_vram:
            flow_model_hr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(hr_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(hr_slat.device)
        shape_slat = hr_slat * std + mean
        del cond_shape_hr, noise_hr, hr_slat, hr_coords_unique
        torch.cuda.empty_cache()

        # ---- Stage 4: Texture (proj) ----
        tex_grid_res = actual_hr_resolution // 16
        cond_tex = self.get_proj_cond_shape(
            self.image_cond_model_tex_1024, [image], shape_slat.coords,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            grid_resolution_override=tex_grid_res,
        )
        tex_slat = self.sample_tex_slat(
            cond_tex, self.models['tex_slat_flow_model_1024'],
            shape_slat, tex_slat_sampler_params
        )
        del cond_tex
        torch.cuda.empty_cache()

        # ---- Stage 5: Decode ----
        res = actual_hr_resolution
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
