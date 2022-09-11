import numpy as np
import torch

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, SoftSilhouetteShader,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams
)


class Silouette_Renderer():
    def __init__(self, size, camera, device='cpu'):
        self.device = device
        self.size = size

        self.camera = camera
        self.silhouette_renderer = self.init_silhouette_renderer()

    def init_silhouette_renderer(self):
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=self.size,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100)

        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
        return silhouette_renderer

    def __call__(self, mesh):
        ''' Right now only render silhouettes
            Input:
            vertices: BN * V * 3
            faces: BN * F * 3
        '''

        silhouette = self.silhouette_renderer(meshes_world=mesh.clone())

        return silhouette