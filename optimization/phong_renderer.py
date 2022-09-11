import numpy as np
import torch

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    BlendParams, PointLights, HardPhongShader, SoftPhongShader,
    RasterizationSettings, MeshRenderer, MeshRasterizer
)


class Phong_Renderer():
    def __init__(self, size, camera, device='cpu'):
        self.device = device
        self.size = size

        self.camera = camera
        self.phong_renderer = self.init_phong_renderer()

    def init_phong_renderer(self):
        blend_params = BlendParams(sigma=1e-4, gamma=0.9)
        raster_settings = RasterizationSettings(
            image_size=self.size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        # We can add a point light in front of the object.
        lights = PointLights(device=self.device, location=((2.0, 2.0, -2.0),))
        phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(blend_params=blend_params, device=self.device, cameras=self.camera, lights=lights)
        )
        return phong_renderer

    def __call__(self, mesh):
        ''' Right now only render silhouettes
            Input:
            vertices: BN * V * 3
            faces: BN * F * 3
        '''

        image = self.phong_renderer(meshes_world=mesh.clone())

        return image