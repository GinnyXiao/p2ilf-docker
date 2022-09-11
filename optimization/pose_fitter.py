import torch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import numpy as np
from tqdm import tqdm
import copy

from .losses import kpts_fitting_loss

from pytorch3d.transforms import (
    axis_angle_to_matrix,
    Rotate,
    Translate)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes


# Optimization Model
class Model(nn.Module):
    def __init__(self, meshes, renderer, camera,
                 image_ref, kpts_3d, kpts_2d,
                 init_mesh_position=np.array([0., 0., 0.3]),
                 init_mesh_rotation=np.array([0., 0., 0.]),
                 kpt_weight=1, mask_weight=1):

        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        self.camera = camera

        self.kpt_weight = kpt_weight
        self.mask_weight = mask_weight

        # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        self.img_size = image_ref.shape
        self.register_buffer('image_ref', image_ref)

        # ground truth keypoints
        self.kpts_3d = kpts_3d
        self.kpts_2d = torch.from_numpy(kpts_2d).float().to(meshes.device)

        # Create an optimizable parameter for the x, y, z position of the mesh.
        self.mesh_position = nn.Parameter(
            torch.from_numpy(init_mesh_position).float().to(meshes.device))

        # Create an optimizable parameter for the axis-angle representation of the mesh orientation.
        # self.mesh_rotation = nn.Parameter(
        #     torch.from_numpy(np.array([9.7102e-08, 2.2214e+00, 2.2214e+00], dtype=np.float32)).to(meshes.device))
        self.mesh_rotation = nn.Parameter(
            torch.from_numpy(init_mesh_rotation).to(meshes.device))

    def forward(self):
        # Render the image using the updated mesh. Based on the new position & orientation of the
        # mesh we calculate the transformed vertices.
        R = axis_angle_to_matrix(self.mesh_rotation)
        t = Rotate(R[None, :]).compose(Translate(self.mesh_position[None, :]))
        tverts = t.transform_points(self.meshes.verts_packed())
        faces = self.meshes.faces_packed()

        # mesh_mv = copy.deepcopy(self.mesh).rotate(R, center=(0, 0, 0))
        # mesh_mv = mesh_mv.translate((self.mesh_position[0], self.mesh_position[1], self.mesh_position[2]))
        #
        # tverts = torch.tensor(mesh_mv.vertices).float()
        # faces = torch.tensor(mesh_mv.triangles).float()

        tmesh = Meshes(
            verts=[tverts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=self.meshes.textures
        )

        # Render sillouette
        image_render = self.renderer(tmesh)

        # Calculate the silhouette loss
        sil_loss = torch.sum((image_render[0, ..., 3] - self.image_ref) ** 2)

        # project 3d keypoints to screen, calculate the projection loss
        kpts_3d = tmesh.verts_packed()[self.kpts_3d]
        kpts_loss = kpts_fitting_loss(self.camera, self.img_size, kpts_3d, self.kpts_2d)

        # add losses
        loss = self.mask_weight * sil_loss + self.kpt_weight * kpts_loss

        return loss, image_render


class Pose_Fitter():

    def __init__(self, meshes, sil_renderer, phong_renderer, camera,
                 image_ref, kpts_3d, kpts_2d,
                 init_mesh_position=np.array([0., 0., 0.3]),
                 init_mesh_rotation=np.array([0., 0., 0.]),
                 kpt_weight=1, mask_weight=1,
                 global_iters=1000, lr=0.01, plot=False,
                 out_file="./p2ilf_optimization_demo2.gif"
                 ):

        self.device = meshes.device
        self.sil_renderer = sil_renderer
        self.phong_renderer = phong_renderer
        self.camera = camera

        # Store options
        self.global_iters = global_iters
        self.plot = plot

        self.writer = imageio.get_writer(out_file, mode='I', duration=0.3)

        self.model = Model(meshes=meshes, renderer=sil_renderer, camera=camera,
                           image_ref=image_ref, kpts_3d=kpts_3d, kpts_2d=kpts_2d,
                           init_mesh_position=init_mesh_position, init_mesh_rotation=init_mesh_rotation,
                           kpt_weight=kpt_weight, mask_weight=mask_weight).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))

    def __call__(self):
        loop = tqdm(range(self.global_iters))
        for i in loop:
            self.optimizer.zero_grad()
            loss, _ = self.model()
            loss.backward()
            self.optimizer.step()

            loop.set_description('Optimizing (loss %.4f)' % loss.data)

            # Save outputs to create a GIF.
            if i % 10 == 0:
                R = axis_angle_to_matrix(self.model.mesh_rotation)
                t = Rotate(R[None, :]).compose(Translate(self.model.mesh_position[None, :]))
                tverts = t.transform_points(self.model.meshes.verts_packed())
                faces = self.model.meshes.faces_packed()
                # verts_rgb = torch.ones_like(tverts)[None]  # (1, V, 3)
                # textures = TexturesVertex(verts_features=verts_rgb.to(device))
                tmesh = Meshes(
                    verts=[tverts.to(self.device)],
                    faces=[faces.to(self.device)],
                    textures=self.model.meshes.textures
                )

                image = self.phong_renderer(tmesh)
                image = image[0, ..., :3].detach().squeeze().cpu().numpy()
                image = img_as_ubyte(image)
                self.writer.append_data(image)

                if self.plot:
                    if i % 100 == 0:
                        plt.figure()
                        plt.imshow(image[..., :3])
                        plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
                        plt.axis("off")

        self.writer.close()