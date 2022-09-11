import SimpleITK
from pathlib import Path

import torch
import os
import numpy as np
import json
import skimage.transform
import cv2
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio


# Util function for loading meshes
from pytorch3d.io import load_obj, save_obj

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex
)

from pytorch3d.transforms import (
    euler_angles_to_matrix,
    axis_angle_to_matrix,
    Rotate,
    Translate,
    Transform3d,
    matrix_to_axis_angle)

from pytorch3d.renderer import (
    FoVPerspectiveCameras)


from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import evalutils

from models.mask_detector import get_mask_detector
from optimization.pose_fitter_mask import Pose_Fitter_Mask
from optimization.phong_renderer import Phong_Renderer
from optimization.sillouette_renderder import Silouette_Renderer


execute_in_docker = False
useOnly2DSeg = 0  # Set flag for 2D segmentation only --> set 0 if you are doing both 2D and 3D contour segmentation

# If you are doing task 1: set only useOnly3DSeg (2D and 3D contour) i.e. set useOnly2DSeg = 0
# if you are doing task 2: set both useOnly3DSeg and useReg to 1  i.e. set useOnly2DSeg = 0

useOnly3DSeg = 1  # set:1 for all three (2D seg, 3D seg and/or 2D-3D registration outputs)
useReg = 1  # set 1 if you are doing 2d-3d registeration --> for all three (2D seg, 3D seg and 2D-3D registration outputs)


class P2ILFChallenge(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            # Reading all input files
            input_path=Path("/input/images/laparoscopic-image/") if execute_in_docker else \
                       Path("tryme/images/laparoscopic-image/"),
            output_file=[Path("/output/2d-liver-contours.json"), Path("/output/3d-liver-contours.json"), \
                         Path("/output/transformed-3d-liver-model.obj")] if execute_in_docker else \
                        [Path("./output/2d-liver-contours.json"), Path("./output/3d-liver-contours.json"), \
                         Path("./output/transformed-3d-liver-model.obj")]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)

        if execute_in_docker:
            path_model = "/opt/algorithm/ckpt/CP50_criteria.pth"
        else:
            path_model = "./models/model_1500.pth"

        self.mask_rcnn_predictor = get_mask_detector(path_model, self.device)
        print("Mask-rcnn model weight loaded")

    '''Instruction 1: YOU will need to work here for writing your results as per your task'''

    def save(self):
        if (useOnly2DSeg == 1):
            # logging first output file  /output/2d-liver-contours.json
            # with open(str(self._output_file[0]), "w") as f:
            #     json.dump(self._case_results[0][0], f)

            # create dummy files
            shutil.copyfile('./dummy/2d-liver-contours.json', self._output_file[0])
            shutil.copyfile('./dummy/3d-liver-contours.json', self._output_file[1])
            # shutil.copyfile('./dummy/transformed-3d-liver-model.obj', self._output_file[2])

        if useOnly3DSeg == 1 or useReg == 1:
            # Hint you can append the results for 2D and 3D segmentation contour dictionaries
            print(
                '\n 3D seg or registration flag on ---> Could not save the contours as your are returning only one result - try appending your results')

            # for i in range(0, 2):
            #     # print(len(self._case_results[i]))
            #     with open(str(self._output_file[i]), "w") as f:
            #         json.dump(self._case_results[0][i], f)

            shutil.copyfile('./dummy/2d-liver-contours.json', self._output_file[0])
            shutil.copyfile('./dummy/3d-liver-contours.json', self._output_file[1])

        if useReg == 0:
            # This is because you need to write all files
            shutil.copyfile('./dummy/transformed-3d-liver-model.obj', self._output_file[2])

        if useReg:
            print('\n writing our transformed-3d-liver-model.obj')

            # TODO: you can write the registration file to the idx - [2] instead of dummy file
            # Let us know if you will need help to figure this out



        print('\n All files written successfully')

    '''Instruction 2: YOU will need to work here for appending your 2D and 3D contour results'''

    def process_case(self, *, idx, case):
        results_append = []
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Detect and score candidates
        results1 = self.predict(input_image=input_image)
        results_append.append(results1)
        # you can call your 3D predict function and log in your result
        # results2 = self.predict_3D(input_image=input_image)

        if useOnly3DSeg:
            # Write resulting candidates to result.json for this case
            results2 = self.predict2()
            results_append.append(results2)

        return results_append

    # Sample provided - write your 3D contour prediction ehre
    def predict2(self):
        """
        2: Save your Output : /output/3d-liver-contours.json
        """
        data_params = []

        return data_params

    ''' Instruction 3: YOU will need to write similar functins for your 2D, 3D and registration - 
    these predict functions can be called by process_case for logging in results -> Tuple[SimpleITK.Image, Path]:'''

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:

        # Hard coded paths
        if execute_in_docker:
            input_path_mesh = Path('/input/3D-liver-model.obj')
            input_path_K = Path('/input/acquisition-camera-metadata.json')

        else:
            input_path_mesh = Path('./tryme/3D-liver-model.obj')
            input_path_K = Path('./tryme/cameraparams.json')

        # ------------------- Load Image Data ---------------------

        image = SimpleITK.GetArrayFromImage(input_image)
        image = np.array(image)
        shape = image.shape
        print(shape)

        scale = 4
        train_size = (shape[0] // scale, shape[1] // scale)

        # ------------------- Load Mesh ---------------------

        # Load the obj and ignore the textures and materials.
        verts, faces_idx, _ = load_obj(input_path_mesh)
        faces = faces_idx.verts_idx

        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

        # Create a Meshes object for the liver. Here we have only one mesh in the batch.
        init_mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures
        )

        # The mesh is in mm scale; we gotta change it to meter scale
        init_mesh.scale_verts_(0.001)

        # offset the mesh to the origin
        center = init_mesh.verts_packed().mean(0)
        offset = torch.zeros(3).to(self.device) - center
        verts_off = Translate(offset[None, :], device=self.device).transform_points(init_mesh.verts_packed())
        mesh = Meshes(
            verts=[verts_off.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures
        )

        # -------------- Set up Camera and Renderer ---------------

        f = open(input_path_K)  # camera parameters
        camera_params = json.load(f)
        fy = float(camera_params['fy'])

        R = torch.tensor([[-1., 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]]).repeat(1, 1, 1).to(self.device)
        T = torch.tensor([0, 0, 0], device=self.device).unsqueeze(dim=0)
        fov = 2 * np.arctan(train_size[0] / (fy * 2))  # * 180 / np.pi

        # create camera and renderers
        camera = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov, degrees=False, znear=0.01)
        phong_renderer = Phong_Renderer(train_size, camera, device=self.device)
        silhouette_renderer = Silouette_Renderer(train_size, camera, device=self.device)

        # ------------------- Start Algorithm -----------------

        # run mask rcnn prediction on the image
        mask_rcnn_outputs = self.mask_rcnn_predictor(image)

        # scale the image and kpts
        mask = mask_rcnn_outputs["instances"].pred_masks.cpu().numpy().squeeze().astype(np.uint8)
        image_ref = cv2.resize(mask, dsize=(train_size[1], train_size[0]), interpolation=cv2.INTER_CUBIC)

        # set up output demo
        out_file = "tryme/p2ilf_optimization_demo.gif"

        # run optimization
        init_mesh_position = np.array([0, 0, 0.3])
        init_mesh_rotation = np.array([-1.5708e+00, 6.8662e-08, 6.8662e-08])
        pose_fitter = Pose_Fitter_Mask(mesh, silhouette_renderer, phong_renderer, camera,
                                       image_ref,
                                       init_mesh_position, init_mesh_rotation,
                                       kpt_weight=1., mask_weight=1.,
                                       global_iters=200, lr=0.01, plot=False,
                                       out_file=out_file)

        pose_fitter()

        # Visualize final results

        pose_fitter.model.meshes.scale_verts_(1000)

        R = axis_angle_to_matrix(pose_fitter.model.mesh_rotation)
        t = Rotate(R[None, :]).compose(Translate(pose_fitter.model.mesh_position[None, :] * 1000))
        tverts = t.transform_points(pose_fitter.model.meshes.verts_packed())
        faces = pose_fitter.model.meshes.faces_packed()

        tmesh = Meshes(
            verts=[tverts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures
        )

        image_render = phong_renderer(tmesh)
        image_render = image_render[0, ..., :3].detach().squeeze().cpu().numpy()

        out_img_file = "tryme/p2ilf_optimization_result_idx.png"
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image_render)
        plt.grid(False)
        plt.title("Optimized position")

        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.grid(False)
        plt.title("Reference image")

        plt.savefig(out_img_file)


        """
        3: Save your Output : /output/transformed-3d-liver-model.obj.json
        """

        out_mesh = Path("/output/transformed-3d-liver-model.obj") if execute_in_docker else \
                   Path("./output/transformed-3d-liver-model.obj")
        save_obj(out_mesh, tverts, faces)

        """
        1: Save your Output : /output/2d-liver-contours.json
        """
        my_dictionary = {"numOfContours": 0, "contour": []}

        return my_dictionary


if __name__ == "__main__":
    P2ILFChallenge().process()