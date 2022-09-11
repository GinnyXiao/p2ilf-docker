from pytorch3d.renderer import FoVPerspectiveCameras
import torch

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def kpts_fitting_loss(camera, image_size, kpts_3d, kpts_2d):
    # Project model keypoints
    projected_keypoints = camera.transform_points_screen(kpts_3d, image_size=image_size)
    sigma = 50

    # Weighted robust reprojection loss
    reprojection_error = gmof(projected_keypoints[:, [0, 1]] - kpts_2d, sigma)
    total_loss = reprojection_error.sum(dim=-1)
    return total_loss.sum()