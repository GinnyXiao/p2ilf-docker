import os
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_mask_detector(ckpt, device):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = str(device)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_1500.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = ckpt
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.DETECTIONS_PER_IMAGE = 1
    mask_rcnn_predictor = DefaultPredictor(cfg)
    return mask_rcnn_predictor


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print("device: ", device)

    ckpt = os.path.join("models/", "model_final.pth")

    detector = get_mask_detector(ckpt, device)

    test_img = np.array(Image.open('tryme/image.jpg'))
    mask_rcnn_outputs = detector(test_img)

    f, axarr = plt.subplots(1, 2, figsize=(20, 10))
    axarr[0].imshow(test_img)
    axarr[1].imshow(
        255 * mask_rcnn_outputs["instances"].pred_masks.numpy().squeeze()[:, :, None].repeat(3, axis=2))
    plt.savefig('tryme/mask.jpg')

