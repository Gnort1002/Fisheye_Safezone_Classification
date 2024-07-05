from models.enet import ENet
from models.mobile_unet import MobileUNet
from models.pspnet import PSPNet

import torch
from engine.utils import load_checkpoint, create_overlayed_image, model_info
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from args import get_arguments
import time
args = get_arguments()


def detect(model, image, imgsz=(480, 360)):
    """Detects the drivable area in a given image.

    Keyword arguments:
    - model (``torch.nn``): The model to use for darea detection.
    - image (``torch.Tensor``): The image to detect the drivable area in.

    Returns:
    The binary mask with the detected darea in value 255 and the rest 0.

    """
    # Convert image to tensor
    image_transform = A.Compose(
        [A.Resize(imgsz[1], imgsz[0]),
         ToTensorV2()])

    image = image.astype(np.float32) / 255
    augmented = image_transform(image=image)
    image = augmented['image']
    image = image.unsqueeze(0).to("cuda")


    # Detect darea
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        darea = model(image)
    end_time = time.time()
    if args.debug:
        print("Inference time: ", end_time - start_time)
    predicted_mask = darea[0].permute(1, 2, 0).detach().cpu().numpy()[:,:,0]
    predicted_mask_binary = (predicted_mask < 0.5).astype(np.uint8) * 255
    return predicted_mask_binary


if __name__ == "__main__":
    # Load model
    if args.model.lower() == "enet":
        model = ENet(2)
    if args.model.lower() == "m_unet":
        model = MobileUNet(2)
    if args.model.lower() == "pspnet":
        model = PSPNet(layers=50, classes=2)
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer, epoch, miou = load_checkpoint(
        model,
        optimizer,
        args.save_dir,
        args.name)
    model_info(model, (args.width, args.height))
    model = model.to("cuda")
    mask_color = (0, 255, 0)
    out_dir = args.mask_out_dir
    img_dir = args.dataset_dir
    for img_name in os.listdir(img_dir):
        img_pth = os.path.join(img_dir, img_name)
        image = cv2.imread(img_pth)
        darea = detect(model, image, (args.width, args.height))
        masked_overlay = create_overlayed_image(image, darea, mask_color)
        cv2.imwrite(os.path.join(out_dir, img_name), masked_overlay)
        # if args.debug:
        #     print("Saved image: ", img_name)
