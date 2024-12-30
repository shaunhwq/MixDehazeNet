import os
from typing import Tuple

import cv2
import numpy as np
import torch

from models.MixDehazeNet import MixDehazeNet


def shortside_resize(image: np.array, min_size: int = 256):
    h, w, _ = image.shape
    is_landscape = w > h
    aspect_ratio = h / w if not is_landscape else w / h

    new_shape = [int(min_size), int(min_size * aspect_ratio)]
    if is_landscape:
        new_shape = new_shape[::-1]
    return cv2.resize(image, new_shape)


def pre_process(image: np.array, device: str, min_size: int = 256) -> torch.Tensor:
    """
    Note: Model was trained on 256x256 crops, might be abit strange that we doing shortside resize

    :param image: Input image to transform to the model input
    :param device: Device to send input to
    :param min_size: Minimum size for the lower dim of [h, w]
    :returns: Tensor input to model, in the shape [b, c, h, w]
    """
    image = shortside_resize(image, min_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = image * 2 - 1

    image = torch.from_numpy(image).permute(2, 0, 1)
    image = image.unsqueeze(0).contiguous().to(device)
    return image


def post_process(model_output: torch.Tensor, input_hw: Tuple[int, int]) -> np.array:
    """
    :param model_output: Output tensor produced by the model [b, c, h, w]
    :param input_hw: Tuple containing input image height and width
    :returns: Output image which can be displayed by OpenCV
    """
    image_tensor = model_output.clamp_(-1, 1)
    # [-1, 1] to [0, 1]
    image_tensor = image_tensor * 0.5 + 0.5

    image_rgb = image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    image_rgb = (image_rgb * 255).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Resize to original
    h, w, c = image_bgr.shape
    in_h, in_w = input_hw
    if not (in_h == h and in_w == w):
        image_bgr = cv2.resize(image_bgr, input_hw[::-1])

    return image_bgr


if __name__ == "__main__":
    video_path = "/Users/shaun/datasets/image_enhancement/dehaze/DVD/DrivingHazy/31_hazy_video.mp4"
    device = "mps"
    weights_path = "weights/Haze4k/MixDehazeNet-l.pth"
    shortside_min_size = 512

    model_name = os.path.splitext(os.path.basename(weights_path).replace("-", "_"))[0]

    model_kwargs = dict(
        MixDehazeNet_t = dict(embed_dims=[24, 48, 96, 48, 24], depths=[1, 1, 2, 1, 1]),
        MixDehazeNet_s = dict(embed_dims=[24, 48, 96, 48, 24], depths=[2, 2, 4, 2, 2]),
        MixDehazeNet_b = dict(embed_dims=[24, 48, 96, 48, 24], depths=[4, 4, 8, 4, 4]),
        MixDehazeNet_l = dict(embed_dims=[24, 48, 96, 48, 24], depths=[8, 8, 16, 8, 8]),
    ).get(model_name)

    assert model_kwargs is not None, "Invalid selection for model"

    # Initialize model
    model = MixDehazeNet(**model_kwargs)
    weights = torch.load(weights_path, map_location="cpu")['state_dict']
    weights = {strKey.replace('module.', ''): weight for strKey, weight in weights.items()}
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)

    while True:
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()

        in_tensor = pre_process(frame, device, shortside_min_size)
        with torch.no_grad():
            model_outputs = model(in_tensor)
        out_image = post_process(model_outputs, frame.shape[:2])

        display_image = np.vstack([frame, out_image])

        cv2.imshow("output", display_image)
        key = cv2.waitKey(1)
        if key & 255 == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
