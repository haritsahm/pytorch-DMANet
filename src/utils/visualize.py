# Source: https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/utils/visualize.py

import cv2
import numpy as np

cityspallete = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]


def set_img_color(img, label, pallete, background=-1, show255=False):
    """Apply color to target class in the image."""

    for i in range(len(pallete)):
        if i != background:
            img[np.where(label == i)] = pallete[i]
    if show255:
        img[np.where(label == 255)] = 255

    return img


def show_prediction(img, pred, colors: str = 'cityscape', overlay: float = 1.0, background=-1):
    """Overlay the output mask with colored class."""

    if colors == 'cityscape':
        pallete = cityspallete
    im = np.array(img, np.uint8)
    mask = im.copy()
    set_img_color(mask, pred, pallete, background)
    out = cv2.addWeighted(im, (1 - overlay), mask, overlay, 0.0)

    return out
