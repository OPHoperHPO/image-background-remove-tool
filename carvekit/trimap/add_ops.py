"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import cv2
import numpy as np
from PIL import Image


def prob_filter(mask: Image.Image, prob_threshold=231) -> Image.Image:
    """
    Applies a filter to the mask by the probability of locating an object in the object area.

    Args:
        prob_threshold (int, default=231): Threshold of probability for mark area as background.
        mask (Image.Image): Predicted object mask

    Raises:
        ValueError: if mask or trimap has wrong color mode

    Returns:
        Image.Image: generated trimap for image.
    """
    if mask.mode != "L":
        raise ValueError("Input mask has wrong color mode.")
    # noinspection PyTypeChecker
    mask_array = np.array(mask)
    mask_array[mask_array > prob_threshold] = 255  # Probability filter for mask
    mask_array[mask_array <= prob_threshold] = 0
    return Image.fromarray(mask_array).convert("L")


def prob_as_unknown_area(
    trimap: Image.Image, mask: Image.Image, prob_threshold=255
) -> Image.Image:
    """
    Marks any uncertainty in the seg mask as an unknown region.

    Args:
        prob_threshold (int, default=255): Threshold of probability for mark area as unknown.
        trimap (Image.Image): Generated trimap.
        mask (Image.Image): Predicted object mask

    Raises:
        ValueError: if mask or trimap has wrong color mode

    Returns:
        Image.Image: Generated trimap for image.
    """
    if mask.mode != "L" or trimap.mode != "L":
        raise ValueError("Input mask has wrong color mode.")
    # noinspection PyTypeChecker
    mask_array = np.array(mask)
    # noinspection PyTypeChecker
    trimap_array = np.array(trimap)
    trimap_array[np.logical_and(mask_array <= prob_threshold, mask_array > 0)] = 127
    return Image.fromarray(trimap_array).convert("L")


def post_erosion(trimap: Image.Image, erosion_iters=1) -> Image.Image:
    """
    Performs erosion on the mask and marks the resulting area as an unknown region.

    Args:
        erosion_iters (int, default=1): The number of iterations of erosion that
        the object's mask will be subjected to before forming an unknown area
        trimap (Image.Image): Generated trimap.

    Returns:
        Image.Image: Generated trimap for image.
    """
    if trimap.mode != "L":
        raise ValueError("Input mask has wrong color mode.")
    # noinspection PyTypeChecker
    trimap_array = np.array(trimap)
    if erosion_iters > 0:
        without_unknown_area = trimap_array.copy()
        without_unknown_area[without_unknown_area == 127] = 0

        erosion_kernel = np.ones((3, 3), np.uint8)
        erode = cv2.erode(
            without_unknown_area, erosion_kernel, iterations=erosion_iters
        )
        erode = np.where(erode == 0, 0, without_unknown_area)
        trimap_array[np.logical_and(erode == 0, without_unknown_area > 0)] = 127
        erode = trimap_array.copy()
    else:
        erode = trimap_array.copy()
    return Image.fromarray(erode).convert("L")
