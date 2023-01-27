"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import PIL.Image
import cv2
import numpy as np


class CV2TrimapGenerator:
    def __init__(self, kernel_size: int = 30, erosion_iters: int = 1):
        """
        Initialize a new CV2TrimapGenerator instance

        Args:
            kernel_size (int, default=30): The size of the offset from the object mask
            in pixels when an unknown area is detected in the trimap
            erosion_iters (int, default=1: The number of iterations of erosion that
            the object's mask will be subjected to before forming an unknown area
        """
        self.kernel_size = kernel_size
        self.erosion_iters = erosion_iters

    def __call__(
        self, original_image: PIL.Image.Image, mask: PIL.Image.Image
    ) -> PIL.Image.Image:
        """
        Generates trimap based on predicted object mask to refine object mask borders.
        Based on cv2 erosion algorithm.

        Args:
            original_image (PIL.Image.Image): Original image
            mask (PIL.Image.Image): Predicted object mask

        Returns:
            PIL.Image.Image: Generated trimap for image.
        """
        if mask.mode != "L":
            raise ValueError("Input mask has wrong color mode.")
        if mask.size != original_image.size:
            raise ValueError("Sizes of input image and predicted mask doesn't equal")
        # noinspection PyTypeChecker
        mask_array = np.array(mask)
        pixels = 2 * self.kernel_size + 1
        kernel = np.ones((pixels, pixels), np.uint8)

        if self.erosion_iters > 0:
            erosion_kernel = np.ones((3, 3), np.uint8)
            erode = cv2.erode(mask_array, erosion_kernel, iterations=self.erosion_iters)
            erode = np.where(erode == 0, 0, mask_array)
        else:
            erode = mask_array.copy()

        dilation = cv2.dilate(erode, kernel, iterations=1)

        dilation = np.where(dilation == 255, 127, dilation)  # WHITE to GRAY
        trimap = np.where(erode > 127, 200, dilation)  # mark the tumor inside GRAY

        trimap = np.where(trimap < 127, 0, trimap)  # Embelishment
        trimap = np.where(trimap > 200, 0, trimap)  # Embelishment
        trimap = np.where(trimap == 200, 255, trimap)  # GRAY to WHITE

        return PIL.Image.fromarray(trimap).convert("L")
