"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
from PIL import Image
from carvekit.trimap.cv_gen import CV2TrimapGenerator
from carvekit.trimap.add_ops import prob_filter, prob_as_unknown_area, post_erosion


class TrimapGenerator(CV2TrimapGenerator):
    def __init__(
        self, prob_threshold: int = 231, kernel_size: int = 30, erosion_iters: int = 5
    ):
        """
        Initialize a TrimapGenerator instance

        Args:
            prob_threshold (int, default=231): Probability threshold at which the
            prob_filter and prob_as_unknown_area operations will be applied
            kernel_size (int, default=30): The size of the offset from the object mask
            in pixels when an unknown area is detected in the trimap
            erosion_iters (int, default=5): The number of iterations of erosion that
            the object's mask will be subjected to before forming an unknown area
        """
        super().__init__(kernel_size, erosion_iters=0)
        self.prob_threshold = prob_threshold
        self.__erosion_iters = erosion_iters

    def __call__(self, original_image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Generates trimap based on predicted object mask to refine object mask borders.
        Based on cv2 erosion algorithm and additional prob. filters.

        Args:
            original_image (Image.Image): Original image
            mask (Image.Image): Predicted object mask

        Returns:
            Image.Image: Generated trimap for image.
        """
        filter_mask = prob_filter(mask=mask, prob_threshold=self.prob_threshold)
        trimap = super(TrimapGenerator, self).__call__(original_image, filter_mask)
        new_trimap = prob_as_unknown_area(
            trimap=trimap, mask=mask, prob_threshold=self.prob_threshold
        )
        new_trimap = post_erosion(new_trimap, self.__erosion_iters)
        return new_trimap
