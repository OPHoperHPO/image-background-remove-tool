"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
from carvekit.ml.wrap.fba_matting import FBAMatting
from typing import Union, List
from PIL import Image
from pathlib import Path
from carvekit.trimap.cv_gen import CV2TrimapGenerator
from carvekit.trimap.generator import TrimapGenerator
from carvekit.utils.mask_utils import apply_mask
from carvekit.utils.pool_utils import thread_pool_processing
from carvekit.utils.image_utils import load_image, convert_image

__all__ = ["MattingMethod"]


class MattingMethod:
    """
    Improving the edges of the object mask using neural networks for matting and algorithms for creating trimap.
    Neural network for matting performs accurate object edge detection by using a special map called trimap,
    with unknown area that we scan for boundary, already known general object area and the background."""

    def __init__(
        self,
        matting_module: Union[FBAMatting],
        trimap_generator: Union[TrimapGenerator, CV2TrimapGenerator],
        device="cpu",
    ):
        """
        Initializes Matting Method class.

        Args:
            matting_module: Initialized matting neural network class
            trimap_generator: Initialized trimap generator class
            device: Processing device used for applying mask to image
        """
        self.device = device
        self.matting_module = matting_module
        self.trimap_generator = trimap_generator

    def __call__(
        self,
        images: List[Union[str, Path, Image.Image]],
        masks: List[Union[str, Path, Image.Image]],
    ):
        """
        Passes data through apply_mask function

        Args:
            images: list of images
            masks: list pf masks

        Returns:
            list of images
        """
        if len(images) != len(masks):
            raise ValueError("Images and Masks lists should have same length!")
        images = thread_pool_processing(lambda x: convert_image(load_image(x)), images)
        masks = thread_pool_processing(
            lambda x: convert_image(load_image(x), mode="L"), masks
        )
        trimaps = thread_pool_processing(
            lambda x: self.trimap_generator(original_image=images[x], mask=masks[x]),
            range(len(images)),
        )
        alpha = self.matting_module(images=images, trimaps=trimaps)
        return list(
            map(
                lambda x: apply_mask(image=images[x], mask=alpha[x]),
                range(len(images)),
            )
        )
