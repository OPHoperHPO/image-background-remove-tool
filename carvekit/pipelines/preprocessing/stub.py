"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
from pathlib import Path
from typing import Union, List

from PIL import Image

__all__ = ["PreprocessingStub"]


class PreprocessingStub:
    """Stub for future preprocessing methods"""

    def __call__(self, interface, images: List[Union[str, Path, Image.Image]]):
        """
        Passes data though interface.segmentation_pipeline() method

        Args:
            interface: Interface instance
            images: list of images

        Returns:
            the result of passing data through segmentation_pipeline method of interface
        """
        return interface.segmentation_pipeline(images=images)
