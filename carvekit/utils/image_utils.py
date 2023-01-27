"""
    Source url: https://github.com/OPHoperHPO/image-background-remove-tool

    Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].

    License: Apache License 2.0
"""

import pathlib
from typing import Union, Any, Tuple

import PIL.Image
import numpy as np
import torch

ALLOWED_SUFFIXES = [".jpg", ".jpeg", ".bmp", ".png", ".webp"]


def to_tensor(x: Any) -> torch.Tensor:
    """
    Returns a PIL.Image.Image as torch tensor without swap tensor dims.

    Args:
        x (PIL.Image.Image): image

    Returns:
        torch.Tensor: image as torch tensor
    """
    return torch.tensor(np.array(x, copy=True))


def load_image(file: Union[str, pathlib.Path, PIL.Image.Image]) -> PIL.Image.Image:
    """Returns a `PIL.Image.Image` class by string path or `pathlib.Path` or `PIL.Image.Image` instance

    Args:
        file (Union[str, pathlib.Path, PIL.Image.Image]): File path or `PIL.Image.Image` instance

    Returns:
        PIL.Image.Image: image instance loaded from `file` location

    Raises:
        ValueError: If file not exists or file is directory or file isn't an image or file is not correct PIL Image

    """
    if isinstance(file, str) and is_image_valid(pathlib.Path(file)):
        return PIL.Image.open(file)
    elif isinstance(file, PIL.Image.Image):
        return file
    elif isinstance(file, pathlib.Path) and is_image_valid(file):
        return PIL.Image.open(str(file))
    else:
        raise ValueError("Unknown input file type")


def convert_image(image: PIL.Image.Image, mode="RGB") -> PIL.Image.Image:
    """Performs image conversion to correct color mode

    Args:
        image (PIL.Image.Image): `PIL.Image.Image` instance
        mode (str, default=RGB): Color mode to convert

    Returns:
        PIL.Image.Image: converted image

    Raises:
        ValueError: If image hasn't convertable color mode, or it is too small
    """
    if is_image_valid(image):
        return image.convert(mode)


def is_image_valid(image: Union[pathlib.Path, PIL.Image.Image]) -> bool:
    """This function performs image validation.

    Args:
        image (Union[pathlib.Path, PIL.Image.Image]): Path to the image or `PIL.Image.Image` instance being checked.

    Returns:
        bool: True if image is valid, False otherwise.

    Raises:
        ValueError: If file not a valid image path or image hasn't convertable color mode, or it is too small

    """
    if isinstance(image, pathlib.Path):
        if not image.exists():
            raise ValueError("File is not exists")
        elif image.is_dir():
            raise ValueError("File is a directory")
        elif image.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(
                f"Unsupported image format. Supported file formats: {', '.join(ALLOWED_SUFFIXES)}"
            )
    elif isinstance(image, PIL.Image.Image):
        if not (image.size[0] > 32 and image.size[1] > 32):
            raise ValueError("Image should be bigger then (32x32) pixels.")
        elif image.mode not in [
            "RGB",
            "RGBA",
            "L",
        ]:
            raise ValueError("Wrong image color mode.")
    else:
        raise ValueError("Unknown input file type")
    return True


def transparency_paste(
    bg_img: PIL.Image.Image, fg_img: PIL.Image.Image, box=(0, 0)
) -> PIL.Image.Image:
    """
    Inserts an image into another image while maintaining transparency.

    Args:
        bg_img (PIL.Image.Image): background image
        fg_img (PIL.Image.Image): foreground image
        box (tuple[int, int]): place to paste

    Returns:
        PIL.Image.Image: Background image with pasted foreground image at point or in the specified box
    """
    fg_img_trans = PIL.Image.new("RGBA", bg_img.size)
    fg_img_trans.paste(fg_img, box, mask=fg_img)
    new_img = PIL.Image.alpha_composite(bg_img, fg_img_trans)
    return new_img


def add_margin(
    pil_img: PIL.Image.Image,
    top: int,
    right: int,
    bottom: int,
    left: int,
    color: Tuple[int, int, int, int],
) -> PIL.Image.Image:
    """
    Adds margin to the image.

    Args:
        pil_img (PIL.Image.Image): Image that needed to add margin.
        top (int): pixels count at top side
        right (int): pixels count at right side
        bottom (int): pixels count at bottom side
        left (int): pixels count at left side
        color (Tuple[int, int, int, int]): color of margin

    Returns:
        PIL.Image.Image: Image with margin.
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    # noinspection PyTypeChecker
    result = PIL.Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result
