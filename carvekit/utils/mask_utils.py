"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import PIL.Image


def apply_mask(
    image: PIL.Image.Image,
    mask: PIL.Image.Image,
) -> PIL.Image.Image:
    """
    Applies mask to foreground.

    Args:
        device: Processing device.
        image: Image with background.
        mask: Alpha Channel mask for this image.

    Returns:
        Image without background, where mask was black.
    """
    background = PIL.Image.new("RGBA", image.size, color=(130, 130, 130, 0))
    return PIL.Image.composite(
        image.convert("RGBA"), background.convert("RGBA"), mask.convert("L")
    ).convert("RGBA")


def extract_alpha_channel(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Extracts alpha channel from the RGBA image.

    Args:
        image: RGBA PIL image

    Returns:
        RGBA alpha channel image
    """
    alpha = image.split()[-1]
    bg = PIL.Image.new("RGBA", image.size, (0, 0, 0, 255))
    bg.paste(alpha, mask=alpha)
    return bg.convert("RGBA")
