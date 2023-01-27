"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool

Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].

License: Apache License 2.0
"""
import PIL.Image
import torch
from carvekit.utils.image_utils import to_tensor


def composite(
    foreground: PIL.Image.Image,
    background: PIL.Image.Image,
    alpha: PIL.Image.Image,
    device="cpu",
):
    """
    Composites foreground with background by following
    https://pymatting.github.io/intro.html#alpha-matting math formula.

    Args:
        foreground (PIL.Image.Image): Image that will be pasted to background image with following alpha mask.
        background (PIL.Image.Image): Background image
        alpha (PIL.Image.Image): Alpha Image
        device (Literal[cpu, cuda]): Processing device

    Returns:
        PIL.Image.Image: Composited image.
    """

    foreground = foreground.convert("RGBA")
    background = background.convert("RGBA")
    alpha_rgba = alpha.convert("RGBA")
    alpha_l = alpha.convert("L")

    fg = to_tensor(foreground).to(device)
    alpha_rgba = to_tensor(alpha_rgba).to(device)
    alpha_l = to_tensor(alpha_l).to(device)
    bg = to_tensor(background).to(device)

    alpha_l = alpha_l / 255
    alpha_rgba = alpha_rgba / 255

    bg = torch.where(torch.logical_not(alpha_rgba >= 1), bg, fg)
    bg[:, :, 0] = alpha_l[:, :] * fg[:, :, 0] + (1 - alpha_l[:, :]) * bg[:, :, 0]
    bg[:, :, 1] = alpha_l[:, :] * fg[:, :, 1] + (1 - alpha_l[:, :]) * bg[:, :, 1]
    bg[:, :, 2] = alpha_l[:, :] * fg[:, :, 2] + (1 - alpha_l[:, :]) * bg[:, :, 2]
    bg[:, :, 3] = alpha_l[:, :] * 255

    del alpha_l, alpha_rgba, fg
    return PIL.Image.fromarray(bg.cpu().numpy()).convert("RGBA")


def apply_mask(
    image: PIL.Image.Image, mask: PIL.Image.Image, device="cpu"
) -> PIL.Image.Image:
    """
    Applies mask to foreground.

    Args:
        image (PIL.Image.Image): Image with background.
        mask (PIL.Image.Image): Alpha Channel mask for this image.
        device (Literal[cpu, cuda]): Processing device.

    Returns:
        PIL.Image.Image: Image without background, where mask was black.
    """
    background = PIL.Image.new("RGBA", image.size, color=(130, 130, 130, 0))
    return composite(image, background, mask, device=device).convert("RGBA")


def extract_alpha_channel(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Extracts alpha channel from the RGBA image.

    Args:
        image: RGBA PIL image

    Returns:
        PIL.Image.Image: RGBA alpha channel image
    """
    alpha = image.split()[-1]
    bg = PIL.Image.new("RGBA", image.size, (0, 0, 0, 255))
    bg.paste(alpha, mask=alpha)
    return bg.convert("RGBA")
