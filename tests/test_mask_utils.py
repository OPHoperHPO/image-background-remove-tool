"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import PIL.Image

from carvekit.utils.mask_utils import apply_mask, extract_alpha_channel


def test_apply_mask():
    assert (
        isinstance(
            apply_mask(
                image=PIL.Image.new("RGB", (512, 512)),
                mask=PIL.Image.new("RGB", (512, 512)),
            ),
            PIL.Image.Image,
        )
        is True
    )


def test_extract_alpha_channel():
    assert (
        isinstance(
            extract_alpha_channel(PIL.Image.new("RGB", (512, 512))), PIL.Image.Image
        )
        is True
    )
