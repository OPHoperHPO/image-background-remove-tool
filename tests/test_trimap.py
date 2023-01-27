"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool

Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].

License: Apache License 2.0
"""
import PIL.Image
import pytest

from carvekit.trimap.add_ops import prob_as_unknown_area


def test_trimap_generator(trimap_instance, image_mask, image_pil):
    te = trimap_instance()
    assert isinstance(te(image_pil, image_mask), PIL.Image.Image)
    assert isinstance(
        te(PIL.Image.new("RGB", (512, 512)), PIL.Image.new("L", (512, 512))),
        PIL.Image.Image,
    )
    assert isinstance(
        te(
            PIL.Image.new("RGB", (512, 512), color=(255, 255, 255)),
            PIL.Image.new("L", (512, 512), color=255),
        ),
        PIL.Image.Image,
    )
    with pytest.raises(ValueError):
        te(PIL.Image.new("RGB", (512, 512)), PIL.Image.new("RGB", (512, 512)))
    with pytest.raises(ValueError):
        te(PIL.Image.new("RGB", (512, 512)), PIL.Image.new("RGB", (512, 512)))


def test_cv2_generator(cv2_trimap_instance, image_pil, image_mask):
    cv2trimapgen = cv2_trimap_instance()
    assert isinstance(cv2trimapgen(image_pil, image_mask), PIL.Image.Image)
    with pytest.raises(ValueError):
        cv2trimapgen(PIL.Image.new("RGB", (512, 512)), PIL.Image.new("RGB", (512, 512)))
    with pytest.raises(ValueError):
        cv2trimapgen(PIL.Image.new("L", (256, 256)), PIL.Image.new("L", (512, 512)))


def test_prob_as_unknown_area(image_pil, image_mask):
    with pytest.raises(ValueError):
        prob_as_unknown_area(image_pil, image_mask)
