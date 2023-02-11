"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

import pytest
import torch
from PIL import Image

from carvekit.ml.wrap.cascadepsp import CascadePSP
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.yolov4 import SimplifiedYoloV4


def test_init():
    CascadePSP(load_pretrained=True)
    CascadePSP(load_pretrained=False)


def test_preprocessing(cascadepsp, converted_pil_image, black_image_pil, image_mask):
    cascadepsp_ = cascadepsp(fp16=False)
    assert (
        isinstance(
            cascadepsp_.data_preprocessing(converted_pil_image)[0], torch.FloatTensor
        )
        is True
    )
    assert (
        isinstance(
            cascadepsp_.data_preprocessing(black_image_pil)[0], torch.FloatTensor
        )
        is True
    )
    assert (
        isinstance(cascadepsp_.data_preprocessing(image_mask)[0], torch.FloatTensor)
        is True
    )
    cascadepsp_ = cascadepsp(fp16=False, device_="cpu", batch_size=2)
    assert (
        isinstance(
            cascadepsp_.data_preprocessing(converted_pil_image)[0], torch.FloatTensor
        )
        is True
    )
    assert (
        isinstance(
            cascadepsp_.data_preprocessing(black_image_pil)[0], torch.FloatTensor
        )
        is True
    )
    assert (
        isinstance(cascadepsp_.data_preprocessing(image_mask)[0], torch.FloatTensor)
        is True
    )


def test_postprocessing(cascadepsp, converted_pil_image, black_image_pil):
    cascadepsp = cascadepsp(False)
    assert isinstance(
        cascadepsp.data_postprocessing(
            torch.ones((7, 320, 320), dtype=torch.float64), black_image_pil.convert("L")
        ),
        Image.Image,
    )


def test_seg(cascadepsp, image_pil, image_str, image_path, black_image_pil, image_mask):
    cascadepsp = cascadepsp(False)
    cascadepsp([image_pil], [image_mask])
    cascadepsp([image_pil, image_str, image_path], [image_mask, image_mask, image_mask])
    cascadepsp(
        [Image.new("RGB", (512, 512)), Image.new("RGB", (512, 512))],
        [Image.new("L", (512, 512)), Image.new("L", (512, 512))],
    )
    with pytest.raises(ValueError):
        cascadepsp([image_pil], [image_mask, image_mask])


def test_seg_with_fp12(
    cascadepsp, image_pil, image_str, image_path, black_image_pil, image_mask
):
    cascadepsp = cascadepsp(True)
    cascadepsp([image_pil], [image_mask])
    cascadepsp([image_pil, image_str, image_path], [image_mask, image_mask, image_mask])
    cascadepsp(
        [Image.new("RGB", (512, 512)), Image.new("RGB", (512, 512))],
        [Image.new("L", (512, 512)), Image.new("L", (512, 512))],
    )
    with pytest.raises(ValueError):
        cascadepsp([image_pil], [image_mask, image_mask])
