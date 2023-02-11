"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

import torch

from carvekit.ml.wrap.yolov4 import SimplifiedYoloV4


def test_init():
    SimplifiedYoloV4()


def test_preprocessing(yoloV4, converted_pil_image, black_image_pil):
    yoloV4 = yoloV4(False)
    assert (
        isinstance(
            yoloV4.data_preprocessing(converted_pil_image),
            torch.FloatTensor,
        )
        is True
    )
    assert (
        isinstance(
            yoloV4.data_preprocessing(black_image_pil),
            torch.FloatTensor,
        )
        is True
    )


def test_inf(
    yoloV4,
    converted_pil_image,
    image_pil,
    image_str,
    image_path,
    black_image_pil,
):
    yoloV4 = yoloV4(False)
    calc_result = yoloV4(
        [
            image_pil,
            image_str,
            image_path,
            black_image_pil,
        ]
    )
    assert calc_result[0][0] == "animals"
    assert calc_result[1][0] == "animals"
    assert calc_result[2][0] == "animals"
    assert len(calc_result[3]) == 0


def test_seg_with_fp16(yoloV4, image_pil, image_str, image_path, black_image_pil):
    yoloV4 = yoloV4(True)
    yoloV4([image_pil, image_str, image_path, black_image_pil])
