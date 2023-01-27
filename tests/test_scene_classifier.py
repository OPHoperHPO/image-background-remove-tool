"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

import torch

from carvekit.ml.wrap.scene_classifier import SceneClassifier


def test_init():
    SceneClassifier()


def test_preprocessing(scene_classifier_model, converted_pil_image, black_image_pil):
    scene_classifier_model = scene_classifier_model(False)
    assert (
        isinstance(
            scene_classifier_model.data_preprocessing(converted_pil_image),
            torch.FloatTensor,
        )
        is True
    )
    assert (
        isinstance(
            scene_classifier_model.data_preprocessing(black_image_pil),
            torch.FloatTensor,
        )
        is True
    )


def test_inf(
    scene_classifier_model,
    converted_pil_image,
    image_pil,
    image_str,
    image_path,
    black_image_pil,
):
    scene_classifier_model = scene_classifier_model(False)
    calc_result = scene_classifier_model(
        [
            converted_pil_image,
            black_image_pil,
            image_pil,
            image_str,
            image_path,
            black_image_pil,
        ]
    )
    assert calc_result[0][0][0] == "soft"
    assert calc_result[1][0][0] == "hard"


def test_seg_with_fp16(
    scene_classifier_model, image_pil, image_str, image_path, black_image_pil
):
    scene_classifier_model = scene_classifier_model(True)
    scene_classifier_model([image_pil, image_str, image_path, black_image_pil])
