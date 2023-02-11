"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import warnings

import torch

from carvekit.api.autointerface import AutoInterface
from carvekit.api.interface import Interface


def test_init(scene_classifier_model, yoloV4):
    scene_classifier = scene_classifier_model(False)
    object_classifier = yoloV4(False)
    devices = ["cpu", "cuda"]

    for device in devices:
        if device == "cuda" and torch.cuda.is_available() is False:
            warnings.warn("Cuda GPU is not available! Testing on cuda skipped!")
            continue
        inf = AutoInterface(scene_classifier, object_classifier)
        del inf


def test_seg(image_pil, image_str, image_path, scene_classifier_model, yoloV4):
    scene_classifier = scene_classifier_model(False)
    scene_classifier.model.to("cpu")
    scene_classifier.device = "cpu"
    object_classifier = yoloV4(False)
    object_classifier.to("cpu")
    object_classifier.device = "cpu"

    interface = AutoInterface(
        scene_classifier,
        object_classifier,
        segmentation_device="cuda" if torch.cuda.is_available() else "cpu",
        postprocessing_device="cuda" if torch.cuda.is_available() else "cpu",
        fp16=True,
    )
    interface([image_pil, image_str, image_path])
