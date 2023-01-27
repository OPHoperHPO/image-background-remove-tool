"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
from pathlib import Path

from PIL import Image
from typing import Union, List

from carvekit.ml.wrap.scene_classifier import SceneClassifier
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.ml.wrap.u2net import U2NET

__all__ = ["AutoScene"]


class AutoScene:
    """AutoScene preprocessing method"""

    def __init__(self, scene_classifier: SceneClassifier):
        """
        Args:
            scene_classifier: SceneClassifier instance
        """
        self.scene_classifier = scene_classifier

    @staticmethod
    def select_net(scene: str):
        """
        Selects the network to be used for segmentation based on the detected scene

        Args:
            scene: scene name
        """
        if scene == "hard":
            return TracerUniversalB7
        elif scene == "soft":
            return U2NET
        elif scene == "digital":
            return TracerUniversalB7  # TODO: not implemented yet

    def __call__(self, interface, images: List[Union[str, Path, Image.Image]]):
        """
        Automatically detects the scene and selects the appropriate network for segmentation

        Args:
            interface: Interface instance
            images: list of images

        Returns:
            list of masks
        """
        scene_analysis = self.scene_classifier(images)
        images_per_scene = {}
        for i, image in enumerate(images):
            scene_name = scene_analysis[i][0][0]
            if scene_name not in images_per_scene:
                images_per_scene[scene_name] = []
            images_per_scene[scene_name].append(image)

        masks_per_scene = {}
        for scene_name, igs in list(images_per_scene.items()):
            net = self.select_net(scene_name)
            if isinstance(interface.segmentation_pipeline, net):
                masks_per_scene[scene_name] = interface.segmentation_pipeline(igs)
            else:
                old_device = interface.segmentation_pipeline.device
                interface.segmentation_pipeline.to(
                    "cpu"
                )  # unload model from gpu, to avoid OOM
                net_instance = net(device=old_device)
                masks_per_scene[scene_name] = net_instance(igs)
                del net_instance
                interface.segmentation_pipeline.to(old_device)  # load model back to gpu

        # restore one list of masks with the same order as images
        masks = []
        for i, image in enumerate(images):
            scene_name = scene_analysis[i][0][0]
            masks.append(
                masks_per_scene[scene_name][images_per_scene[scene_name].index(image)]
            )

        return masks
