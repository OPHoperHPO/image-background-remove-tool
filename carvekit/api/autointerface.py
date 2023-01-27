"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
from collections import Counter
from pathlib import Path

from PIL import Image
from typing import Union, List, Dict

from carvekit.api.interface import Interface
from carvekit.ml.wrap.basnet import BASNET
from carvekit.ml.wrap.cascadepsp import CascadePSP
from carvekit.ml.wrap.deeplab_v3 import DeepLabV3
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.scene_classifier import SceneClassifier
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.ml.wrap.u2net import U2NET
from carvekit.ml.wrap.yolov4 import SimplifiedYoloV4
from carvekit.pipelines.postprocessing import CasMattingMethod
from carvekit.trimap.generator import TrimapGenerator

__all__ = ["AutoInterface"]

from carvekit.utils.image_utils import load_image

from carvekit.utils.pool_utils import thread_pool_processing


class AutoInterface(Interface):
    def __init__(
        self,
        scene_classifier: SceneClassifier,
        object_classifier: SimplifiedYoloV4,
        segmentation_batch_size: int = 3,
        refining_batch_size: int = 1,
        refining_image_size: int = 900,
        postprocessing_batch_size: int = 1,
        postprocessing_image_size: int = 2048,
        segmentation_device: str = "cpu",
        postprocessing_device: str = "cpu",
        fp16=False,
    ):
        """
        Args:
            scene_classifier: SceneClassifier instance
            object_classifier: YoloV4_COCO instance
        """
        self.scene_classifier = scene_classifier
        self.object_classifier = object_classifier
        self.segmentation_batch_size = segmentation_batch_size
        self.refining_batch_size = refining_batch_size
        self.refining_image_size = refining_image_size
        self.postprocessing_batch_size = postprocessing_batch_size
        self.postprocessing_image_size = postprocessing_image_size
        self.segmentation_device = segmentation_device
        self.postprocessing_device = postprocessing_device
        self.fp16 = fp16
        super().__init__(
            seg_pipe=None, post_pipe=None, pre_pipe=None
        )  # just for compatibility with Interface class

    @staticmethod
    def select_params_for_net(net: Union[TracerUniversalB7, U2NET, DeepLabV3]):
        """
        Selects the parameters for the network depending on the scene

        Args:
            net: network
        """
        if net == TracerUniversalB7:
            return {"prob_threshold": 231, "kernel_size": 30, "erosion_iters": 5}
        elif net == U2NET:
            return {"prob_threshold": 231, "kernel_size": 30, "erosion_iters": 5}
        elif net == DeepLabV3:
            return {"prob_threshold": 231, "kernel_size": 40, "erosion_iters": 20}
        elif net == BASNET:
            return {"prob_threshold": 231, "kernel_size": 30, "erosion_iters": 5}
        else:
            raise ValueError("Unknown network type")

    def select_net(self, scene: str, images_info: List[dict]):
        # TODO: Update this function, when new networks will be added
        if scene == "hard":
            for image_info in images_info:
                objects = image_info["objects"]
                if len(objects) == 0:
                    image_info[
                        "net"
                    ] = TracerUniversalB7  # It seems that the image is empty, but we will try to process it
                    continue
                obj_counter: Dict = dict(Counter([obj for obj in objects]))
                # fill empty classes
                for _tag in self.object_classifier.db:
                    if _tag not in obj_counter:
                        obj_counter[_tag] = 0

                non_empty_classes = [obj for obj in obj_counter if obj_counter[obj] > 0]

                if obj_counter["human"] > 0 and len(non_empty_classes) == 1:
                    # Human only case. Hard Scene? It may be a photo of a person in far/middle distance.
                    image_info["net"] = TracerUniversalB7
                    # TODO: will use DeepLabV3+ for this image, it is more suitable for this case,
                    #  but needs checks for small bbox
                elif obj_counter["human"] > 0 and len(non_empty_classes) > 1:
                    # Okay, we have a human without extra hairs and something else. Hard border
                    image_info["net"] = TracerUniversalB7
                elif obj_counter["cars"] > 0:
                    # Cars case
                    image_info["net"] = TracerUniversalB7
                elif obj_counter["animals"] > 0:
                    # Animals case
                    image_info["net"] = U2NET  # animals should be always in soft scenes
                else:
                    # We have no idea what is in the image, so we will try to process it with universal model
                    image_info["net"] = TracerUniversalB7

        elif scene == "soft":
            for image_info in images_info:
                objects = image_info["objects"]
                if len(objects) == 0:
                    image_info[
                        "net"
                    ] = TracerUniversalB7  # It seems that the image is empty, but we will try to process it
                    continue
                obj_counter: Dict = dict(Counter([obj for obj in objects]))
                # fill empty classes
                for _tag in self.object_classifier.db:
                    if _tag not in obj_counter:
                        obj_counter[_tag] = 0

                non_empty_classes = [obj for obj in obj_counter if obj_counter[obj] > 0]

                if obj_counter["human"] > 0 and len(non_empty_classes) == 1:
                    # Human only case. It may be a portrait
                    image_info["net"] = U2NET
                elif obj_counter["human"] > 0 and len(non_empty_classes) > 1:
                    # Okay, we have a human with hairs and something else
                    image_info["net"] = U2NET
                elif obj_counter["cars"] > 0:
                    # Cars case.
                    image_info["net"] = TracerUniversalB7
                elif obj_counter["animals"] > 0:
                    # Animals case
                    image_info["net"] = U2NET  # animals should be always in soft scenes
                else:
                    # We have no idea what is in the image, so we will try to process it with universal model
                    image_info["net"] = TracerUniversalB7
        elif scene == "digital":
            for image_info in images_info:  # TODO: not implemented yet
                image_info[
                    "net"
                ] = TracerUniversalB7  # It seems that the image is empty, but we will try to process it

    def __call__(self, images: List[Union[str, Path, Image.Image]]):
        """
        Automatically detects the scene and selects the appropriate network for segmentation

        Args:
            interface: Interface instance
            images: list of images

        Returns:
            list of masks
        """
        loaded_images = thread_pool_processing(load_image, images)

        scene_analysis = self.scene_classifier(loaded_images)
        images_objects = self.object_classifier(loaded_images)

        images_per_scene = {}
        for i, image in enumerate(loaded_images):
            scene_name = scene_analysis[i][0][0]
            if scene_name not in images_per_scene:
                images_per_scene[scene_name] = []
            images_per_scene[scene_name].append(
                {"image": image, "objects": images_objects[i]}
            )

        for scene_name, images_info in list(images_per_scene.items()):
            self.select_net(scene_name, images_info)

        # groups images by net
        for scene_name, images_info in list(images_per_scene.items()):
            groups = {}
            for image_info in images_info:
                net = image_info["net"]
                if net not in groups:
                    groups[net] = []
                groups[net].append(image_info)
            for net, gimages_info in list(groups.items()):
                sc_images = [image_info["image"] for image_info in gimages_info]
                masks = net(
                    device=self.segmentation_device,
                    batch_size=self.segmentation_batch_size,
                    fp16=self.fp16,
                )(sc_images)

                for i, image_info in enumerate(gimages_info):
                    image_info["mask"] = masks[i]

        cascadepsp = CascadePSP(
            device=self.postprocessing_device,
            fp16=self.fp16,
            input_tensor_size=self.refining_image_size,
            batch_size=self.refining_batch_size,
        )

        fba = FBAMatting(
            device=self.postprocessing_device,
            batch_size=self.postprocessing_batch_size,
            input_tensor_size=self.postprocessing_image_size,
            fp16=self.fp16,
        )
        # groups images by net
        for scene_name, images_info in list(images_per_scene.items()):
            groups = {}
            for image_info in images_info:
                net = image_info["net"]
                if net not in groups:
                    groups[net] = []
                groups[net].append(image_info)
            for net, gimages_info in list(groups.items()):
                sc_images = [image_info["image"] for image_info in gimages_info]
                # noinspection PyArgumentList
                trimap_generator = TrimapGenerator(**self.select_params_for_net(net))
                matting_method = CasMattingMethod(
                    refining_module=cascadepsp,
                    matting_module=fba,
                    trimap_generator=trimap_generator,
                    device=self.postprocessing_device,
                )
                masks = [image_info["mask"] for image_info in gimages_info]
                result = matting_method(sc_images, masks)

                for i, image_info in enumerate(gimages_info):
                    image_info["result"] = result[i]

        # Reconstructing the original order of image
        result = []
        for image in loaded_images:
            for scene_name, images_info in list(images_per_scene.items()):
                for image_info in images_info:
                    if image_info["image"] == image:
                        result.append(image_info["result"])
                        break
        if len(result) != len(images):
            raise RuntimeError(
                "Something went wrong with restoring original order. Please report this bug."
            )
        return result
