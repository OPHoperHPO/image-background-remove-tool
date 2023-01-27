"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

import pathlib

import PIL.Image
import PIL.Image
import numpy as np
import pydantic
import torch
from torch.autograd import Variable
from typing import List, Union

from carvekit.ml.arch.yolov4.models import Yolov4
from carvekit.ml.arch.yolov4.utils import post_processing
from carvekit.ml.files.models_loc import yolov4_coco_pretrained
from carvekit.utils.image_utils import load_image, convert_image
from carvekit.utils.models_utils import get_precision_autocast, cast_network
from carvekit.utils.pool_utils import thread_pool_processing, batch_generator

__all__ = ["YoloV4_COCO", "SimplifiedYoloV4"]


class Object(pydantic.BaseModel):
    """Object class"""

    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class YoloV4_COCO(Yolov4):
    """YoloV4 COCO model wrapper"""

    def __init__(
        self,
        n_classes: int = 80,
        device="cpu",
        classes: List[str] = None,
        input_image_size: Union[List[int], int] = 608,
        batch_size: int = 4,
        load_pretrained: bool = True,
        fp16: bool = False,
        model_path: Union[str, pathlib.Path] = None,
    ):
        """
        Initialize the YoloV4 COCO.

        Args:
            n_classes: number of classes
            device: processing device
            input_image_size: input image size
            batch_size: the number of images that the neural network processes in one run
            fp16: use fp16 precision
            model_path: path to model weights
            load_pretrained: load pretrained weights
        """
        if model_path is None:
            model_path = yolov4_coco_pretrained()
        self.fp16 = fp16
        self.device = device
        self.batch_size = batch_size
        if isinstance(input_image_size, list):
            self.input_image_size = input_image_size[:2]
        else:
            self.input_image_size = (input_image_size, input_image_size)

        if load_pretrained:
            state_dict = torch.load(model_path, map_location="cpu")
            self.classes = state_dict["classes"]
            super().__init__(n_classes=len(state_dict["classes"]), inference=True)
            self.load_state_dict(state_dict["state"])
        else:
            self.classes = classes
            super().__init__(n_classes=n_classes, inference=True)

        self.to(device)
        self.eval()

    def data_preprocessing(self, data: PIL.Image.Image) -> torch.FloatTensor:
        """
        Transform input image to suitable data format for neural network

        Args:
            data: input image

        Returns:
            input for neural network

        """
        image = data.resize(self.input_image_size)
        # noinspection PyTypeChecker
        image = np.array(image).astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = image / 255.0
        image = torch.from_numpy(image).float()
        return torch.unsqueeze(image, 0).type(torch.FloatTensor)

    def data_postprocessing(
        self, data: List[torch.FloatTensor], images: List[PIL.Image.Image]
    ) -> List[Object]:
        """
        Transforms output data from neural network to suitable data
        format for using with other components of this framework.

        Args:
            data: output data from neural network
            images: input images


        Returns:
            list of objects for each image

        """
        output = post_processing(0.4, 0.6, data)
        images_objects = []
        for image_idx, image_objects in enumerate(output):
            image_size = images[image_idx].size
            objects = []
            for obj in image_objects:
                objects.append(
                    Object(
                        class_name=self.classes[obj[6]],
                        confidence=obj[5],
                        x1=int(obj[0] * image_size[0]),
                        y1=int(obj[1] * image_size[1]),
                        x2=int(obj[2] * image_size[0]),
                        y2=int(obj[3] * image_size[1]),
                    )
                )
            images_objects.append(objects)

        return images_objects

    def __call__(
        self, images: List[Union[str, pathlib.Path, PIL.Image.Image]]
    ) -> List[List[Object]]:
        """
        Passes input images though neural network

        Args:
            images: input images

        Returns:
            list of objects for each image

        """
        collect_masks = []
        autocast, dtype = get_precision_autocast(device=self.device, fp16=self.fp16)
        with autocast:
            cast_network(self, dtype)
            for image_batch in batch_generator(images, self.batch_size):
                converted_images = thread_pool_processing(
                    lambda x: convert_image(load_image(x)), image_batch
                )
                batches = torch.vstack(
                    thread_pool_processing(self.data_preprocessing, converted_images)
                )
                with torch.no_grad():
                    batches = Variable(batches).to(self.device)
                    out = super().__call__(batches)
                    out_cpu = [out_i.cpu() for out_i in out]
                    del batches, out
                out = self.data_postprocessing(out_cpu, converted_images)
                collect_masks += out

        return collect_masks


class SimplifiedYoloV4(YoloV4_COCO):
    """
    The YoloV4 COCO classifier, but classifies only 7 supercategories.

    human - Scenes of people, such as portrait photographs
    animals - Scenes with animals
    objects - Scenes with normal objects
    cars - Scenes with cars
    other - Other scenes
    """

    db = {
        "human": ["person"],
        "animals": [
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
        ],
        "cars": [
            "car",
            "motorbike",
            "bus",
            "truck",
        ],
        "objects": [
            "bicycle",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ],
        "other": ["aeroplane", "train", "boat"],
    }

    def data_postprocessing(
        self, data: List[torch.FloatTensor], images: List[PIL.Image.Image]
    ) -> List[List[str]]:
        """
        Transforms output data from neural network to suitable data
        format for using with other components of this framework.

        Args:
            data: output data from neural network
            images: input images
        """
        objects = super().data_postprocessing(data, images)
        new_output = []

        for image_objects in objects:
            new_objects = []
            for obj in image_objects:
                for key, values in list(self.db.items()):
                    if obj.class_name in values:
                        new_objects.append(key)  # We don't need bbox at this moment
            new_output.append(new_objects)

        return new_output
