"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool

Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].

License: Apache License 2.0
"""
import pathlib
from typing import List, Union

import PIL.Image
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from carvekit.ml.files.models_loc import deeplab_pretrained
from carvekit.utils.image_utils import convert_image, load_image
from carvekit.utils.models_utils import get_precision_autocast, cast_network
from carvekit.utils.pool_utils import batch_generator, thread_pool_processing

__all__ = ["DeepLabV3"]


class DeepLabV3:
    def __init__(
        self,
        device="cpu",
        batch_size: int = 10,
        input_image_size: Union[List[int], int] = 1024,
        load_pretrained: bool = True,
        fp16: bool = False,
    ):
        """
        Initialize the `DeepLabV3` model

        Args:
            device (Literal[cpu, cuda], default=cpu): processing device
            input_image_size (): input image size
            batch_size (int, default=10): the number of images that the neural network processes in one run
            load_pretrained (bool, default=True): loading pretrained model
            fp16 (bool, default=False): use half precision

        """
        self.device = device
        self.batch_size = batch_size
        self.network = deeplabv3_resnet101(
            pretrained=False, pretrained_backbone=False, aux_loss=True
        )
        self.network.to(self.device)
        if load_pretrained:
            self.network.load_state_dict(
                torch.load(deeplab_pretrained(), map_location=self.device)
            )
        if isinstance(input_image_size, list):
            self.input_image_size = input_image_size[:2]
        else:
            self.input_image_size = (input_image_size, input_image_size)
        self.network.eval()
        self.fp16 = fp16
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def to(self, device: str):
        """
        Moves neural network to specified processing device

        Args:
            device (Literal[cpu, cuda]): the desired device.

        """
        self.network.to(device)

    def data_preprocessing(self, data: PIL.Image.Image) -> torch.Tensor:
        """
        Transform input image to suitable data format for neural network

        Args:
            data (PIL.Image.Image): input image

        Returns:
            torch.Tensor: input for neural network

        """
        copy = data.copy()
        copy.thumbnail(self.input_image_size, resample=3)
        return self.transform(copy)

    @staticmethod
    def data_postprocessing(
        data: torch.Tensor, original_image: PIL.Image.Image
    ) -> PIL.Image.Image:
        """
        Transforms output data from neural network to suitable data
        format for using with other components of this framework.

        Args:
            data (torch.Tensor): output data from neural network
            original_image (PIL.Image.Image): input image which was used for predicted data

        Returns:
            PIL.Image.Image: Segmentation mask as `PIL Image` instance

        """
        return (
            Image.fromarray(data.numpy() * 255).convert("L").resize(original_image.size)
        )

    def __call__(
        self, images: List[Union[str, pathlib.Path, PIL.Image.Image]]
    ) -> List[PIL.Image.Image]:
        """
        Passes input images though neural network and returns segmentation masks as `PIL.Image.Image` instances

        Args:
            images (List[Union[str, pathlib.Path, PIL.Image.Image]]): input images

        Returns:
            List[PIL.Image.Image]: segmentation masks as for input images, as `PIL.Image.Image` instances

        """
        collect_masks = []
        autocast, dtype = get_precision_autocast(device=self.device, fp16=self.fp16)
        with autocast:
            cast_network(self.network, dtype)
            for image_batch in batch_generator(images, self.batch_size):
                converted_images = thread_pool_processing(
                    lambda x: convert_image(load_image(x)), image_batch
                )
                batches = thread_pool_processing(
                    self.data_preprocessing, converted_images
                )
                with torch.no_grad():
                    masks = [
                        self.network(i.to(self.device).unsqueeze(0))["out"][0]
                        .argmax(0)
                        .byte()
                        .cpu()
                        for i in batches
                    ]
                    del batches
                masks = thread_pool_processing(
                    lambda x: self.data_postprocessing(masks[x], converted_images[x]),
                    range(len(converted_images)),
                )
                collect_masks += masks
        return collect_masks
