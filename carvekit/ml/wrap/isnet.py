"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import pathlib

import PIL.Image
import numpy as np
import torch
from PIL import Image
from typing import List, Union
from torchvision.transforms.functional import normalize

from carvekit.ml.arch.isnet.isnet import ISNetDIS
from carvekit.ml.files.models_loc import isnet_full_pretrained
from carvekit.utils.image_utils import load_image, convert_image
from carvekit.utils.models_utils import get_precision_autocast, cast_network
from carvekit.utils.pool_utils import thread_pool_processing, batch_generator

__all__ = ["ISNet"]


class ISNet(ISNetDIS):
    """ISNet model interface"""

    def __init__(
        self,
        device="cpu",
        input_image_size: Union[List[int], int] = 1024,
        batch_size: int = 1,
        load_pretrained: bool = True,
        fp16: bool = False,
    ):
        """
        Initialize the ISNet model

        Args:
            device: processing device
            input_image_size: input image size
            batch_size: the number of images that the neural network processes in one run
            load_pretrained: loading pretrained model
            fp16: use fp16 precision

        """
        super(ISNet, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16
        if isinstance(input_image_size, list):
            self.input_image_size = input_image_size[:2]
        else:
            self.input_image_size = (input_image_size, input_image_size)
        self.to(device)
        if load_pretrained:
            self.load_state_dict(
                torch.load(isnet_full_pretrained(), map_location=self.device)
            )

        self.eval()

    def data_preprocessing(self, data: PIL.Image.Image) -> torch.FloatTensor:
        """
        Transform input image to suitable data format for neural network

        Args:
            data: input image

        Returns:
            input for neural network

        """
        resized = data.resize(self.input_image_size, resample=3)
        # noinspection PyTypeChecker
        resized_arr = torch.from_numpy(np.array(resized, dtype=float)).permute(2, 0, 1)
        resized_arr = resized_arr.unsqueeze(0)
        resized_arr = resized_arr / 255.0
        resized_arr = normalize(resized_arr, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return resized_arr.type(torch.FloatTensor)

    @staticmethod
    def data_postprocessing(
        data: torch.tensor, original_image: PIL.Image.Image
    ) -> PIL.Image.Image:
        """
        Transforms output data from neural network to suitable data
        format for using with other components of this framework.

        Args:
            data: output data from neural network
            original_image: input image which was used for predicted data

        Returns:
            Segmentation mask as PIL Image instance

        """
        data = data.squeeze(0)
        ma = torch.max(data)
        mi = torch.min(data)
        data = (data - mi) / (ma - mi)
        mask = Image.fromarray(
            (data * 255).cpu().data.numpy().astype(np.uint8)
        ).convert("L")
        mask = mask.resize(original_image.size, resample=3)
        return mask

    def __call__(
        self, images: List[Union[str, pathlib.Path, PIL.Image.Image]]
    ) -> List[PIL.Image.Image]:
        """
        Passes input images though neural network and returns segmentation masks as PIL.Image.Image instances

        Args:
            images: input images

        Returns:
            segmentation masks as for input images, as PIL.Image.Image instances

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
                    batches = batches.to(self.device)
                    masks = super(ISNetDIS, self).__call__(batches)[0][0]
                    masks_cpu = masks.cpu()
                    del batches, masks
                masks = thread_pool_processing(
                    lambda x: self.data_postprocessing(
                        masks_cpu[x], converted_images[x]
                    ),
                    range(len(converted_images)),
                )
                collect_masks += masks
        return collect_masks
