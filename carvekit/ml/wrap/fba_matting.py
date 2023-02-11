"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import pathlib
from typing import Union, List, Tuple

import PIL
import cv2
import numpy as np
import torch
from PIL import Image

from carvekit.ml.arch.fba_matting.models import FBA
from carvekit.ml.arch.fba_matting.transforms import (
    trimap_transform,
    groupnorm_normalise_image,
)
from carvekit.ml.files.models_loc import fba_pretrained
from carvekit.utils.image_utils import convert_image, load_image
from carvekit.utils.models_utils import get_precision_autocast, cast_network
from carvekit.utils.pool_utils import batch_generator, thread_pool_processing

__all__ = ["FBAMatting"]


class FBAMatting(FBA):
    """
    FBA Matting Neural Network to improve edges on image.
    """

    def __init__(
        self,
        device="cpu",
        input_tensor_size: Union[List[int], int] = 2048,  # 1500,
        batch_size: int = 2,
        encoder="resnet50_GN_WS",
        load_pretrained: bool = True,
        fp16: bool = False,
        disable_noise_filter=False,
    ):
        """
        Initialize the FBAMatting model

        Args:
            device: processing device
            input_tensor_size: input image size
            batch_size: the number of images that the neural network processes in one run
            encoder: neural network encoder head
            load_pretrained: loading pretrained model
            fp16: use half precision
            disable_noise_filter: disables noise filter

        """
        super(FBAMatting, self).__init__(encoder=encoder)
        self.fp16 = fp16
        self.device = device
        self.batch_size = batch_size
        self.disable_noise_filter = disable_noise_filter
        if isinstance(input_tensor_size, list):
            self.input_image_size = input_tensor_size[:2]
        else:
            self.input_image_size = (input_tensor_size, input_tensor_size)
        self.to(device)
        if load_pretrained:
            self.load_state_dict(torch.load(fba_pretrained(), map_location=self.device))
        self.eval()

    def data_preprocessing(
        self, data: Union[PIL.Image.Image, np.ndarray]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Transform input image to suitable data format for neural network

        Args:
            data: input image

        Returns:
            input for neural network

        """
        resized = data.copy()
        if self.batch_size == 1:
            resized.thumbnail(self.input_image_size, resample=3)
        else:
            resized = resized.resize(self.input_image_size, resample=3)
        # noinspection PyTypeChecker
        image = np.array(resized, dtype=np.float64)
        image = image / 255.0  # Normalize image to [0, 1] values range
        if resized.mode == "RGB":
            image = image[:, :, ::-1]
        elif resized.mode == "L":
            image2 = np.copy(image)
            h, w = image2.shape
            image = np.zeros((h, w, 2))  # Transform trimap to binary data format
            image[image2 == 1, 1] = 1
            image[image2 == 0, 0] = 1
        else:
            raise ValueError("Incorrect color mode for image")
        h, w = image.shape[:2]  # Scale input mlt to 8
        h1 = int(np.ceil(1.0 * h / 8) * 8)
        w1 = int(np.ceil(1.0 * w / 8) * 8)
        x_scale = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_LANCZOS4)
        image_tensor = torch.from_numpy(x_scale).permute(2, 0, 1)[None, :, :, :].float()
        if resized.mode == "RGB":
            return image_tensor, groupnorm_normalise_image(
                image_tensor.clone(), format="nchw"
            )
        else:
            return (
                image_tensor,
                torch.from_numpy(trimap_transform(x_scale))
                .permute(2, 0, 1)[None, :, :, :]
                .float(),
            )

    def data_postprocessing(
        self, data: torch.tensor, trimap: PIL.Image.Image
    ) -> PIL.Image.Image:
        """
        Transforms output data from neural network to suitable data
        format for using with other components of this framework.

        Args:
            data: output data from neural network
            trimap: Map with the area we need to refine

        Returns:
            Segmentation mask as PIL Image instance

        """
        if trimap.mode != "L":
            raise ValueError("Incorrect color mode for trimap")
        pred = data.numpy().transpose((1, 2, 0))
        pred = cv2.resize(pred, trimap.size, cv2.INTER_LANCZOS4)[:, :, 0]
        # noinspection PyTypeChecker
        # Clean mask by removing all false predictions outside trimap and already known area
        trimap_arr = np.array(trimap.copy())
        pred[trimap_arr[:, :] == 0] = 0
        # pred[trimap_arr[:, :] == 255] = 1
        if not self.disable_noise_filter:
            pred[pred < 0.3] = 0
        return Image.fromarray(pred * 255).convert("L")

    def __call__(
        self,
        images: List[Union[str, pathlib.Path, PIL.Image.Image]],
        trimaps: List[Union[str, pathlib.Path, PIL.Image.Image]],
    ) -> List[PIL.Image.Image]:
        """
        Passes input images though neural network and returns segmentation masks as PIL.Image.Image instances

        Args:
            images: input images
            trimaps: Maps with the areas we need to refine

        Returns:
            segmentation masks as for input images, as PIL.Image.Image instances

        """

        if len(images) != len(trimaps):
            raise ValueError(
                "Len of specified arrays of images and trimaps should be equal!"
            )

        collect_masks = []
        autocast, dtype = get_precision_autocast(device=self.device, fp16=self.fp16)
        with autocast:
            cast_network(self, dtype)
            for idx_batch in batch_generator(range(len(images)), self.batch_size):
                inpt_images = thread_pool_processing(
                    lambda x: convert_image(load_image(images[x])), idx_batch
                )

                inpt_trimaps = thread_pool_processing(
                    lambda x: convert_image(load_image(trimaps[x]), mode="L"), idx_batch
                )

                inpt_img_batches = thread_pool_processing(
                    self.data_preprocessing, inpt_images
                )
                inpt_trimaps_batches = thread_pool_processing(
                    self.data_preprocessing, inpt_trimaps
                )

                inpt_img_batches_transformed = torch.vstack(
                    [i[1] for i in inpt_img_batches]
                )
                inpt_img_batches = torch.vstack([i[0] for i in inpt_img_batches])

                inpt_trimaps_transformed = torch.vstack(
                    [i[1] for i in inpt_trimaps_batches]
                )
                inpt_trimaps_batches = torch.vstack(
                    [i[0] for i in inpt_trimaps_batches]
                )

                with torch.no_grad():
                    inpt_img_batches = inpt_img_batches.to(self.device)
                    inpt_trimaps_batches = inpt_trimaps_batches.to(self.device)
                    inpt_img_batches_transformed = inpt_img_batches_transformed.to(
                        self.device
                    )
                    inpt_trimaps_transformed = inpt_trimaps_transformed.to(self.device)

                    output = super(FBAMatting, self).__call__(
                        inpt_img_batches,
                        inpt_trimaps_batches,
                        inpt_img_batches_transformed,
                        inpt_trimaps_transformed,
                    )
                    output_cpu = output.cpu()
                    del (
                        inpt_img_batches,
                        inpt_trimaps_batches,
                        inpt_img_batches_transformed,
                        inpt_trimaps_transformed,
                        output,
                    )
                masks = thread_pool_processing(
                    lambda x: self.data_postprocessing(output_cpu[x], inpt_trimaps[x]),
                    range(len(inpt_images)),
                )
                collect_masks += masks
            return collect_masks
