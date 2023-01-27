"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import pathlib
import warnings

from typing import List, Union
import PIL.Image
import numpy as np
import torch
from PIL import Image

from carvekit.ml.arch.u2net.u2net import U2NETArchitecture
from carvekit.ml.files.models_loc import u2net_full_pretrained
from carvekit.utils.image_utils import load_image, convert_image
from carvekit.utils.pool_utils import thread_pool_processing, batch_generator

__all__ = ["U2NET"]


class U2NET(U2NETArchitecture):
    """U^2-Net model interface"""

    def __init__(
        self,
        layers_cfg="full",
        device="cpu",
        input_image_size: Union[List[int], int] = 320,
        batch_size: int = 10,
        load_pretrained: bool = True,
        fp16: bool = False,
    ):
        """
        Initialize the U2NET model

        Args:
            layers_cfg: neural network layers configuration
            device: processing device
            input_image_size: input image size
            batch_size: the number of images that the neural network processes in one run
            load_pretrained: loading pretrained model
            fp16: use fp16 precision // not supported at this moment.

        """
        super(U2NET, self).__init__(cfg_type=layers_cfg, out_ch=1)
        if fp16:
            warnings.warn("FP16 is not supported at this moment for U2NET model")
        self.device = device
        self.batch_size = batch_size
        if isinstance(input_image_size, list):
            self.input_image_size = input_image_size[:2]
        else:
            self.input_image_size = (input_image_size, input_image_size)
        self.to(device)
        if load_pretrained:
            self.load_state_dict(
                torch.load(u2net_full_pretrained(), map_location=self.device)
            )

        self.eval()

    def data_preprocessing(self, data: PIL.Image.Image) -> torch.FloatTensor:
        """
        Transform input image to suitable data format for neural network

        Args:
            data (PIL.Image.Image): input image

        Returns:
            torch.FloatTensor: input for neural network

        """
        resized = data.resize(self.input_image_size, resample=3)
        # noinspection PyTypeChecker
        resized_arr = np.array(resized, dtype=float)
        temp_image = np.zeros((resized_arr.shape[0], resized_arr.shape[1], 3))
        if np.max(resized_arr) != 0:
            resized_arr /= np.max(resized_arr)
        temp_image[:, :, 0] = (resized_arr[:, :, 0] - 0.485) / 0.229
        temp_image[:, :, 1] = (resized_arr[:, :, 1] - 0.456) / 0.224
        temp_image[:, :, 2] = (resized_arr[:, :, 2] - 0.406) / 0.225
        temp_image = temp_image.transpose((2, 0, 1))
        temp_image = np.expand_dims(temp_image, 0)
        return torch.from_numpy(temp_image).type(torch.FloatTensor)

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
        data = data.unsqueeze(0)
        mask = data[:, 0, :, :]
        ma = torch.max(mask)  # Normalizes prediction
        mi = torch.min(mask)
        predict = ((mask - mi) / (ma - mi)).squeeze()
        predict_np = predict.cpu().data.numpy() * 255
        mask = Image.fromarray(predict_np).convert("L")
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
        for image_batch in batch_generator(images, self.batch_size):
            converted_images = thread_pool_processing(
                lambda x: convert_image(load_image(x)), image_batch
            )
            batches = torch.vstack(
                thread_pool_processing(self.data_preprocessing, converted_images)
            )
            with torch.no_grad():
                batches = batches.to(self.device)
                masks, d2, d3, d4, d5, d6, d7 = super(U2NET, self).__call__(batches)
                masks_cpu = masks.cpu()
                del d2, d3, d4, d5, d6, d7, batches, masks
            masks = thread_pool_processing(
                lambda x: self.data_postprocessing(masks_cpu[x], converted_images[x]),
                range(len(converted_images)),
            )
            collect_masks += masks
        return collect_masks
