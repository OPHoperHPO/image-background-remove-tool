"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import pathlib
import warnings

import PIL
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Union, List

from carvekit.ml.arch.cascadepsp.pspnet import RefinementModule
from carvekit.ml.arch.cascadepsp.utils import (
    process_im_single_pass,
    process_high_res_im,
)
from carvekit.ml.files.models_loc import cascadepsp_pretrained
from carvekit.utils.image_utils import convert_image, load_image
from carvekit.utils.models_utils import get_precision_autocast, cast_network
from carvekit.utils.pool_utils import batch_generator, thread_pool_processing

__all__ = ["CascadePSP"]


class CascadePSP(RefinementModule):
    """
    CascadePSP to refine the mask from segmentation network
    """

    def __init__(
        self,
        device="cpu",
        input_tensor_size: int = 900,
        batch_size: int = 1,
        load_pretrained: bool = True,
        fp16: bool = False,
        mask_binary_threshold=127,
        global_step_only=False,
        processing_accelerate_image_size=2048,
    ):
        """
        Initialize the CascadePSP model

        Args:
            device: processing device
            input_tensor_size: input image size
            batch_size: the number of images that the neural network processes in one run
            load_pretrained: loading pretrained model
            fp16: use half precision
            global_step_only: if True, only global step will be used for prediction. See paper for details.
            mask_binary_threshold: threshold for binary mask, default 70, set to 0 for no threshold
            processing_accelerate_image_size: thumbnail size for image processing acceleration. Set to 0 to disable

        """
        super().__init__()
        self.fp16 = fp16
        self.device = device
        self.batch_size = batch_size
        self.mask_binary_threshold = mask_binary_threshold
        self.global_step_only = global_step_only
        self.processing_accelerate_image_size = processing_accelerate_image_size
        self.input_tensor_size = input_tensor_size

        self.to(device)
        if batch_size > 1:
            warnings.warn(
                "Batch size > 1 is experimental feature for CascadePSP."
                " Please, don't use it if you have GPU with small memory!"
            )
        if load_pretrained:
            self.load_state_dict(
                torch.load(cascadepsp_pretrained(), map_location=self.device)
            )
        self.eval()

        self._image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self._seg_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def data_preprocessing(self, data: Union[PIL.Image.Image]) -> torch.FloatTensor:
        """
        Transform input image to suitable data format for neural network

        Args:
            data: input image

        Returns:
            input for neural network

        """
        preprocessed_data = data.copy()
        if self.batch_size == 1 and self.processing_accelerate_image_size > 0:
            # Okay, we have only one image, so
            # we can use image processing acceleration for accelerate high resolution image processing
            preprocessed_data.thumbnail(
                (
                    self.processing_accelerate_image_size,
                    self.processing_accelerate_image_size,
                )
            )
        elif self.batch_size == 1:
            pass  # No need to do anything
        elif self.batch_size > 1 and self.global_step_only is True:
            # If we have more than one image and we use only global step,
            # there aren't any reason to use image processing acceleration,
            # because we will use only global step for prediction and anyway it will be resized to input_tensor_size
            preprocessed_data = preprocessed_data.resize(
                (self.input_tensor_size, self.input_tensor_size)
            )
        elif (
            self.batch_size > 1
            and self.global_step_only is False
            and self.processing_accelerate_image_size > 0
        ):
            # If we have more than one image and we use local step,
            # we can use image processing acceleration for accelerate high resolution image processing
            # but we need to resize image to processing_accelerate_image_size to stack it with other images
            preprocessed_data = preprocessed_data.resize(
                (
                    self.processing_accelerate_image_size,
                    self.processing_accelerate_image_size,
                )
            )
        elif (
            self.batch_size > 1
            and self.global_step_only is False
            and not (self.processing_accelerate_image_size > 0)
        ):
            raise ValueError(
                "If you use local step with batch_size > 2, "
                "you need to set processing_accelerate_image_size > 0,"
                "since we cannot stack images with different sizes to one batch"
            )
        else:  # some extra cases
            preprocessed_data = preprocessed_data.resize(
                (
                    self.processing_accelerate_image_size,
                    self.processing_accelerate_image_size,
                )
            )

        if data.mode == "RGB":
            preprocessed_data = self._image_transform(
                np.array(preprocessed_data)
            ).unsqueeze(0)
        elif data.mode == "L":
            preprocessed_data = np.array(preprocessed_data)
            if 0 < self.mask_binary_threshold <= 255:
                preprocessed_data = (
                    preprocessed_data > self.mask_binary_threshold
                ).astype(np.uint8) * 255
            elif self.mask_binary_threshold > 255 or self.mask_binary_threshold < 0:
                warnings.warn(
                    "mask_binary_threshold should be in range [0, 255], "
                    "but got {}. Disabling mask_binary_threshold!".format(
                        self.mask_binary_threshold
                    )
                )

            preprocessed_data = self._seg_transform(preprocessed_data).unsqueeze(
                0
            )  # [H,W,1]

        return preprocessed_data

    @staticmethod
    def data_postprocessing(
        data: torch.Tensor, mask: PIL.Image.Image
    ) -> PIL.Image.Image:
        """
        Transforms output data from neural network to suitable data
        format for using with other components of this framework.

        Args:
            data: output data from neural network
            mask: input mask

        Returns:
            Segmentation mask as PIL Image instance

        """
        refined_mask = (data[0, :, :].cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(refined_mask).convert("L").resize(mask.size)

    def safe_forward(self, im, seg, inter_s8=None, inter_s4=None):
        """
        Slightly pads the input image such that its length is a multiple of 8
        """
        b, _, ph, pw = seg.shape
        if (ph % 8 != 0) or (pw % 8 != 0):
            newH = (ph // 8 + 1) * 8
            newW = (pw // 8 + 1) * 8
            p_im = torch.zeros(b, 3, newH, newW, device=im.device)
            p_seg = torch.zeros(b, 1, newH, newW, device=im.device) - 1

            p_im[:, :, 0:ph, 0:pw] = im
            p_seg[:, :, 0:ph, 0:pw] = seg
            im = p_im
            seg = p_seg

            if inter_s8 is not None:
                p_inter_s8 = torch.zeros(b, 1, newH, newW, device=im.device) - 1
                p_inter_s8[:, :, 0:ph, 0:pw] = inter_s8
                inter_s8 = p_inter_s8
            if inter_s4 is not None:
                p_inter_s4 = torch.zeros(b, 1, newH, newW, device=im.device) - 1
                p_inter_s4[:, :, 0:ph, 0:pw] = inter_s4
                inter_s4 = p_inter_s4

        images = super().__call__(im, seg, inter_s8, inter_s4)
        return_im = {}

        for key in ["pred_224", "pred_28_3", "pred_56_2"]:
            return_im[key] = images[key][:, :, 0:ph, 0:pw]
        del images

        return return_im

    def __call__(
        self,
        images: List[Union[str, pathlib.Path, PIL.Image.Image]],
        masks: List[Union[str, pathlib.Path, PIL.Image.Image]],
    ) -> List[PIL.Image.Image]:
        """
        Passes input images though neural network and returns segmentation masks as PIL.Image.Image instances

        Args:
            images: input images
            masks: Segmentation masks to refine

        Returns:
            segmentation masks as for input images, as PIL.Image.Image instances

        """

        if len(images) != len(masks):
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

                inpt_masks = thread_pool_processing(
                    lambda x: convert_image(load_image(masks[x]), mode="L"), idx_batch
                )

                inpt_img_batches = thread_pool_processing(
                    self.data_preprocessing, inpt_images
                )
                inpt_masks_batches = thread_pool_processing(
                    self.data_preprocessing, inpt_masks
                )
                if self.batch_size > 1:  # We need to stack images, if batch_size > 1
                    inpt_img_batches = torch.vstack(inpt_img_batches)
                    inpt_masks_batches = torch.vstack(inpt_masks_batches)
                else:
                    inpt_img_batches = inpt_img_batches[
                        0
                    ]  # Get only one image from list
                    inpt_masks_batches = inpt_masks_batches[0]

                with torch.no_grad():
                    inpt_img_batches = inpt_img_batches.to(self.device)
                    inpt_masks_batches = inpt_masks_batches.to(self.device)
                    if self.global_step_only:
                        refined_batches = process_im_single_pass(
                            self,
                            inpt_img_batches,
                            inpt_masks_batches,
                            self.input_tensor_size,
                        )

                    else:
                        refined_batches = process_high_res_im(
                            self,
                            inpt_img_batches,
                            inpt_masks_batches,
                            self.input_tensor_size,
                        )

                    refined_masks = refined_batches.cpu()
                    del (inpt_img_batches, inpt_masks_batches, refined_batches)
                collect_masks += thread_pool_processing(
                    lambda x: self.data_postprocessing(refined_masks[x], inpt_masks[x]),
                    range(len(inpt_masks)),
                )
            return collect_masks
