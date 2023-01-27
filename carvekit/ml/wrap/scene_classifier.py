"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import pathlib

import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import List, Union, Tuple
from torch.autograd import Variable

from carvekit.ml.files.models_loc import scene_classifier_pretrained
from carvekit.utils.image_utils import load_image, convert_image
from carvekit.utils.models_utils import get_precision_autocast, cast_network
from carvekit.utils.pool_utils import thread_pool_processing, batch_generator

__all__ = ["SceneClassifier"]


class SceneClassifier:
    """
    SceneClassifier model interface

    Description:
        Performs a primary analysis of the image in order to select the necessary method for removing the background.
        The choice is made by classifying the scene type.

        The output can be the following types:
        - hard
        - soft
        - digital

    """

    def __init__(
        self,
        topk: int = 1,
        device="cpu",
        batch_size: int = 4,
        fp16: bool = False,
        model_path: Union[str, pathlib.Path] = None,
    ):
        """
        Initialize the Scene Classifier.

        Args:
            topk: number of top classes to return
            device: processing device
            batch_size: the number of images that the neural network processes in one run
            fp16: use fp16 precision

        """
        if model_path is None:
            model_path = scene_classifier_pretrained()
        self.topk = topk
        self.fp16 = fp16
        self.device = device
        self.batch_size = batch_size

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        state_dict = torch.load(model_path, map_location=device)
        self.model = state_dict["model"]
        self.class_to_idx = state_dict["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.model.to(device)
        self.model.eval()

    def data_preprocessing(self, data: PIL.Image.Image) -> torch.FloatTensor:
        """
        Transform input image to suitable data format for neural network

        Args:
            data: input image

        Returns:
            input for neural network

        """

        return torch.unsqueeze(self.transform(data), 0).type(torch.FloatTensor)

    def data_postprocessing(self, data: torch.Tensor) -> Tuple[List[str], List[float]]:
        """
        Transforms output data from neural network to suitable data
        format for using with other components of this framework.

        Args:
            data: output data from neural network

        Returns:
            Top-k class of scene type, probability of these classes

        """
        ps = F.softmax(data.float(), dim=0)
        topk = ps.cpu().topk(self.topk)

        probs, classes = (e.data.numpy().squeeze().tolist() for e in topk)
        if isinstance(classes, int):
            classes = [classes]
            probs = [probs]
        return list(map(lambda x: self.idx_to_class[x], classes)), probs

    def __call__(
        self, images: List[Union[str, pathlib.Path, PIL.Image.Image]]
    ) -> Tuple[List[str], List[float]]:
        """
        Passes input images though neural network and returns class predictions.

        Args:
            images: input images

        Returns:
            Top-k class of scene type, probability of these classes for every passed image

        """
        collect_masks = []
        autocast, dtype = get_precision_autocast(device=self.device, fp16=self.fp16)
        with autocast:
            cast_network(self.model, dtype)
            for image_batch in batch_generator(images, self.batch_size):
                converted_images = thread_pool_processing(
                    lambda x: convert_image(load_image(x)), image_batch
                )
                batches = torch.vstack(
                    thread_pool_processing(self.data_preprocessing, converted_images)
                )
                with torch.no_grad():
                    batches = Variable(batches).to(self.device)
                    masks = self.model.forward(batches)
                    masks_cpu = masks.cpu()
                    del batches, masks
                masks = thread_pool_processing(
                    lambda x: self.data_postprocessing(masks_cpu[x]),
                    range(len(converted_images)),
                )
                collect_masks += masks

        return collect_masks
