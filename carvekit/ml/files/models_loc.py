"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import pathlib
from carvekit.ml.files import checkpoints_dir
from carvekit.utils.download_models import downloader


def u2net_full_pretrained() -> pathlib.Path:
    """Returns u2net pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("u2net.pth")


def isnet_full_pretrained() -> pathlib.Path:
    """Returns isnet pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("isnet.pth")


def basnet_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("basnet.pth")


def deeplab_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("deeplab.pth")


def fba_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("fba_matting.pth")


def tracer_b7_pretrained() -> pathlib.Path:
    """Returns TRACER with EfficientNet v1 b7 encoder pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("tracer_b7.pth")


def scene_classifier_pretrained() -> pathlib.Path:
    """Returns scene classifier pretrained model location
    This model is used to classify scenes into 3 categories: hard, soft, digital

    hard - scenes with hard edges, such as objects, buildings, etc.
    soft - scenes with soft edges, such as portraits, hairs, animal, etc.
    digital - digital scenes, such as screenshots, graphics, etc.

    more info: https://huggingface.co/Carve/scene_classifier

    Returns:
        pathlib.Path to model location
    """
    return downloader("scene_classifier.pth")


def yolov4_coco_pretrained() -> pathlib.Path:
    """Returns yolov4 classifier pretrained model location
    This model is used to classify objects in images.

    Training dataset: COCO 2017
    Training classes: 80

    It's a modified version of the original model from https://github.com/Tianxiaomo/pytorch-YOLOv4 (pytorch)
    We have only added coco classnames to the model.

    Returns:
        pathlib.Path to model location
    """
    return downloader("yolov4_coco_with_classes.pth")


def cascadepsp_pretrained() -> pathlib.Path:
    """Returns cascade psp pretrained model location
    This model is used to refine segmentation masks.

    Training dataset: MSRA-10K, DUT-OMRON, ECSSD and FSS-1000
    more info: https://huggingface.co/Carve/cascadepsp

    Returns:
        pathlib.Path to model location
    """
    return downloader("cascadepsp.pth")


def download_all():
    u2net_full_pretrained()
    fba_pretrained()
    deeplab_pretrained()
    basnet_pretrained()
    tracer_b7_pretrained()
    scene_classifier_pretrained()
    yolov4_coco_pretrained()
    cascadepsp_pretrained()
