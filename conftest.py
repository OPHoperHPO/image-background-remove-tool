"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from typing import Callable, Tuple, List, Union, Optional, Any

from carvekit.api.high import HiInterface
from carvekit.api.interface import Interface
from carvekit.ml.wrap.cascadepsp import CascadePSP
from carvekit.ml.wrap.yolov4 import SimplifiedYoloV4
from carvekit.trimap.cv_gen import CV2TrimapGenerator
from carvekit.trimap.generator import TrimapGenerator
from carvekit.utils.image_utils import convert_image, load_image
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub

from carvekit.ml.wrap.u2net import U2NET
from carvekit.ml.wrap.basnet import BASNET
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.deeplab_v3 import DeepLabV3
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.ml.wrap.scene_classifier import SceneClassifier

device = "cpu" if not torch.cuda.is_available() else "cuda"


@pytest.fixture()
def u2net_model() -> Callable[[bool], U2NET]:
    return lambda fb16: U2NET(
        layers_cfg="full",
        device=device,
        input_image_size=320,
        batch_size=10,
        load_pretrained=True,
        fp16=fb16,
    )


@pytest.fixture()
def scene_classifier_model() -> Callable[[bool], SceneClassifier]:
    return lambda fb16: SceneClassifier(
        device=device,
        batch_size=5,
        fp16=fb16,
    )


@pytest.fixture()
def tracer_model() -> Callable[[bool], TracerUniversalB7]:
    return lambda fb16: TracerUniversalB7(
        device=device,
        input_image_size=320,
        batch_size=10,
        load_pretrained=True,
        fp16=fb16,
    )


@pytest.fixture()
def trimap_instance() -> Callable[[], TrimapGenerator]:
    return lambda: TrimapGenerator()


@pytest.fixture()
def cv2_trimap_instance() -> Callable[[], CV2TrimapGenerator]:
    return lambda: CV2TrimapGenerator(kernel_size=30, erosion_iters=0)


@pytest.fixture()
def preprocessing_stub_instance() -> Callable[[], PreprocessingStub]:
    return lambda: PreprocessingStub()


@pytest.fixture()
def matting_method_instance(fba_model, trimap_instance):
    return lambda: MattingMethod(
        matting_module=fba_model(False),
        trimap_generator=trimap_instance(),
        device="cpu",
    )


@pytest.fixture()
def high_interface_instance() -> Callable[[], HiInterface]:
    return lambda: HiInterface(
        batch_size_seg=5,
        batch_size_matting=1,
        device=device,
        seg_mask_size=320,
        matting_mask_size=2048,
    )


@pytest.fixture()
def interface_instance(
    u2net_model, preprocessing_stub_instance, matting_method_instance
) -> Callable[[], Interface]:
    return lambda: Interface(
        u2net_model(False),
        pre_pipe=preprocessing_stub_instance(),
        post_pipe=matting_method_instance(),
        device=device,
    )


@pytest.fixture()
def fba_model() -> Callable[[bool], FBAMatting]:
    return lambda fp16: FBAMatting(
        device=device,
        input_tensor_size=1024,
        batch_size=2,
        load_pretrained=True,
        fp16=fp16,
    )


@pytest.fixture()
def deeplabv3_model() -> Callable[[bool], DeepLabV3]:
    return lambda fp16: DeepLabV3(
        device=device,
        batch_size=10,
        load_pretrained=True,
        fp16=fp16,
    )


@pytest.fixture()
def basnet_model() -> Callable[[bool], BASNET]:
    return lambda fp16: BASNET(
        device=device,
        input_image_size=320,
        batch_size=10,
        load_pretrained=True,
        fp16=fp16,
    )


@pytest.fixture()
def image_str(image_path) -> str:
    return str(image_path.absolute())


@pytest.fixture()
def image_path() -> Path:
    return Path(__file__).parent.joinpath("tests").joinpath("data", "cat.jpg")


@pytest.fixture()
def image_mask(image_path) -> Image.Image:
    return Image.open(image_path.with_name("cat_mask").with_suffix(".png"))


@pytest.fixture()
def image_trimap(image_path) -> Image.Image:
    return Image.open(image_path.with_name("cat_trimap").with_suffix(".png")).convert(
        "L"
    )


@pytest.fixture()
def image_pil(image_path) -> Image.Image:
    return Image.open(image_path)


@pytest.fixture()
def black_image_pil() -> Image.Image:
    return Image.new("RGB", (512, 512))


@pytest.fixture()
def converted_pil_image(image_pil) -> Image.Image:
    return convert_image(load_image(image_pil))


@pytest.fixture()
def available_models(
    u2net_model,
    deeplabv3_model,
    basnet_model,
    preprocessing_stub_instance,
    matting_method_instance,
) -> Tuple[
    List[Union[Callable[[], U2NET], Callable[[], DeepLabV3], Callable[[], BASNET]]],
    List[Optional[Callable[[], PreprocessingStub]]],
    List[Union[Optional[Callable[[], MattingMethod]], Any]],
]:
    models = [u2net_model, deeplabv3_model, basnet_model]
    pre_pipes = [None, preprocessing_stub_instance]
    post_pipes = [None, matting_method_instance]
    return models, pre_pipes, post_pipes


@pytest.fixture()
def cascadepsp() -> Callable[[bool, str, int], CascadePSP]:
    def cascadepsp_getter(fp16=False, device_=device, batch_size=1):
        return CascadePSP(
            device=device_,
            fp16=fp16,
            input_tensor_size=1024,
            batch_size=batch_size,
        )

    return cascadepsp_getter


@pytest.fixture()
def yoloV4() -> Callable[[bool], SimplifiedYoloV4]:
    return lambda fp16: SimplifiedYoloV4(
        device=device,
        fp16=fp16,
        batch_size=5,
    )
