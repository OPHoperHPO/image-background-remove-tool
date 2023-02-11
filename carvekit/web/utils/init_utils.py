import warnings
from os import getenv
from typing import Union

from loguru import logger

from carvekit.ml.wrap.cascadepsp import CascadePSP
from carvekit.ml.wrap.isnet import ISNet
from carvekit.ml.wrap.scene_classifier import SceneClassifier
from carvekit.web.schemas.config import WebAPIConfig, MLConfig, AuthConfig

from carvekit.api.interface import Interface
from carvekit.api.autointerface import AutoInterface

from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.u2net import U2NET
from carvekit.ml.wrap.deeplab_v3 import DeepLabV3
from carvekit.ml.wrap.basnet import BASNET
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.ml.wrap.yolov4 import SimplifiedYoloV4


from carvekit.pipelines.postprocessing import MattingMethod, CasMattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub, AutoScene
from carvekit.trimap.generator import TrimapGenerator


def init_config() -> WebAPIConfig:
    default_config = WebAPIConfig()
    config = WebAPIConfig(
        **dict(
            port=int(getenv("CARVEKIT_PORT", default_config.port)),
            host=getenv("CARVEKIT_HOST", default_config.host),
            ml=MLConfig(
                segmentation_network=getenv(
                    "CARVEKIT_SEGMENTATION_NETWORK",
                    default_config.ml.segmentation_network,
                ),
                preprocessing_method=getenv(
                    "CARVEKIT_PREPROCESSING_METHOD",
                    default_config.ml.preprocessing_method,
                ),
                postprocessing_method=getenv(
                    "CARVEKIT_POSTPROCESSING_METHOD",
                    default_config.ml.postprocessing_method,
                ),
                device=getenv("CARVEKIT_DEVICE", default_config.ml.device),
                batch_size_pre=int(
                    getenv("CARVEKIT_BATCH_SIZE_PRE", default_config.ml.batch_size_pre)
                ),
                batch_size_seg=int(
                    getenv("CARVEKIT_BATCH_SIZE_SEG", default_config.ml.batch_size_seg)
                ),
                batch_size_matting=int(
                    getenv(
                        "CARVEKIT_BATCH_SIZE_MATTING",
                        default_config.ml.batch_size_matting,
                    )
                ),
                batch_size_refine=int(
                    getenv(
                        "CARVEKIT_BATCH_SIZE_REFINE",
                        default_config.ml.batch_size_refine,
                    )
                ),
                seg_mask_size=int(
                    getenv("CARVEKIT_SEG_MASK_SIZE", default_config.ml.seg_mask_size)
                ),
                matting_mask_size=int(
                    getenv(
                        "CARVEKIT_MATTING_MASK_SIZE",
                        default_config.ml.matting_mask_size,
                    )
                ),
                refine_mask_size=int(
                    getenv(
                        "CARVEKIT_REFINE_MASK_SIZE",
                        default_config.ml.refine_mask_size,
                    )
                ),
                fp16=bool(int(getenv("CARVEKIT_FP16", default_config.ml.fp16))),
                trimap_prob_threshold=int(
                    getenv(
                        "CARVEKIT_TRIMAP_PROB_THRESHOLD",
                        default_config.ml.trimap_prob_threshold,
                    )
                ),
                trimap_dilation=int(
                    getenv(
                        "CARVEKIT_TRIMAP_DILATION", default_config.ml.trimap_dilation
                    )
                ),
                trimap_erosion=int(
                    getenv("CARVEKIT_TRIMAP_EROSION", default_config.ml.trimap_erosion)
                ),
            ),
            auth=AuthConfig(
                auth=bool(
                    int(getenv("CARVEKIT_AUTH_ENABLE", default_config.auth.auth))
                ),
                admin_token=getenv(
                    "CARVEKIT_ADMIN_TOKEN", default_config.auth.admin_token
                ),
                allowed_tokens=default_config.auth.allowed_tokens
                if getenv("CARVEKIT_ALLOWED_TOKENS") is None
                else getenv("CARVEKIT_ALLOWED_TOKENS").split(","),
            ),
        )
    )

    logger.info(f"Admin token for Web API is {config.auth.admin_token}")
    logger.debug(f"Running Web API with this config: {config.json()}")
    return config


def init_interface(config: Union[WebAPIConfig, MLConfig]) -> Interface:
    if isinstance(config, WebAPIConfig):
        config = config.ml
    if config.preprocessing_method == "auto":
        warnings.warn(
            "Preprocessing_method is set to `auto`."
            "We will use automatic methods to determine the best methods for your images! "
            "Please note that this is not always the best option and all other options will be ignored!"
        )
        scene_classifier = SceneClassifier(
            device=config.device, batch_size=config.batch_size_pre, fp16=config.fp16
        )
        object_classifier = SimplifiedYoloV4(
            device=config.device, batch_size=config.batch_size_pre, fp16=config.fp16
        )
        return AutoInterface(
            scene_classifier=scene_classifier,
            object_classifier=object_classifier,
            segmentation_batch_size=config.batch_size_seg,
            postprocessing_batch_size=config.batch_size_matting,
            postprocessing_image_size=config.matting_mask_size,
            segmentation_device=config.device,
            postprocessing_device=config.device,
            fp16=config.fp16,
        )

    else:
        if config.segmentation_network == "u2net":
            seg_net = U2NET(
                device=config.device,
                batch_size=config.batch_size_seg,
                input_image_size=config.seg_mask_size,
                fp16=config.fp16,
            )
        elif config.segmentation_network == "isnet":
            seg_net = ISNet(
                device=config.device,
                batch_size=config.batch_size_seg,
                input_image_size=config.seg_mask_size,
                fp16=config.fp16,
            )
        elif config.segmentation_network == "deeplabv3":
            seg_net = DeepLabV3(
                device=config.device,
                batch_size=config.batch_size_seg,
                input_image_size=config.seg_mask_size,
                fp16=config.fp16,
            )
        elif config.segmentation_network == "basnet":
            seg_net = BASNET(
                device=config.device,
                batch_size=config.batch_size_seg,
                input_image_size=config.seg_mask_size,
                fp16=config.fp16,
            )
        elif config.segmentation_network == "tracer_b7":
            seg_net = TracerUniversalB7(
                device=config.device,
                batch_size=config.batch_size_seg,
                input_image_size=config.seg_mask_size,
                fp16=config.fp16,
            )
        else:
            seg_net = TracerUniversalB7(
                device=config.device,
                batch_size=config.batch_size_seg,
                input_image_size=config.seg_mask_size,
                fp16=config.fp16,
            )

        if config.preprocessing_method == "stub":
            preprocessing = PreprocessingStub()
        elif config.preprocessing_method == "none":
            preprocessing = None
        elif config.preprocessing_method == "autoscene":
            preprocessing = AutoScene(
                scene_classifier=SceneClassifier(
                    device=config.device,
                    batch_size=config.batch_size_pre,
                    fp16=config.fp16,
                )
            )
        else:
            preprocessing = None

        if config.postprocessing_method == "fba":
            fba = FBAMatting(
                device=config.device,
                batch_size=config.batch_size_matting,
                input_tensor_size=config.matting_mask_size,
                fp16=config.fp16,
            )
            trimap_generator = TrimapGenerator(
                prob_threshold=config.trimap_prob_threshold,
                kernel_size=config.trimap_dilation,
                erosion_iters=config.trimap_erosion,
            )
            postprocessing = MattingMethod(
                device=config.device,
                matting_module=fba,
                trimap_generator=trimap_generator,
            )
        elif config.postprocessing_method == "cascade_fba":
            cascadepsp = CascadePSP(
                device=config.device,
                batch_size=config.batch_size_refine,
                input_tensor_size=config.refine_mask_size,
                fp16=config.fp16,
            )
            fba = FBAMatting(
                device=config.device,
                batch_size=config.batch_size_matting,
                input_tensor_size=config.matting_mask_size,
                fp16=config.fp16,
            )
            trimap_generator = TrimapGenerator(
                prob_threshold=config.trimap_prob_threshold,
                kernel_size=config.trimap_dilation,
                erosion_iters=config.trimap_erosion,
            )
            postprocessing = CasMattingMethod(
                device=config.device,
                matting_module=fba,
                trimap_generator=trimap_generator,
                refining_module=cascadepsp,
            )
        elif config.postprocessing_method == "none":
            postprocessing = None
        else:
            postprocessing = None

        interface = Interface(
            pre_pipe=preprocessing,
            post_pipe=postprocessing,
            seg_pipe=seg_net,
            device=config.device,
        )
    return interface
