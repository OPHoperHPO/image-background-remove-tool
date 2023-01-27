"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool

Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].

License: Apache License 2.0
"""
import warnings

from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.ml.wrap.cascadepsp import CascadePSP
from carvekit.ml.wrap.scene_classifier import SceneClassifier
from carvekit.pipelines.preprocessing import AutoScene
from carvekit.ml.wrap.u2net import U2NET
from carvekit.pipelines.postprocessing import CasMattingMethod
from carvekit.trimap.generator import TrimapGenerator


class HiInterface(Interface):
    def __init__(
        self,
        object_type: str = "auto",
        batch_size_pre=5,
        batch_size_seg=2,
        batch_size_matting=1,
        batch_size_refine=1,
        device="cpu",
        seg_mask_size=640,
        matting_mask_size=2048,
        refine_mask_size=900,
        trimap_prob_threshold=231,
        trimap_dilation=30,
        trimap_erosion_iters=5,
        fp16=False,
    ):
        """
        Initializes High Level interface.

        Args:
            object_type (str, default=object): Interest object type. Can be "object" or "hairs-like".
            matting_mask_size (int, default=2048):  The size of the input image for the matting neural network.
            seg_mask_size (int, default=640): The size of the input image for the segmentation neural network.
            batch_size_pre (int, default=5: Number of images processed per one preprocessing method call.
            batch_size_seg (int, default=2): Number of images processed per one segmentation neural network call.
            batch_size_matting (int, matting=1): Number of images processed per one matting neural network call.
            device (Literal[cpu, cuda], default=cpu): Processing device
            fp16 (bool, default=False): Use half precision. Reduce memory usage and increase speed.
            .. CAUTION:: ⚠️ **Experimental support**
            trimap_prob_threshold (int, default=231): Probability threshold at which the prob_filter and prob_as_unknown_area operations will be applied
            trimap_dilation (int, default=30): The size of the offset radius from the object mask in pixels when forming an unknown area
            trimap_erosion_iters (int, default=5): The number of iterations of erosion that the object's mask will be subjected to before forming an unknown area
            refine_mask_size (int, default=900): The size of the input image for the refinement neural network.
            batch_size_refine (int, default=1): Number of images processed per one refinement neural network call.


        .. NOTE::
            1. Changing seg_mask_size may cause an `out-of-memory` error if the value is too large, and it may also
            result in reduced precision. I do not recommend changing this value. You can change `matting_mask_size` in
            range from `(1024 to 4096)` to improve object edge refining quality, but it will cause extra large RAM and
            video memory consume. Also, you can change batch size to accelerate background removal, but it also causes
            extra large video memory consume, if value is too big.
            2. Changing `trimap_prob_threshold`, `trimap_kernel_size`, `trimap_erosion_iters` may improve object edge
            refining quality.
        """
        preprocess_pipeline = None

        if object_type == "object":
            self._segnet = TracerUniversalB7(
                device=device,
                batch_size=batch_size_seg,
                input_image_size=seg_mask_size,
                fp16=fp16,
            )
        elif object_type == "hairs-like":
            self._segnet = U2NET(
                device=device,
                batch_size=batch_size_seg,
                input_image_size=seg_mask_size,
                fp16=fp16,
            )
        elif object_type == "auto":
            # Using Tracer by default,
            # but it will dynamically switch to other if needed
            self._segnet = TracerUniversalB7(
                device=device,
                batch_size=batch_size_seg,
                input_image_size=seg_mask_size,
                fp16=fp16,
            )
            self._scene_classifier = SceneClassifier(
                device=device, fp16=fp16, batch_size=batch_size_pre
            )
            preprocess_pipeline = AutoScene(scene_classifier=self._scene_classifier)

        else:
            warnings.warn(
                f"Unknown object type: {object_type}. Using default object type: object"
            )
            self._segnet = TracerUniversalB7(
                device=device,
                batch_size=batch_size_seg,
                input_image_size=seg_mask_size,
                fp16=fp16,
            )

        self._cascade_psp = CascadePSP(
            device=device,
            batch_size=batch_size_refine,
            input_tensor_size=refine_mask_size,
            fp16=fp16,
        )
        self._fba = FBAMatting(
            batch_size=batch_size_matting,
            device=device,
            input_tensor_size=matting_mask_size,
            fp16=fp16,
        )
        self._trimap_generator = TrimapGenerator(
            prob_threshold=trimap_prob_threshold,
            kernel_size=trimap_dilation,
            erosion_iters=trimap_erosion_iters,
        )
        super(HiInterface, self).__init__(
            pre_pipe=preprocess_pipeline,
            seg_pipe=self._segnet,
            post_pipe=CasMattingMethod(
                refining_module=self._cascade_psp,
                matting_module=self._fba,
                trimap_generator=self._trimap_generator,
                device=device,
            ),
            device=device,
        )
