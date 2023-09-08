import secrets
from typing import List
from typing_extensions import Literal

import torch.cuda
from pydantic import BaseModel, field_validator, Field


class AuthConfig(BaseModel):
    """Config for web api token authentication"""

    auth: bool = True
    """Enables Token Authentication for API"""
    admin_token: str = secrets.token_hex(32)
    """Admin Token"""
    allowed_tokens: List[str] = [secrets.token_hex(32)]
    """All allowed tokens"""


class MLConfig(BaseModel):
    """Config for ml part of framework"""

    segmentation_network: Literal[
        "u2net", "deeplabv3", "basnet", "tracer_b7"
    ] = "tracer_b7"
    """Segmentation Network"""
    preprocessing_method: Literal["none", "stub"] = "none"
    """Pre-processing Method"""
    postprocessing_method: Literal["fba", "none"] = "fba"
    """Post-Processing Network"""
    device: str = "cpu"
    """Processing device"""
    batch_size_seg: int = Field(default=5, gt=0)
    """Batch size for segmentation network"""
    batch_size_matting: int = Field(default=1, gt=0)
    """Batch size for matting network"""
    seg_mask_size: int = Field(default=640, gt=0)
    """The size of the input image for the segmentation neural network."""
    matting_mask_size: int = Field(default=2048, gt=0)
    """The size of the input image for the matting neural network."""
    fp16: bool = False
    """Use half precision for inference"""
    trimap_dilation: int = 30
    """Dilation size for trimap"""
    trimap_erosion: int = 5
    """Erosion levels for trimap"""
    trimap_prob_threshold: int = 231
    """Probability threshold for trimap generation"""

    @field_validator("device")
    def device_validator(cls, value: str):
        if not torch.cuda.is_available() and "cuda" in value:
            raise ValueError(
                "GPU is not available, but specified as processing device!"
            )
        if "cuda" not in value and "cpu" != value:
            raise ValueError("Unknown processing device! It should be cpu or cuda!")
        return value


class WebAPIConfig(BaseModel):
    """FastAPI app config"""

    port: int = 5000
    """Web API port"""
    host: str = "0.0.0.0"
    """Web API host"""
    ml: MLConfig = MLConfig()
    """Config for ml part of framework"""
    auth: AuthConfig = AuthConfig()
    """Config for web api token authentication """
