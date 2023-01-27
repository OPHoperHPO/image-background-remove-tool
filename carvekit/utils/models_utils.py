"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool

Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].

License: Apache License 2.0
"""

import random
import warnings
from typing import Union, Tuple, Any

import torch
from torch import autocast


class EmptyAutocast(object):
    """
    Empty class for any auto-casting disabling.
    """

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def __call__(self, func):
        return


def get_precision_autocast(
    device="cpu", fp16=True, override_dtype=None
) -> Union[
    Tuple[EmptyAutocast, Union[torch.dtype, Any]],
    Tuple[autocast, Union[torch.dtype, Any]],
]:
    """
    Returns precision and auto-cast settings for given device and fp16 settings.

    Args:
        device (Literal[cpu, cuda]): Device to get precision and auto-cast settings for.
        fp16 (bool): Whether to use fp16 precision.
        override_dtype (bool): Override dtype for auto-cast.

    Returns:
        Union[Tuple[EmptyAutocast, Union[torch.dtype, Any]],Tuple[autocast, Union[torch.dtype, Any]]]: Autocast object, dtype
    """
    dtype = torch.float32
    cache_enabled = None

    if device == "cpu" and fp16:
        warnings.warn("FP16 is not supported on CPU. Using FP32 instead.")
        dtype = torch.float32

        # TODO: Implement BFP16 on CPU. There are unexpected slowdowns on cpu on a clean environment.
        # warnings.warn(
        #     "Accuracy BFP16 has experimental support on the CPU. "
        #     "This may result in an unexpected reduction in quality."
        # )
        # dtype = (
        #     torch.bfloat16
        # )  # Using bfloat16 for CPU, since autocast is not supported for float16

    if "cuda" in device and fp16:
        dtype = torch.float16
        cache_enabled = True

    if override_dtype is not None:
        dtype = override_dtype

    if dtype == torch.float32 and device == "cpu":
        return EmptyAutocast(), dtype

    return (
        torch.autocast(
            device_type=device, dtype=dtype, enabled=True, cache_enabled=cache_enabled
        ),
        dtype,
    )


def cast_network(network: torch.nn.Module, dtype: torch.dtype):
    """
    Cast network to given dtype

    Args:
        network (torch.nn.Module): Network to be casted
        dtype (torch.dtype): Dtype to cast network to
    """
    if dtype == torch.float16:
        network.half()
    elif dtype == torch.bfloat16:
        network.bfloat16()
    elif dtype == torch.float32:
        network.float()
    else:
        raise ValueError(f"Unknown dtype {dtype}")


def fix_seed(seed: int = 42):
    """
    Sets fixed random seed

    Args:
        seed (int, default=42): Random seed to be set
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False
    return True


def suppress_warnings():
    # Suppress PyTorch 1.11.0 warning associated with changing order of args in nn.MaxPool2d layer,
    # since source code is not affected by this issue and there aren't any other correct way to hide this message.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Note that order of the arguments: ceil_mode and "
        "return_indices will changeto match the args list "
        "in nn.MaxPool2d in a future release.",
        module="torch",
    )
