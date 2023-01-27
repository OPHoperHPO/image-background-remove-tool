"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool

Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].

License: Apache License 2.0
"""
from pathlib import Path
from PIL import Image
import warnings
from typing import Optional


def save_file(output: Optional[Path], input_path: Path, image: Image.Image):
    """
    Saves an image to the file system

    Args:
        output (Optional[pathlib.Path]): Output path [dir or end file]
        input_path (pathlib.Path): Input path of the image
        image (Image.Image): Image to be saved.
    """
    if isinstance(output, Path) and str(output) != "none":
        if output.is_dir() and output.exists():
            image.save(output.joinpath(input_path.with_suffix(".png").name))
        elif output.suffix != "":
            if output.suffix != ".png":
                warnings.warn(
                    f"Only export with .png extension is supported! Your {output.suffix}"
                    f" extension will be ignored and replaced with .png!"
                )
            image.save(output.with_suffix(".png"))
        else:
            raise ValueError("Wrong output path!")
    elif output is None or str(output) == "none":
        image.save(
            input_path.with_name(
                input_path.stem.split(".")[0] + "_bg_removed"
            ).with_suffix(".png")
        )
