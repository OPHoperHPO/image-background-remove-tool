import re
from typing import Optional

from pydantic import BaseModel, field_validator
from typing_extensions import Literal


class Parameters(BaseModel):
    image_file_b64: Optional[str] = ""
    image_url: Optional[str] = ""
    size: Optional[Literal["preview", "full", "auto"]] = "preview"
    type: Optional[
        Literal["auto", "product", "person", "car"]
    ] = "auto"  # Not supported at the moment
    format: Optional[Literal["auto", "jpg", "png", "zip"]] = "auto"
    roi: str = "0% 0% 100% 100%"
    crop: bool = False
    crop_margin: Optional[str] = "0px"
    scale: Optional[str] = "original"
    position: Optional[str] = "original"
    channels: Optional[Literal["rgba", "alpha"]] = "rgba"
    add_shadow: str = "false"  # Not supported at the moment
    semitransparency: str = "false"  # Not supported at the moment
    bg_color: Optional[str] = ""
    bg_image_url: Optional[str] = ""

    @field_validator("crop_margin")
    def crop_margin_validator(cls, value: Optional[str]):
        if not re.match(r"[0-9]+(px|%)$", value):
            raise ValueError(
                "crop_margin paramter is not valid"
            )  # TODO: Add support of several values
        if "%" in value and (int(value[:-1]) < 0 or int(value[:-1]) > 100):
            raise ValueError("crop_margin mast be in range between 0% and 100%")
        return value

    @field_validator("scale")
    def scale_validator(cls, value: Optional[str]):
        if value != "original" and (
            not re.match(r"[0-9]+%$", value)
            or not int(value[:-1]) <= 100
            or not int(value[:-1]) >= 10
        ):
            raise ValueError("scale must be original or in between of 10% and 100%")

        if value == "original":
            return 100

        return int(value[:-1])

    @field_validator("position")
    def position_validator(cls, value: Optional[str]):
        if len(value.split(" ")) > 2:
            raise ValueError(
                "Position must be a value from 0 to 100 "
                "for both vertical and horizontal axises or for both axises respectively"
            )

        if value == "original":
            return "original"
        elif len(value.split(" ")) == 1:
            return [int(value[:-1]), int(value[:-1])]
        else:
            return [int(value.split(" ")[0][:-1]), int(value.split(" ")[1][:-1])]

    @field_validator("bg_color")
    def bg_color_validator(cls, value: Optional[str]):
        if not re.match(r"(#?[0-9a-f]{3}){0,2}$", value):
            raise ValueError("bg_color is not in hex")
        if len(value) and value[0] != "#":
            value = "#" + value
        return value
