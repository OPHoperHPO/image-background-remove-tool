"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import os
import pytest
from pathlib import Path
from carvekit.utils.download_models import sha512_checksum_calc
from carvekit.ml.files.models_loc import (
    u2net_full_pretrained,
    fba_pretrained,
    deeplab_pretrained,
    basnet_pretrained,
    download_all,
    checkpoints_dir,
    downloader,
    tracer_b7_pretrained,
    scene_classifier_pretrained,
)
from carvekit.utils.models_utils import fix_seed, suppress_warnings


def test_fix_seed():
    fix_seed(seed=42)


def test_suppress_warnings():
    suppress_warnings()


def test_download_all():
    download_all()


def test_download_model():
    hh = checkpoints_dir / "u2net-universal" / "u2net.pth"
    hh.write_text("1234")
    assert downloader("u2net.pth") == hh
    os.remove(hh)
    with pytest.raises(FileNotFoundError):
        downloader("NotExistedPath/2.dl")


def test_sha512():
    hh = checkpoints_dir / "basnet-universal" / "basnet.pth"
    hh.write_text("1234")
    assert (
        sha512_checksum_calc(hh)
        == "d404559f602eab6fd602ac7680dacbfaadd13630335e951f097a"
        "f3900e9de176b6db28512f2e000"
        "b9d04fba5133e8b1c6e8df59db3a8ab9d60be4b97cc9e81db"
    )


def test_check_model():
    invalid_hash_file = checkpoints_dir / "basnet-universal" / "basnet.pth"
    invalid_hash_file.write_text("1234")
    downloader("basnet.pth")
    assert (
        sha512_checksum_calc(invalid_hash_file)
        != "d404559f602eab6fd602ac7680dacbfaadd13630335e951f097a"
        "f3900e9de176b6db28512f2e000"
        "b9d04fba5133e8b1c6e8df59db3a8ab9d60be4b97cc9e81db"
    )


def test_check_for_exists():
    assert u2net_full_pretrained().exists()
    assert fba_pretrained().exists()
    assert deeplab_pretrained().exists()
    assert basnet_pretrained().exists()
    assert tracer_b7_pretrained().exists()
    assert scene_classifier_pretrained().exists()
