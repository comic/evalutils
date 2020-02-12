import os
from distutils.util import strtobool
from pathlib import Path

import pytest
from SimpleITK import GetArrayFromImage

from evalutils.exceptions import FileLoaderError
from evalutils.io import (
    CSVLoader,
    ImageIOLoader,
    SimpleITKLoader,
    first_int_in_filename_key,
    get_first_int_in,
)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("1.csv", 1),
        ("02.csv", 2),
        ("03_dfsdfs.csv", 3),
        ("04_04_fdsa.csv", 4),
        ("00005.00004.0003.fgdgdf.432.csv", 5000040003),
        ("1 copy (1).csv", 1),
    ],
)
def test_get_first_int_in(filename, expected):
    assert get_first_int_in(filename) == expected


def test_first_int_in_filename_key():
    fname = Path(__file__).parent / "22" / "1_mask.png"
    assert first_int_in_filename_key(fname) == f"{1:>64}"
    fname = Path(__file__).parent / "22" / "nonumberhere.png"
    assert first_int_in_filename_key(fname) == "nonumberhere"


def test_image_io_loader():
    fname = Path(__file__).parent / "resources" / "images" / "1_mask.png"
    loader = ImageIOLoader()
    o = loader.load(fname=fname)[0]
    img = loader.load_image(fname)
    assert o["path"] == fname
    assert o["hash"] == loader.hash_image(img)
    assert img.shape == (584, 565)

    with pytest.raises(FileLoaderError):
        non_image_fname = (
            Path(__file__).parent
            / "resources"
            / "csv"
            / "algorithm_result.csv"
        )
        ImageIOLoader().load(fname=non_image_fname)


def test_csv_loader():
    fname = (
        Path(__file__).parent / "resources" / "csv" / "algorithm_result.csv"
    )
    records = CSVLoader().load(fname=fname)

    for record in records:
        assert set(record.keys()) == {"file_id", "label"}

    assert len(records) == 13201

    with pytest.raises(FileLoaderError):
        non_csv_filename = (
            Path(__file__).parent / "resources" / "images" / "1_mask.png"
        )
        CSVLoader().load(fname=non_csv_filename)


@pytest.mark.skipif(
    strtobool(os.environ.get("APPVEYOR", "False").lower()),
    reason="This test is not supported by standard appveyor",
)
def test_itk_loader():
    fname = (
        Path(__file__).parent
        / "resources"
        / "itk"
        / "1.0.000.000000.0.00.0.0000000000.0000.0000000000.000.mhd"
    )
    loader = SimpleITKLoader()
    o = loader.load(fname=fname)[0]
    img = loader.load_image(fname)

    assert o["path"] == fname
    assert o["hash"] == loader.hash_image(img)
    assert GetArrayFromImage(img).shape == (476, 512, 512)

    with pytest.raises(FileLoaderError):
        SimpleITKLoader().load(fname=fname.with_name(f"{fname.stem}.zraw"))
