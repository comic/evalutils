# -*- coding: utf-8 -*-
from pathlib import Path

import pytest
from SimpleITK import GetArrayFromImage

from evalutils.exceptions import FileLoaderError
from evalutils.io import (
    get_first_int_in,
    ImageIOLoader,
    CSVLoader,
    SimpleITKLoader,
    first_int_in_filename_key
)


@pytest.mark.parametrize("filename,expected", [
    ('1.csv', 1),
    ('02.csv', 2),
    ('03_dfsdfs.csv', 3),
    ('04_04_fdsa.csv', 4),
    ('00005.00004.0003.fgdgdf.432.csv', 5000040003),
    ('1 copy (1).csv', 1),
])
def test_get_first_int_in(filename, expected):
    assert get_first_int_in(filename) == expected


def test_first_int_in_filename_key():
    fname = Path(__file__).parent / '22' / '1_mask.png'
    assert first_int_in_filename_key(fname) == f'{1:>64}'
    fname = Path(__file__).parent / '22'/ 'nonumberhere.png'
    assert first_int_in_filename_key(fname) == 'nonumberhere'


def test_image_io_loader():
    fname = Path(__file__).parent / 'resources' / 'images' / '1_mask.png'
    o = ImageIOLoader.load(fname=fname)
    assert o['path'] == fname
    assert o['img'].shape == (584, 565)

    with pytest.raises(FileLoaderError):
        non_image_fname = (
            Path(__file__).parent /
            'resources' /
            'csv' /
            'algorithm_result.csv'
        )
        ImageIOLoader.load(fname=non_image_fname)


def test_csv_loader():
    fname = (
        Path(__file__).parent /
        'resources' /
        'csv' /
        'algorithm_result.csv'
    )
    o = CSVLoader.load(fname=fname)

    assert set(o.keys()) == {'file_id', 'label'}
    assert len(o['file_id']) == len(o['label']) == 13201

    with pytest.raises(FileLoaderError):
        non_csv_filename = (
            Path(__file__).parent / 'resources' / 'images' / '1_mask.png'
        )
        CSVLoader.load(fname=non_csv_filename)


def test_itk_loader():
    fname = (
        Path(__file__).parent /
        'resources' /
        'itk' /
        '1.0.000.000000.0.00.0.0000000000.0000.0000000000.000.mhd'
    )
    o = SimpleITKLoader.load(fname=fname)

    assert o['path'] == fname
    assert GetArrayFromImage(o['img']).shape == (476, 512, 512)

    with pytest.raises(FileLoaderError):
        SimpleITKLoader.load(fname=fname.with_name(f'{fname.stem}.zraw'))
