# -*- coding: utf-8 -*-
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict

from SimpleITK import ReadImage
from imageio import imread
from pandas import read_csv
from pandas.errors import ParserError, EmptyDataError

from .exceptions import FileLoaderError

logger = logging.getLogger(__name__)


def get_first_int_in(s: str) -> Union[int, None]:
    """ TODO

    Parameters
    ----------
    s
        TODO

    Returns
    -------
        TODO

    Raises
    ------
    AttributeError
        If there is not an int contained in the string

    """
    r = re.compile(r"\D*((?:\d+\.?)+)\D*")
    m = r.search(s)

    return int(m.group(1).replace(".", ""))


def first_int_in_filename_key(fname: Path) -> str:
    try:
        return f"{get_first_int_in(fname.stem):>64}"
    except AttributeError:
        logger.warning(f"Could not find int in '{fname.stem}'.")
        return fname.stem


class FileLoader(ABC):
    @staticmethod
    @abstractmethod
    def load(*, fname: Path) -> Dict:
        """

        TODO

        Notes
        -----
        For this to work with the validators you must:

            If you load an image it must be saved in the `img` column

            If you reference a Path it must be saved in the `path` column


        Parameters
        ----------
        fname
            TODO

        Returns
        -------
            TODO

        Raises
        ------
        FileLoaderError
            If a file cannot be loaded as the specified type

        """
        raise FileLoaderError


class ImageIOLoader(FileLoader):
    @staticmethod
    def load(*, fname: Path) -> Dict:
        try:
            img = imread(fname, as_gray=True)
        except ValueError:
            raise FileLoaderError(f"Could not load {fname} using {__name__}.")
        return {
            "img": img,
            "path": fname,
        }


class SimpleITKLoader(FileLoader):
    @staticmethod
    def load(*, fname: Path) -> Dict:
        try:
            img = ReadImage(str(fname))
        except RuntimeError:
            raise FileLoaderError(f"Could not load {fname} using {__name__}.")
        return {
            "img": img,
            "path": fname,
        }


class CSVLoader(FileLoader):
    @staticmethod
    def load(*, fname: Path) -> Dict:
        try:
            return read_csv(fname, skipinitialspace=True).to_dict()
        except UnicodeDecodeError:
            raise FileLoaderError(f"Could not load {fname} using {__name__}.")
        except(ParserError, EmptyDataError):
            raise ValueError(
                f"CSV file could not be loaded: we could not load "
                f"{fname.name} using `pandas.read_csv`."
            )
