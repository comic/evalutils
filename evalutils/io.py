import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union

from SimpleITK import GetArrayFromImage, ReadImage
from imageio import imread
from pandas import read_csv
from pandas.errors import EmptyDataError, ParserError

from .exceptions import FileLoaderError

logger = logging.getLogger(__name__)


def get_first_int_in(s: str) -> Union[int, str]:
    """
    Gets the first integer in a string.

    Parameters
    ----------
    s
        The string to search for an int

    Returns
    -------
        The first integer found in the string

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
        logger.warning(f"Could not find an int in the string '{fname.stem}'.")
        return fname.stem


class FileLoader(ABC):
    @abstractmethod
    def load(self, *, fname: Path) -> List[Dict]:
        """
        Tries to load the file given by the path fname.

        Notes
        -----
        For this to work with the validators you must:

            If you load an image it must save the hash in the `hash` column

            If you reference a Path it must be saved in the `path` column


        Parameters
        ----------
        fname
            The file that the loader will try to load

        Returns
        -------
            A list containing all of the cases in this file

        Raises
        ------
        FileLoaderError
            If a file cannot be loaded as the specified type

        """
        raise FileLoaderError


class ImageLoader(FileLoader):
    """
    A specialised file loader for images. As images are large they will not
    all be loaded into memory, so score_case needs to load them again later
    via load_image.
    """

    def load(self, *, fname: Path):
        try:
            img = self.load_image(fname)
        except (ValueError, RuntimeError):
            raise FileLoaderError(
                f"Could not load {fname} using {self.__class__.__qualname__}."
            )
        return [{"hash": self.hash_image(img), "path": fname}]

    @staticmethod
    def load_image(fname: Path):
        """
        Loads the image

        Parameters
        ----------
        fname
            The path that the loader will try to load

        Returns
        -------
            The image
        """
        raise NotImplementedError

    @staticmethod
    def hash_image(image) -> int:
        """
        Generates a hash of the image

        Parameters
        ----------
        image
            The image to hash

        Returns
        -------
            The hash of the image

        """
        raise NotImplementedError


class ImageIOLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        return imread(fname, as_gray=True)

    @staticmethod
    def hash_image(image):
        return hash(image.tostring())


class SimpleITKLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        return ReadImage(str(fname))

    @staticmethod
    def hash_image(image):
        return hash(GetArrayFromImage(image).tostring())


class CSVLoader(FileLoader):
    def load(self, *, fname: Path):
        try:
            return read_csv(fname, skipinitialspace=True).to_dict(
                orient="records"
            )
        except UnicodeDecodeError:
            raise FileLoaderError(f"Could not load {fname} using {__name__}.")
        except (ParserError, EmptyDataError):
            raise ValueError(
                f"CSV file could not be loaded: we could not load "
                f"{fname.name} using `pandas.read_csv`."
            )
