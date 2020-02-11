from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

from pandas import DataFrame

from .exceptions import ValidationError
from .io import first_int_in_filename_key


class DataFrameValidator(ABC):
    @abstractmethod
    def validate(self, *, df: DataFrame):
        """ Validates a single aspect of a DataFrame

        Parameters
        ----------
        df
            The DataFrame to be validated

        Returns
        -------
            None if the DataFrame is valid

        Raises
        ------
        ValidationError
            If the DataFrame is not valid

        """
        raise ValidationError


class UniquePathIndicesValidator(DataFrameValidator):
    """
    Validates that the indicies from the filenames are unique
    """

    def validate(self, *, df: DataFrame):
        try:
            paths = df["path"]
        except KeyError:
            raise ValidationError("Column `path` not found in DataFrame.")

        idx = [first_int_in_filename_key(Path(p)) for p in paths]

        if len(set(idx)) != len(paths):
            raise ValidationError(
                "The first number is each filename is not unique, please "
                "check that your files are named correctly."
            )


class UniqueImagesValidator(DataFrameValidator):
    """
    Validates that each image in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["hash"]
        except KeyError:
            raise ValidationError("Column `hash` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The images are not unique, please submit a unique image for "
                "each case."
            )


class ExpectedColumnNamesValidator(DataFrameValidator):
    def __init__(
        self, *, expected: Tuple[str, ...], extra_cols_check: bool = True
    ):
        """
        Validates that the DataFrame has the expected columns

        Parameters
        ----------
        expected
            The expected columns in the DataFrame
        extra_cols_check
            Perform the check for extra columns, default is true but you may
            want to disable this if you're sure that extra columns can be
            ignored.

        Raises
        ------
        ValueError
            If no columns are defined

        """
        if len(expected) == 0:
            raise ValueError(
                "You must define what columns you expect to find in the "
                f"DataFrame in order to use {self.__class__.__name__}."
            )

        self._expected = expected
        self._extra_cols_check = extra_cols_check
        super().__init__()

    def validate(self, *, df: DataFrame):

        undefined_cols = [c for c in self._expected if c not in df.columns]

        if undefined_cols:
            raise ValidationError(
                f"We expected to find the following columns but we didn't: "
                f"{undefined_cols}. Please check the column labels, and "
                f"note that this is case sensitive. We only found: "
                f"{df.columns}."
            )

        extra_cols = [c for c in df.columns if c not in self._expected]

        if self._extra_cols_check and extra_cols:
            raise ValidationError(
                f"We only expected to find the columns {self._expected}. "
                f"However, we also found that extra columns were defined: "
                f"{extra_cols}. Please remove them."
            )


class NumberOfCasesValidator(DataFrameValidator):
    def __init__(self, *, num_cases: int):
        """
        Validates that there are the correct number of cases in the set.

        Parameters
        ----------
        num_cases
            The number of cases that we expect to find.
        """
        if num_cases <= 0:
            raise ValueError(
                "The expected number of cases must be greater than zero in "
                f"{self.__class__.__name__}."
            )

        self._num_cases = num_cases
        super().__init__()

    def validate(self, *, df: DataFrame):
        if len(df) != self._num_cases:
            raise ValidationError(
                f"We expected to find {self._num_cases}, but we found "
                f"{len(df)}. Please correct the number of predictions."
            )
