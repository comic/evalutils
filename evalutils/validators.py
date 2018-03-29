# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from pathlib import Path

from pandas import DataFrame

from evalutils.exceptions import ValidationError
from evalutils.io import first_int_in_filename_key


class DataFrameValidator(ABC):
    @abstractmethod
    def validate(self, *, df: DataFrame):
        """ Validates a single aspect of a dataframe

        Parameters
        ----------
        df
            The dataframe to be validated

        Returns
        -------
            None if the dataframe is valid

        Raises
        ------
        ValidationError
            If the dataframe is not valid

        """
        raise ValidationError


class UniquePathIndicesValidator(DataFrameValidator):
    def validate(self, *, df: DataFrame):
        try:
            paths = df['path']
        except KeyError:
            raise ValidationError('Column `path` not found in dataframe.')

        idx = [first_int_in_filename_key(Path(p)) for p in paths]

        if len(set(idx)) != len(paths):
            raise ValidationError(
                'The first number is each filename is not unique, please '
                'check that your files are named correctly.'
            )


class UniqueImagesValidator(DataFrameValidator):
    pass


class ExpectedColumnNamesValidator(DataFrameValidator):
    pass


class NumberOfCasesValidator(DataFrameValidator):
    pass
