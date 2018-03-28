# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

from pandas import DataFrame

from evalutils.exceptions import ValidationError


class DataFrameValidator(ABC):
    @abstractmethod
    def validate(self, df: DataFrame):
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

class UniqueFileIndicesValidator(DataFrameValidator):
    pass

class UniqueImagesValidator(DataFrameValidator):
    pass

class ExpectedColumnNamesValidator(DataFrameValidator):
    pass

class NumberOfCasesValidator(DataFrameValidator):
    pass
