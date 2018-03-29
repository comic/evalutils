# -*- coding: utf-8 -*-
from pathlib import Path

import pytest
from pandas import DataFrame

from evalutils.exceptions import ValidationError
from evalutils.validators import (
    UniquePathIndicesValidator, ExpectedColumnNamesValidator,
)


def test_path_indicies_validator_no_column():
    df = DataFrame()
    with pytest.raises(ValidationError):
        UniquePathIndicesValidator().validate(df=df)


def test_path_indicies_vaidator_duplicate_indicies():
    df = DataFrame({'path': ['01.foo', '1.bar', '03.baz']})
    with pytest.raises(ValidationError):
        UniquePathIndicesValidator().validate(df=df)


def test_path_indicies_validator_ok():
    df = DataFrame({'path': [Path('01.jpg'), Path('11.jpg')]})
    UniquePathIndicesValidator().validate(df=df)


def test_expected_columns_creation():
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        ExpectedColumnNamesValidator(expected=())


def test_expected_columns_one_undefined():
    validator = ExpectedColumnNamesValidator(expected=('foo', 'bar',))
    df = DataFrame(columns=['foo'])
    with pytest.raises(ValidationError):
        validator.validate(df=df)


def test_expected_columns_overdefined():
    validator = ExpectedColumnNamesValidator(expected=('foo', 'bar',))
    df = DataFrame(columns=['foo', 'bar', 'baz'])
    with pytest.raises(ValidationError):
        validator.validate(df=df)


def test_expected_columns_ok():
    validator = ExpectedColumnNamesValidator(expected=('foo', 'bar',))
    df = DataFrame(columns=['foo', 'bar'])
    validator.validate(df=df)
