# -*- coding: utf-8 -*-
from pathlib import Path

import pytest
from pandas import DataFrame

from evalutils.exceptions import ValidationError
from evalutils.validators import UniquePathIndicesValidator


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
