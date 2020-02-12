from pathlib import Path

import pytest
from pandas import DataFrame

from evalutils.exceptions import ValidationError
from evalutils.io import ImageIOLoader
from evalutils.validators import (
    ExpectedColumnNamesValidator,
    NumberOfCasesValidator,
    UniqueImagesValidator,
    UniquePathIndicesValidator,
)


def test_path_indicies_validator_no_column():
    df = DataFrame()
    with pytest.raises(ValidationError):
        UniquePathIndicesValidator().validate(df=df)


def test_path_indicies_vaidator_duplicate_indicies():
    df = DataFrame({"path": ["01.foo", "1.bar", "03.baz"]})
    with pytest.raises(ValidationError):
        UniquePathIndicesValidator().validate(df=df)


def test_path_indicies_validator_ok():
    df = DataFrame({"path": [Path("01.jpg"), Path("11.jpg")]})
    UniquePathIndicesValidator().validate(df=df)


def test_expected_columns_creation():
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        ExpectedColumnNamesValidator(expected=())


def test_expected_columns_one_undefined():
    validator = ExpectedColumnNamesValidator(expected=("foo", "bar"))
    df = DataFrame(columns=["foo"])
    with pytest.raises(ValidationError):
        validator.validate(df=df)


def test_expected_columns_overdefined():
    validator = ExpectedColumnNamesValidator(expected=("foo", "bar"))
    df = DataFrame(columns=["foo", "bar", "baz"])
    with pytest.raises(ValidationError):
        validator.validate(df=df)


def test_expected_columns_ok():
    validator = ExpectedColumnNamesValidator(expected=("foo", "bar"))
    df = DataFrame(columns=["foo", "bar"])
    validator.validate(df=df)


@pytest.fixture(scope="module")
def images():
    image1 = ImageIOLoader().load(
        fname=Path(__file__).parent / "resources" / "images" / "1_mask.png"
    )[0]
    image2 = ImageIOLoader().load(
        fname=Path(__file__).parent / "resources" / "images" / "2_mask.png"
    )[0]
    return [image1, image2]


def test_unique_images_no_img_col():
    df = DataFrame(columns=["foo"])
    with pytest.raises(ValidationError):
        UniqueImagesValidator().validate(df=df)


def test_unique_images_duplicate(images):
    df = DataFrame([images[0], images[0]])
    assert df.columns.all() in ["img", "path"]

    with pytest.raises(ValidationError):
        UniqueImagesValidator().validate(df=df)


def test_unique_images_ok(images):
    df = DataFrame(images)
    UniqueImagesValidator().validate(df=df)


def test_number_of_predictions_badly_configured():
    with pytest.raises(ValueError):
        NumberOfCasesValidator(num_cases=-32)


def test_number_of_predictions_wrong():
    df = DataFrame({"foo": ["bar"] * 1337})
    with pytest.raises(ValidationError):
        NumberOfCasesValidator(num_cases=1000).validate(df=df)
    with pytest.raises(ValidationError):
        NumberOfCasesValidator(num_cases=2000).validate(df=df)


def test_number_of_predictions_ok():
    df = DataFrame({"foo": ["bar"] * 10})
    NumberOfCasesValidator(num_cases=10).validate(df=df)
