#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import pytest
from pandas import Series

from evalutils import Evaluation
from evalutils.exceptions import ConfigurationError, FileLoaderError
from evalutils.io import CSVLoader
from evalutils.validators import ExpectedColumnNamesValidator


class TestEval(Evaluation):
    def __init__(self):
        super().__init__(
            file_loader=CSVLoader(),
            ground_truth_path=(
                Path(__file__).parent / 'resources' / 'reference'
            ),
            predictions_path=(
                Path(__file__).parent / 'resources' / 'submission'
            ),
            output_file=Path('/tmp/metrics.json'),
            join_key='case',
            validators=(
                ExpectedColumnNamesValidator(expected=('case', 'class',)),
            ),
        )


def test_class_creation():
    TestEval().evaluate()


def test_csv_with_no_join():
    class C(Evaluation):
        def __init__(self, **kwargs):
            super().__init__(
                ground_truth_path=Path('/tmp'),
                validators=(),
                file_loader=CSVLoader(),
                **kwargs,
            )

    with pytest.raises(ConfigurationError):
        C()

    # Check that it's ok when we define it
    C(join_key='test')


def test_wrong_loader():
    class C(Evaluation):
        def __init__(self):
            super().__init__(
                file_loader=CSVLoader(),
                ground_truth_path=(
                    Path(__file__).parent / 'resources' / 'itk'
                ),
                join_key='test',
                validators=(),
            )

    with pytest.raises(FileLoaderError):
        C().evaluate()


def test_series_aggregation():
    class C(TestEval):
        @staticmethod
        def score_case(*, idx: int, case: Series):
            return {
                "accuracy": 1.0 if case["class_ground_truth"] == case[
                    "class_prediction"] else 0.0,
            }

    e = C()
    e.evaluate()

    assert e._metrics["aggregates"]["accuracy"]["mean"] == 0.5
    assert len(e._metrics) == 2
    assert len(e._metrics["case"]["accuracy"]) == 8
    assert len(e._metrics["aggregates"]) == 1
    assert len(e._metrics["aggregates"]["accuracy"]) == 8
