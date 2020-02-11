from pathlib import Path

import pytest
from pandas import Series

from evalutils import ClassificationEvaluation
from evalutils.exceptions import ConfigurationError, FileLoaderError
from evalutils.io import CSVLoader
from evalutils.validators import ExpectedColumnNamesValidator


class ClassificationTestEval(ClassificationEvaluation):
    def __init__(self, outdir):
        super().__init__(
            file_loader=CSVLoader(),
            ground_truth_path=(
                Path(__file__).parent
                / "resources"
                / "classification"
                / "reference"
            ),
            predictions_path=(
                Path(__file__).parent
                / "resources"
                / "classification"
                / "submission"
            ),
            output_file=Path(outdir) / "metrics.json",
            join_key="case",
            validators=(
                ExpectedColumnNamesValidator(expected=("case", "class")),
            ),
        )


def test_class_creation(tmpdir):
    ClassificationTestEval(outdir=tmpdir).evaluate()


def test_csv_with_no_join():
    class C(ClassificationEvaluation):
        def __init__(self, **kwargs):
            super().__init__(
                ground_truth_path=Path("/tmp"),
                validators=(),
                file_loader=CSVLoader(),
                **kwargs,
            )

    with pytest.raises(ConfigurationError):
        C()

    # Check that it's ok when we define it
    C(join_key="test")


def test_wrong_loader():
    class C(ClassificationEvaluation):
        def __init__(self):
            super().__init__(
                file_loader=CSVLoader(),
                ground_truth_path=(
                    Path(__file__).parent / "resources" / "itk"
                ),
                join_key="test",
                validators=(),
            )

    with pytest.raises(FileLoaderError):
        C().evaluate()


def test_series_aggregation(tmpdir):
    class C(ClassificationTestEval):
        def score_case(self, *, idx: int, case: Series):
            return {
                "accuracy": 1.0
                if case["class_ground_truth"] == case["class_prediction"]
                else 0.0,
                "case_id": str(idx),
            }

    e = C(outdir=tmpdir)
    e.evaluate()

    assert e._metrics["aggregates"]["accuracy"]["mean"] == 0.5
    assert len(e._metrics) == 2
    assert len(e._metrics["case"]["accuracy"]) == 8
    assert len(e._metrics["aggregates"]) == 2
    assert len(e._metrics["aggregates"]["accuracy"]) == 8
