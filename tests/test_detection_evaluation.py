from pathlib import Path

from evalutils import DetectionEvaluation
from evalutils.io import CSVLoader
from evalutils.validators import ExpectedColumnNamesValidator


class DetectionEvaluationTest(DetectionEvaluation):
    def __init__(self):
        super().__init__(
            file_loader=CSVLoader(),
            validators=(
                ExpectedColumnNamesValidator(
                    expected=('image_id', 'x', 'y', 'score')
                ),
            ),
            join_key='image_id',
            ground_truth_path=(
                Path(__file__).parent /
                'resources' /
                'detection' /
                'reference'
            ),
            predictions_path=(
                Path(__file__).parent /
                'resources' /
                'detection' /
                'submission'
            ),
            output_file=Path('/tmp/metrics.json'),
        )


def test_class_merging():
    ev = DetectionEvaluationTest()
    ev.evaluate()

    assert len(ev._cases) == 3

