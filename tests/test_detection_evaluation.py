from pathlib import Path

from pandas import DataFrame

from evalutils import DetectionEvaluation
from evalutils.io import CSVLoader
from evalutils.utils import score_detection
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
        self.detection_threshold = 0.5
        self.detection_radius = 1.0

    def score_case(self, *, idx: int, case: DataFrame):
        ground_truth = case.loc['ground_truth']
        predictions = case.loc['predictions']

        score = score_detection(
            ground_truth=[(p['x'], p['y']) for _, p in ground_truth.iterrows()
                          if p['score'] > self.detection_threshold],
            predictions=[(p['x'], p['y']) for _, p in predictions.iterrows()
                         if p['score'] > self.detection_threshold],
            radius=self.detection_radius,
        )

        return score._asdict()


def test_class_merging():
    ev = DetectionEvaluationTest()
    ev.evaluate()

    assert len(ev._cases) == 21
    assert ev._metrics['aggregates']['precision'] == 0.25
    assert ev._metrics['aggregates']['recall'] == 1 / 3
    assert ev._metrics['aggregates']['f1_score'] == 4 / 14
