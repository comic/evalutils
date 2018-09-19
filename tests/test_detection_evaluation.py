from pathlib import Path

from evalutils import DetectionEvaluation
from evalutils.io import CSVLoader
from evalutils.validators import ExpectedColumnNamesValidator


class DetectionEvaluationTest(DetectionEvaluation):
    def __init__(self, outdir):
        super().__init__(
            file_loader=CSVLoader(),
            validators=(
                ExpectedColumnNamesValidator(
                    expected=("image_id", "x", "y", "score")
                ),
            ),
            join_key="image_id",
            ground_truth_path=(
                Path(__file__).parent / "resources" / "detection" / "reference"
            ),
            predictions_path=(
                Path(__file__).parent
                / "resources"
                / "detection"
                / "submission"
            ),
            output_file=Path(outdir) / "metrics.json",
            detection_threshold=0.5,
            detection_radius=1.0,
        )

    def get_points(self, *, case, key: str):
        try:
            points = case.loc[key]
        except KeyError:
            return []

        return [
            (p["x"], p["y"])
            for _, p in points.iterrows()
            if p["score"] > self._detection_threshold
        ]


def test_detection_evaluation(tmpdir):
    ev = DetectionEvaluationTest(outdir=tmpdir)
    ev.evaluate()

    assert len(ev._cases) == 21
    assert ev._metrics["aggregates"]["precision"] == 0.25
    assert ev._metrics["aggregates"]["recall"] == 1 / 3
    assert ev._metrics["aggregates"]["f1_score"] == 4 / 14
