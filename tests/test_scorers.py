from csv import DictReader
from pathlib import Path

from evalutils.scorers import find_hits_for_targets, score_detection


def load_points_csv(filepath):
    with open(Path(__file__).parent / filepath, "r") as f:
        positions = DictReader(f, skipinitialspace=True)
        points = [(float(p["x"]), float(p["y"])) for p in positions]

    return points


def test_point_merging():
    predictions = load_points_csv("resources/points/predictions.csv")
    ground_truth = load_points_csv("resources/points/reference.csv")

    perfect_detection = score_detection(
        ground_truth=ground_truth, predictions=ground_truth
    )

    assert perfect_detection.true_positives == len(ground_truth)
    assert perfect_detection.false_positives == 0
    assert perfect_detection.false_negatives == 0

    detection_score = score_detection(
        ground_truth=ground_truth, predictions=predictions
    )

    assert detection_score.true_positives == 13
    assert detection_score.false_positives == 1452
    assert detection_score.false_negatives == 551


targets = [
    (10.0, 10.0),  # 1, TP+=1, FP-=1
    (5.0, 5.0),  # -, FN+=1
    (2.0, 2.0),  # 3, 2, TP+=1, FP-=1
    (11.2, 11.2),  # 1, already taken by 0
]

preds = [
    (0, 0),
    (10.5, 10.5),
    (1.0, 2.0),
    (2.0 + 0.5 ** 0.5, 2.0 + 0.5 ** 0.5),
    (30, 30),
]


def test_find_hits():
    neighbours = find_hits_for_targets(
        targets=targets, predictions=preds, radius=1.0
    )

    assert neighbours[0] == [1]
    assert list(neighbours[1]) == []
    assert list(neighbours[2]) == [3, 2]
    assert list(neighbours[3]) == [1]
    assert len(neighbours) == len(targets)


def test_score_detection():
    detection_score = score_detection(ground_truth=targets, predictions=preds)

    assert detection_score.false_negatives == 2
    assert detection_score.true_positives == 2
    assert detection_score.false_positives == 3


def test_multi_hit():
    detection_score = score_detection(
        ground_truth=[(0, 0), (0, 0.5), (0, -0.5)], predictions=[(0.0, 0.0)]
    )

    assert detection_score.false_positives == 0
    assert detection_score.true_positives == 1
    assert detection_score.false_negatives == 2

    detection_score = score_detection(
        ground_truth=[(0, 0), (0, 0.5), (0, -0.5)],
        predictions=[(0.0, 0.0), (0.0, 0.0)],
    )

    assert detection_score.false_positives == 0
    assert detection_score.true_positives == 2
    assert detection_score.false_negatives == 1
