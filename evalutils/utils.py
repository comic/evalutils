from collections import namedtuple
from typing import List, Tuple

from sklearn.neighbors import BallTree

DetectionScore = namedtuple(
    "DetectionScore", ["true_positives", "false_negatives", "false_positives"],
)


def score_detection(
    *,
    ground_truth: List[Tuple[float, ...]],
    predictions: List[Tuple[float, ...]],
    radius: float = 1.0,
) -> DetectionScore:
    """
    Generates the number of true positives, false positives and false negatives
    for the ground truth points given the predicted points.

    If multiple predicted points hit one ground truth point then this is
    considered as 1 true positive, and 0 false negatives.

    If one predicted point is a hit for N ground truth points then this is
    considered as 1 true positive, and N-1 false negatives.

    Parameters
    ----------
    ground_truth
        A list of the ground truth points
    predictions
        A list of the predicted points
    radius
        The maximum distance that two points can be separated by in order to
        be considered a hit

    Returns
    -------

    A tuple containing the number of true positives, false positives and
    false negatives.

    """
    hits_for_targets = find_hits_for_targets(
        targets=ground_truth, predictions=predictions, radius=radius,
    )

    true_positives = 0
    false_negatives = 0
    prediction_hit_a_target = [False] * len(predictions)

    for hits_for_target in hits_for_targets:
        if len(hits_for_target) > 0:
            true_positives += 1
            for idx in hits_for_target:
                prediction_hit_a_target[idx] = True
        else:
            false_negatives += 1

    false_positives = prediction_hit_a_target.count(False)

    if (true_positives + false_positives) != len(predictions):
        # A predicted point could be counted as a hit for many ground truth
        # points, so correct this if this happens
        double_counted_points = (
            (true_positives + false_positives) - len(predictions)
        )
        true_positives -= double_counted_points
        false_negatives += double_counted_points

    assert true_positives + false_negatives == len(ground_truth)
    assert true_positives + false_positives == len(predictions)

    return DetectionScore(
        true_positives=true_positives,
        false_negatives=false_negatives,
        false_positives=false_positives,
    )


def find_hits_for_targets(
    *,
    targets: List[Tuple[float, ...]],
    predictions: List[Tuple[float, ...]],
    radius: float,
) -> List[Tuple[int, ...]]:
    """
    Generates a list of the predicted points that are within a radius r of the
    targets.

    Parameters
    ----------
    targets
        A list of target points
    predictions
        A list of predicted points
    radius
        The maximum distance that two points can be apart for them to be
        considered a hit

    Returns
    -------

    A list which has the same length as the targets list. Each element within
    this list contains another list that contains the indicies of the
    predictions that are considered hits.

    """
    predictions_tree = BallTree(predictions)
    hits = predictions_tree.query_radius(X=targets, r=radius)
    return hits
