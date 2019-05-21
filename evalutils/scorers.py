from collections import namedtuple
from typing import List, Tuple

from numpy import array
from sklearn.neighbors import BallTree

DetectionScore = namedtuple(
    "DetectionScore", ["true_positives", "false_negatives", "false_positives"]
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

    if len(ground_truth) == 0:
        return DetectionScore(
            true_positives=0,
            false_negatives=0,
            false_positives=len(predictions),
        )
    elif len(predictions) == 0:
        return DetectionScore(
            true_positives=0,
            false_negatives=len(ground_truth),
            false_positives=0,
        )

    hits_for_targets = find_hits_for_targets(
        targets=ground_truth, predictions=predictions, radius=radius
    )

    true_positives = 0
    false_negatives = 0
    prediction_hit_a_target = [False] * len(predictions)

    for hits_for_target in hits_for_targets:
        for hit_idx in hits_for_target:
            # Go from the nearest to the farthest hit, mark the closest one
            # as a hit for this point
            if not prediction_hit_a_target[hit_idx]:
                prediction_hit_a_target[hit_idx] = True
                true_positives += 1
                break
        else:
            false_negatives += 1

    false_positives = prediction_hit_a_target.count(False)

    assert 0 <= true_positives <= min(len(predictions), len(ground_truth))
    assert 0 <= false_positives <= len(predictions)
    assert 0 <= false_negatives <= len(ground_truth)

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
    targets. The indicies are returned in sorted order, from closest to
    farthest point.

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
    predictions_tree = BallTree(array(predictions))
    hits, _ = predictions_tree.query_radius(
        X=targets, r=radius, return_distance=True, sort_results=True
    )
    return hits
