from csv import DictReader
from pathlib import Path

from evalutils.annotations import Point
from evalutils.utils import merge_points


def test_point_merging():

    p1 = Point(pos=[0,0], score=0)
    p2 = Point(pos=[1,1], score=1)

    p1 += p2

    assert list(p1.pos) == [0.5, 0.5]
    assert p1.score == 0.5

    p3 = p1 + p2

    assert list(p1.pos) == [0.5, 0.5]
    assert p1.score == 0.5
    assert list(p3.pos) == [0.75, 0.75]

    with open(Path(__file__).parent / 'resources/points/predictions.csv', 'r') as f:
        predictions = DictReader(f, skipinitialspace=True)
        points = [Point(pos=[int(p['x']), int(p['y'])]) for p in predictions]

    idx = merge_points(points=points)

    assert len(idx) == 0
