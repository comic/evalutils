import pytest

from evalutils.annotations import BoundingBox


@pytest.mark.parametrize(
    "x1,x2,y1,y2,expected",
    [
        (10, 20, 30, 40, 100),
        (20, 10, 30, 40, 100),
        (20, 10, 40, 30, 100),
        (-10, -20, -30, -40, 100),
        (10, 11, 30, 40, 10),
    ],
)
def test_bbox_area(x1, x2, y1, y2, expected):
    b = BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2)
    assert b.area == expected


def test_invalid_bbox():
    with pytest.raises(ValueError):
        BoundingBox(x1=10.0, x2=10.0, y1=10.0, y2=11.0)

    with pytest.raises(ValueError):
        BoundingBox(x1=10.0, x2=11.0, y1=10.0, y2=10.0)


@pytest.mark.parametrize(
    "bb1,bb2,expected",
    [
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            100,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=5, x2=15, y1=30, y2=40),
            50,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=10, y1=30, y2=40),
            0,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=9, y1=30, y2=40),
            0,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=11, y1=30, y2=40),
            10,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=11, y1=35, y2=40),
            5,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=11, y1=40, y2=41),
            0,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=-10, x2=-20, y1=30, y2=40),
            0,
        ),
    ],
)
def test_bbox_intersection(bb1, bb2, expected):
    assert bb1.intersection(other=bb2) == expected
    assert bb2.intersection(other=bb1) == expected


@pytest.mark.parametrize(
    "bb1,bb2,expected",
    [
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            100,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=5, x2=15, y1=30, y2=40),
            150,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=10, y1=30, y2=40),
            200,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=9, y1=30, y2=40),
            190,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=11, y1=30, y2=40),
            200,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=11, y1=35, y2=40),
            150,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=11, y1=40, y2=41),
            111,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=-10, x2=-20, y1=30, y2=40),
            200,
        ),
    ],
)
def test_bbox_union(bb1, bb2, expected):
    assert bb1.union(other=bb2) == expected
    assert bb2.union(other=bb1) == expected


@pytest.mark.parametrize(
    "bb1,bb2,expected",
    [
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            1,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=5, x2=15, y1=30, y2=40),
            1 / 3,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=10, y1=30, y2=40),
            0,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=9, y1=30, y2=40),
            0,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=11, y1=30, y2=40),
            0.05,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=11, y1=35, y2=40),
            5 / 150,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=0, x2=11, y1=40, y2=41),
            0,
        ),
        (
            BoundingBox(x1=10, x2=20, y1=30, y2=40),
            BoundingBox(x1=-10, x2=-20, y1=30, y2=40),
            0,
        ),
    ],
)
def test_bbox_jaccard(bb1, bb2, expected):
    assert bb1.jaccard_index(other=bb2) == expected
    assert bb2.jaccard_index(other=bb1) == expected


def test_eq():
    bb1 = BoundingBox(x1=10, x2=20, y1=30, y2=40)
    bb2 = BoundingBox(x1=10, x2=20, y1=30, y2=40)

    assert bb1 == bb2
    assert bb1 != BoundingBox(x1=11, x2=20, y1=30, y2=40)
