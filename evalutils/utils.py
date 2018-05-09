from typing import List

from sklearn.neighbors import BallTree

from evalutils.annotations import Point


def merge_points(*, points: List[Point], radius=1.0):

    positions = [p.pos for p in points]

    d1 = BallTree(positions)
    merge_indicies = d1.query_radius(X=positions, r=radius)

    merged = [False] * len(points)
    out = []

    for idx, neighbours in enumerate(merge_indicies):
        if merged[idx]:
            continue
        else:
            p = points[neighbours[0]]
            merged[idx] = True

            for i in neighbours[1:]:
                if not merged[i]:
                    p += points[i]
                    merged[i] = True

            out.append(p)

    return out
