import SimpleITK
import numpy as np
import pytest
import scipy.ndimage as ndimage
import sklearn
from numpy.testing import assert_array_almost_equal
from scipy.ndimage.morphology import generate_binary_structure

import evalutils.stats as stats


@pytest.fixture(autouse=True)
def reset_seeds():
    np.random.seed(42)
    yield


@pytest.mark.parametrize(
    "y_true", [np.random.randint(0, 2, (30, 20, 10)).astype(np.int)]
)
@pytest.mark.parametrize(
    "y_pred", [np.random.randint(0, 2, (30, 20, 10)).astype(np.int)]
)
@pytest.mark.parametrize("labels", [[0], [0, 1], [0, 2], [0, 1, 2]])
def test_calculate_confusion_matrix(y_true, y_pred, labels):
    result = stats.calculate_confusion_matrix(y_true, y_pred, labels)
    result2 = sklearn.metrics.confusion_matrix(
        y_true.flatten(), y_pred.flatten(), labels
    )
    assert result.shape[0] == len(labels) and result.shape[1] == len(labels)
    assert np.equal(result, result2).all()


def test_accuracies_from_cm():
    cm = np.array([[5, 2, 1], [1, 2, 1], [1, 0, 4]])
    expected = np.array(
        [5 + 2 + 1 + 4, 2 + 5 + 4 + 1 + 1, 4 + 2 + 1 + 2 + 5], dtype=np.float
    ) // float(np.sum(cm))
    accs = stats.accuracies_from_confusion_matrix(cm)
    assert np.equal(expected, accs).all()


def test_jaccard_from_cm():
    cm = np.array([[5, 2, 1], [1, 2, 1], [1, 0, 4]])
    expected = np.array([5, 2, 4], dtype=np.float) / np.array(
        [2 + 1 + 1 + 1 + 5, 2 + 1 + 1 + 2, 4 + 1 + 1 + 1], dtype=np.float
    )
    accs = stats.jaccard_from_confusion_matrix(cm)
    assert np.allclose(expected, accs)


def test_dice_from_cm():
    cm = np.array([[5, 2, 1], [1, 2, 1], [1, 0, 4]])
    expected = (
        np.array([5, 2, 4], dtype=np.float)
        * 2
        / np.array(
            [2 + 1 + 1 + 1 + 5 + 5, 2 + 1 + 1 + 2 + 2, 4 + 4 + 1 + 1 + 1],
            dtype=np.float,
        )
    )
    accs = stats.dice_from_confusion_matrix(cm)
    assert np.allclose(expected, accs)


def test_ravd():
    a = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=np.bool)
    b = np.array([[1, 0, 1], [1, 0, 1], [0, 0, 1]], dtype=np.bool)
    r1 = stats.relative_absolute_volume_difference(a, b)
    r2 = stats.relative_absolute_volume_difference(b, a)
    assert r1 != r2
    assert r1 == (6.0 - 5.0) / 6.0
    assert r2 == (6.0 - 5.0) / 5.0


@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
def test_avd(voxelspace):
    a = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=np.bool)
    b = np.array([[1, 0, 1], [1, 0, 1], [0, 0, 1]], dtype=np.bool)
    r1 = stats.absolute_volume_difference(a, b, voxelspace)
    r2 = stats.absolute_volume_difference(b, a, voxelspace)
    assert r1 == r2
    assert r1 == (6.0 - 5.0) * (
        1 if voxelspace is None else np.prod(voxelspace)
    )


# A connectivity of 1 was omitted, because the HausdorffDistanceImageFilter
# does not support it.
# A connectivity of 0 and 2 appear to achieve similar Hausdorff metrics,
# although only one is used within the HausdorffDistanceImageFilter
@pytest.mark.parametrize(
    "a,b",
    [
        [
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
        ]
        for _ in range(20)
    ]
    + [
        [
            np.array(
                [
                    [True, True, False, True, True, False],
                    [False, False, True, False, False, True],
                    [False, True, False, True, False, True],
                    [False, True, False, False, False, True],
                    [True, False, False, False, False, True],
                    [True, True, False, False, True, True],
                ]
            ),
            np.array(
                [
                    [False, False, True, False, True, False],
                    [True, False, True, False, False, True],
                    [False, False, False, True, False, True],
                    [True, True, True, True, True, False],
                    [False, True, True, True, False, False],
                    [True, False, True, False, True, False],
                ]
            ),
        ]
    ],
)
@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8), (12.0, 4.0)])
@pytest.mark.parametrize("connectivity", [0, 2])
def test_hd(a, b, voxelspace, connectivity):
    hd = stats.hausdorff_distance(a, b, voxelspace, connectivity=connectivity)
    hd2 = stats.hausdorff_distance(b, a, voxelspace, connectivity=connectivity)

    assert hd == hd2

    a2 = SimpleITK.GetImageFromArray(a.astype(np.uint8))
    b2 = SimpleITK.GetImageFromArray(b.astype(np.uint8))
    if voxelspace is None:
        voxelspace = [1, 1]
    a2.SetSpacing(voxelspace[::-1])
    b2.SetSpacing(voxelspace[::-1])
    hdfilter = SimpleITK.HausdorffDistanceImageFilter()
    hdfilter.Execute(a2, b2)
    hd3 = hdfilter.GetHausdorffDistance()
    hdfilter.Execute(b2, a2)
    hd4 = hdfilter.GetHausdorffDistance()

    assert np.isclose(hd, hd3)
    assert np.isclose(hd2, hd4)


@pytest.mark.parametrize(
    "a,b",
    [
        [
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
        ]
        for _ in range(20)
    ],
)
@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
@pytest.mark.parametrize("connectivity", [0, 1, 2])
def test_modified_hd(a, b, voxelspace, connectivity):
    d_a = sitk_surface_distance(a, b, voxelspace, connectivity)
    d_b = sitk_surface_distance(b, a, voxelspace, connectivity)
    mhd = max(d_a.mean(), d_b.mean())

    mhd2 = stats.modified_hausdorff_distance(a, b, voxelspace, connectivity)
    mhd3 = stats.modified_hausdorff_distance(b, a, voxelspace, connectivity)

    assert np.isclose(mhd, mhd2)
    assert np.isclose(mhd, mhd3)
    assert mhd2 == mhd3


@pytest.mark.parametrize(
    "a,b",
    [
        [
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
        ]
        for _ in range(20)
    ],
)
@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
@pytest.mark.parametrize("connectivity", [0, 1, 2])
@pytest.mark.parametrize("percentile", [0, 0.8, 0.95, 1])
def test_percentile_hd(a, b, voxelspace, connectivity, percentile):
    d_a = sitk_surface_distance(a, b, voxelspace, connectivity)
    d_b = sitk_surface_distance(b, a, voxelspace, connectivity)
    d_a.sort()
    d_b.sort()
    phd = max(
        d_a[int((len(d_a) - 1) * percentile)],
        d_b[int((len(d_b) - 1) * percentile)],
    )

    phd2 = stats.percentile_hausdorff_distance(
        a, b, percentile, voxelspace, connectivity
    )
    phd3 = stats.percentile_hausdorff_distance(
        b, a, percentile, voxelspace, connectivity
    )

    assert np.isclose(phd, phd2)
    assert np.isclose(phd, phd3)
    assert phd2 == phd3


@pytest.mark.parametrize(
    "a,b",
    [
        [
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
        ]
        for _ in range(20)
    ],
)
@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
def test_mean_contour_distance(a, b, voxelspace):
    d_a = sitk_directed_contour_distance(a, b, voxelspace)
    d_b = sitk_directed_contour_distance(b, a, voxelspace)
    mhd = max(d_a.mean(), d_b.mean())

    mhd2 = stats.mean_contour_distance(a, b, voxelspace)
    mhd3 = stats.mean_contour_distance(b, a, voxelspace)

    assert np.isclose(mhd, mhd2)
    assert np.isclose(mhd, mhd3)
    assert mhd2 == mhd3


def sitk_surface_distance(a, b, voxelspace, connectivity):
    a2 = SimpleITK.GetImageFromArray((a == 1).astype(np.uint8))
    b2 = SimpleITK.GetImageFromArray((b == 1).astype(np.uint8))
    if voxelspace is None:
        voxelspace = [1, 1]
    a2.SetSpacing(voxelspace[::-1])
    b2.SetSpacing(voxelspace[::-1])

    if connectivity > 0:
        eroder = SimpleITK.BinaryErodeImageFilter()
        eroder.SetBackgroundValue(0)
        eroder.SetKernelType(
            SimpleITK.sitkCross if connectivity <= 1 else SimpleITK.sitkBox
        )
        eroder.SetKernelRadius(1)
        eroder.BoundaryToForegroundOff()

        a2 = SimpleITK.And(
            a2, SimpleITK.InvertIntensity(eroder.Execute(a2), maximum=1)
        )
        b2 = SimpleITK.And(
            b2, SimpleITK.InvertIntensity(eroder.Execute(b2), maximum=1)
        )

    padf = SimpleITK.ConstantPadImageFilter()
    padf.SetConstant(0)
    padf.SetPadLowerBound((1, 1))
    padf.SetPadUpperBound((1, 1))
    b2p = padf.Execute(b2)

    f = SimpleITK.SignedMaurerDistanceMapImageFilter()
    f.SetUseImageSpacing(True)
    f.SetBackgroundValue(1)
    f.InsideIsPositiveOn()
    f.SetSquaredDistance(False)

    d = -SimpleITK.GetArrayFromImage(
        f.Execute(SimpleITK.InvertIntensity(b2p, maximum=1))
    )[1:-1, :][:, 1:-1]
    d[d < 0] = 0

    return d[SimpleITK.GetArrayFromImage(a2) == 1]


def sitk_directed_contour_distance(a, b, voxelspace):
    a2 = SimpleITK.GetImageFromArray((a == 1).astype(np.uint8))
    b2 = SimpleITK.GetImageFromArray((b == 1).astype(np.uint8))
    if voxelspace is None:
        voxelspace = [1, 1]
    a2.SetSpacing(voxelspace[::-1])
    b2.SetSpacing(voxelspace[::-1])

    footprint = generate_binary_structure(a.ndim, a.ndim)
    footprint = SimpleITK.GetImageFromArray(footprint.astype(np.uint8))
    conv = SimpleITK.ConvolutionImageFilter()
    conv.SetBoundaryCondition(
        BoundaryCondition=1
    )  # ZeroFluxNeumannBoundaryCondition enum
    conv.NormalizeOff()
    mask = SimpleITK.GetArrayFromImage(conv.Execute(a2, footprint)) < np.sum(
        footprint
    )

    padf = SimpleITK.ConstantPadImageFilter()
    padf.SetConstant(0)
    padf.SetPadLowerBound((1, 1))
    padf.SetPadUpperBound((1, 1))
    b2p = padf.Execute(b2)

    f = SimpleITK.SignedMaurerDistanceMapImageFilter()
    f.SetUseImageSpacing(True)
    f.SetBackgroundValue(1)
    f.InsideIsPositiveOn()
    f.SetSquaredDistance(False)

    d = -SimpleITK.GetArrayFromImage(
        f.Execute(SimpleITK.InvertIntensity(b2p, maximum=1))
    )[1:-1, :][:, 1:-1]
    d[d < 0] = 0

    return d[(SimpleITK.GetArrayFromImage(a2) == 1) & mask]


@pytest.mark.parametrize(
    "a,b",
    [
        [
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
        ]
        for _ in range(20)
    ],
)
@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
def test_directed_contour_distance(a, b, voxelspace):
    sd = stats.__directed_contour_distances(a, b, voxelspace)
    sd2 = stats.__directed_contour_distances(b, a, voxelspace)
    sd3 = sitk_directed_contour_distance(a, b, voxelspace)
    sd4 = sitk_directed_contour_distance(b, a, voxelspace)

    assert len(sd) == len(sd3) and np.allclose(sd, sd3)
    assert len(sd2) == len(sd4) and np.allclose(sd2, sd4)
    assert len(sd) != len(sd2) or not np.allclose(sd, sd2)


@pytest.mark.parametrize(
    "a,b",
    [
        [
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
        ]
        for _ in range(20)
    ],
)
@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
@pytest.mark.parametrize("connectivity", [0, 1, 2])
def test_surface_distance(a, b, voxelspace, connectivity):
    sd = stats.__surface_distances(a, b, voxelspace, connectivity=connectivity)
    sd2 = stats.__surface_distances(
        b, a, voxelspace, connectivity=connectivity
    )
    sd3 = sitk_surface_distance(a, b, voxelspace, connectivity)
    sd4 = sitk_surface_distance(b, a, voxelspace, connectivity)

    assert len(sd) == len(sd3) and np.allclose(sd, sd3)
    assert len(sd2) == len(sd4) and np.allclose(sd2, sd4)
    assert len(sd) != len(sd2) or not np.allclose(sd, sd2)


@pytest.mark.parametrize(
    "a", [np.random.randint(0, 2, (30, 20, 10)).astype(np.bool)]
)
@pytest.mark.parametrize(
    "b", [np.random.randint(0, 2, (30, 20, 10)).astype(np.bool)]
)
@pytest.mark.parametrize("voxelspace", [None])  # , (0.3, 0.8, 1.2)])
@pytest.mark.parametrize("connectivity", [0, 1, 2])
def test_hd_and_contour_functions(a, b, voxelspace, connectivity):
    r1 = stats.hausdorff_distance(a, b, voxelspace, connectivity)
    r2 = stats.percentile_hausdorff_distance(a, b, 1, voxelspace, connectivity)
    r3 = stats.modified_hausdorff_distance(a, b, voxelspace, connectivity)
    r4 = stats.percentile_hausdorff_distance(
        a, b, 0.95, voxelspace, connectivity
    )
    r5 = stats.relative_absolute_volume_difference(a, b)
    r6 = stats.relative_absolute_volume_difference(b, a)
    r7 = stats.mean_contour_distance(a, b, voxelspace)
    r8 = stats.mean_contour_distance(b, a, voxelspace)
    r = stats.hausdorff_distance_measures(
        a, b, voxelspace, connectivity, percentile=0.95
    )
    r9 = r.distance
    r10 = r.percentile_distance
    r11 = r.modified_distance
    assert r1 == r9
    assert r4 == r10
    assert np.allclose(r3, r11)
    assert r1 == r2
    assert r7 == r8
    assert r1 != r3
    assert r1 != r4
    assert r5 != r6
    assert r1 != r5
    assert r1 != r6
    assert r1 != r7


@pytest.mark.parametrize("a", [np.random.randint(0, 3, (30, 20, 10))])
@pytest.mark.parametrize("b", [np.random.randint(0, 3, (30, 20, 10))])
@pytest.mark.parametrize("classes", [[0, 1, 2]])
def test_cm_functions(a, b, classes):
    cm = stats.calculate_confusion_matrix(a, b, classes)
    r1 = stats.dice_from_confusion_matrix(cm)
    r2 = stats.jaccard_from_confusion_matrix(cm)
    r3 = stats.jaccard_to_dice(r2)
    r4 = stats.dice_to_jaccard(r1)
    r5 = stats.accuracies_from_confusion_matrix(cm)
    for r in [r1, r2, r3, r4, r5]:
        assert len(r) == len(classes)
        assert not np.isnan(r).any()
    assert np.allclose(r1, r3)
    assert np.allclose(r2, r4)
    assert not np.allclose(r2, r5)
    assert not np.allclose(r1, r2)
    assert not np.allclose(r3, r4)
    assert not np.allclose(r1, r5)
    assert not np.allclose(r3, r5)


# The following code is modified from the SciPy test suite
#
# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

INTEGER_TYPES = [
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
]

FLOAT_TYPES = [np.float32, np.float64]

TYPES = INTEGER_TYPES + FLOAT_TYPES


@pytest.mark.parametrize("type_", TYPES)
def test_distance_transform_edt01(type_):
    # euclidean distance transform (edt)
    data = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        type_,
    )
    out, ft = stats.distance_transform_edt_float32(data, return_indices=True)
    bf = ndimage.distance_transform_bf(data, "euclidean")
    assert_array_almost_equal(bf, out)

    dt = ft - np.indices(ft.shape[1:], dtype=ft.dtype)
    dt = dt.astype(np.float64)
    np.multiply(dt, dt, dt)
    dt = np.add.reduce(dt, axis=0)
    np.sqrt(dt, dt)

    assert_array_almost_equal(bf, dt)


@pytest.mark.parametrize("type_", TYPES)
def test_distance_transform_edt02(type_):
    data = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        type_,
    )
    tdt, tft = stats.distance_transform_edt_float32(data, return_indices=True)
    dts = []
    fts = []
    dt = np.zeros(data.shape, dtype=np.float32)
    stats.distance_transform_edt_float32(data, distances=dt)
    dts.append(dt)
    ft = stats.distance_transform_edt_float32(
        data, return_distances=0, return_indices=True
    )
    fts.append(ft)
    ft = np.indices(data.shape, dtype=np.int32)
    stats.distance_transform_edt_float32(
        data, return_distances=False, return_indices=True, indices=ft
    )
    fts.append(ft)
    dt, ft = stats.distance_transform_edt_float32(data, return_indices=True)
    dts.append(dt)
    fts.append(ft)
    dt = np.zeros(data.shape, dtype=np.float32)
    ft = stats.distance_transform_edt_float32(
        data, distances=dt, return_indices=True
    )
    dts.append(dt)
    fts.append(ft)
    ft = np.indices(data.shape, dtype=np.int32)
    dt = stats.distance_transform_edt_float32(
        data, return_indices=True, indices=ft
    )
    dts.append(dt)
    fts.append(ft)
    dt = np.zeros(data.shape, dtype=np.float32)
    ft = np.indices(data.shape, dtype=np.int32)
    stats.distance_transform_edt_float32(
        data, distances=dt, return_indices=True, indices=ft
    )
    dts.append(dt)
    fts.append(ft)
    for dt in dts:
        assert_array_almost_equal(tdt, dt)
    for ft in fts:
        assert_array_almost_equal(tft, ft)


@pytest.mark.parametrize("type_", TYPES)
def test_distance_transform_edt03(type_):
    data = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        type_,
    )
    ref = ndimage.distance_transform_bf(data, "euclidean", sampling=[2, 2])
    out = stats.distance_transform_edt_float32(data, sampling=[2, 2])
    assert_array_almost_equal(ref, out)


@pytest.mark.parametrize("type_", TYPES)
def test_distance_transform_edt4(type_):
    data = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        type_,
    )
    ref = ndimage.distance_transform_bf(data, "euclidean", sampling=[2, 1])
    out = stats.distance_transform_edt_float32(data, sampling=[2, 1])
    assert_array_almost_equal(ref, out)


def test_distance_transform_edt5():
    # Ticket #954 regression test
    out = stats.distance_transform_edt_float32(False)
    assert_array_almost_equal(out, [0.0])
