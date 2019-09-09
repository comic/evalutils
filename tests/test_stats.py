import SimpleITK as sitk
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
    "Y_true", [np.random.randint(0, 2, (30, 20, 10)).astype(np.int)]
)
@pytest.mark.parametrize(
    "Y_pred", [np.random.randint(0, 2, (30, 20, 10)).astype(np.int)]
)
@pytest.mark.parametrize("labels", [[0], [0, 1], [0, 2], [0, 1, 2]])
def test_calculate_confusion_matrix(Y_true, Y_pred, labels):
    result = stats.calculate_confusion_matrix(Y_true, Y_pred, labels)
    result2 = sklearn.metrics.confusion_matrix(
        Y_true.flatten(), Y_pred.flatten(), labels
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
    A = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=np.bool)
    B = np.array([[1, 0, 1], [1, 0, 1], [0, 0, 1]], dtype=np.bool)
    r1 = stats.relative_absolute_volume_difference(A, B)
    r2 = stats.relative_absolute_volume_difference(B, A)
    assert r1 != r2
    assert r1 == (6.0 - 5.0) / 6.0
    assert r2 == (6.0 - 5.0) / 5.0


@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
def test_avd(voxelspace):
    A = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=np.bool)
    B = np.array([[1, 0, 1], [1, 0, 1], [0, 0, 1]], dtype=np.bool)
    r1 = stats.absolute_volume_difference(A, B, voxelspace)
    r2 = stats.absolute_volume_difference(B, A, voxelspace)
    assert r1 == r2
    assert r1 == (6.0 - 5.0) * (
        1 if voxelspace is None else np.prod(voxelspace)
    )


# A connectivity of 1 was omitted, because the HausdorffDistanceImageFilter does not support it.
# A connectivity of 0 and 2 appear to achieve similar Hausdorff metrics, although only one is used within the
# HausdorffDistanceImageFilter
@pytest.mark.parametrize(
    "A,B",
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
def test_hd(A, B, voxelspace, connectivity):
    hd = stats.hausdorff_distance(A, B, voxelspace, connectivity=connectivity)
    hd2 = stats.hausdorff_distance(B, A, voxelspace, connectivity=connectivity)

    assert hd == hd2

    A2 = sitk.GetImageFromArray(A.astype(np.uint8))
    B2 = sitk.GetImageFromArray(B.astype(np.uint8))
    if voxelspace is None:
        voxelspace = [1, 1]
    A2.SetSpacing(voxelspace[::-1])
    B2.SetSpacing(voxelspace[::-1])
    hdfilter = sitk.HausdorffDistanceImageFilter()
    hdfilter.Execute(A2, B2)
    hd3 = hdfilter.GetHausdorffDistance()
    hdfilter.Execute(B2, A2)
    hd4 = hdfilter.GetHausdorffDistance()

    assert np.isclose(hd, hd3)
    assert np.isclose(hd2, hd4)


@pytest.mark.parametrize(
    "A,B",
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
def test_modified_hd(A, B, voxelspace, connectivity):
    dA = sitk_surface_distance(A, B, voxelspace, connectivity)
    dB = sitk_surface_distance(B, A, voxelspace, connectivity)
    mhd = max(dA.mean(), dB.mean())

    mhd2 = stats.modified_hausdorff_distance(A, B, voxelspace, connectivity)
    mhd3 = stats.modified_hausdorff_distance(B, A, voxelspace, connectivity)

    assert np.isclose(mhd, mhd2)
    assert np.isclose(mhd, mhd3)
    assert mhd2 == mhd3


@pytest.mark.parametrize(
    "A,B",
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
def test_percentile_hd(A, B, voxelspace, connectivity, percentile):
    dA = sitk_surface_distance(A, B, voxelspace, connectivity)
    dB = sitk_surface_distance(B, A, voxelspace, connectivity)
    dA.sort()
    dB.sort()
    phd = max(
        dA[int((len(dA) - 1) * percentile)],
        dB[int((len(dB) - 1) * percentile)],
    )

    phd2 = stats.percentile_hausdorff_distance(
        A, B, percentile, voxelspace, connectivity
    )
    phd3 = stats.percentile_hausdorff_distance(
        B, A, percentile, voxelspace, connectivity
    )

    assert np.isclose(phd, phd2)
    assert np.isclose(phd, phd3)
    assert phd2 == phd3


@pytest.mark.parametrize(
    "A,B",
    [
        [
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
        ]
        for _ in range(20)
    ],
)
@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
def test_mean_contour_distance(A, B, voxelspace):
    dA = sitk_directed_contour_distance(A, B, voxelspace)
    dB = sitk_directed_contour_distance(B, A, voxelspace)
    mhd = max(dA.mean(), dB.mean())

    mhd2 = stats.mean_contour_distance(A, B, voxelspace)
    mhd3 = stats.mean_contour_distance(B, A, voxelspace)

    assert np.isclose(mhd, mhd2)
    assert np.isclose(mhd, mhd3)
    assert mhd2 == mhd3


def sitk_surface_distance(A, B, voxelspace, connectivity):
    A2 = sitk.GetImageFromArray((A == 1).astype(np.uint8))
    B2 = sitk.GetImageFromArray((B == 1).astype(np.uint8))
    if voxelspace is None:
        voxelspace = [1, 1]
    A2.SetSpacing(voxelspace[::-1])
    B2.SetSpacing(voxelspace[::-1])

    if connectivity > 0:
        eroder = sitk.BinaryErodeImageFilter()
        eroder.SetBackgroundValue(0)
        eroder.SetKernelType(
            sitk.sitkCross if connectivity <= 1 else sitk.sitkBox
        )
        eroder.SetKernelRadius(1)
        eroder.BoundaryToForegroundOff()

        A2 = sitk.And(A2, sitk.InvertIntensity(eroder.Execute(A2), maximum=1))
        B2 = sitk.And(B2, sitk.InvertIntensity(eroder.Execute(B2), maximum=1))

    padf = sitk.ConstantPadImageFilter()
    padf.SetConstant(0)
    padf.SetPadLowerBound((1, 1))
    padf.SetPadUpperBound((1, 1))
    B2p = padf.Execute(B2)

    f = sitk.SignedMaurerDistanceMapImageFilter()
    f.SetUseImageSpacing(True)
    f.SetBackgroundValue(1)
    f.InsideIsPositiveOn()
    f.SetSquaredDistance(False)

    D = -sitk.GetArrayFromImage(
        f.Execute(sitk.InvertIntensity(B2p, maximum=1))
    )[1:-1, :][:, 1:-1]
    D[D < 0] = 0

    return D[sitk.GetArrayFromImage(A2) == 1]


def sitk_directed_contour_distance(A, B, voxelspace):
    A2 = sitk.GetImageFromArray((A == 1).astype(np.uint8))
    B2 = sitk.GetImageFromArray((B == 1).astype(np.uint8))
    if voxelspace is None:
        voxelspace = [1, 1]
    A2.SetSpacing(voxelspace[::-1])
    B2.SetSpacing(voxelspace[::-1])

    footprint = generate_binary_structure(A.ndim, A.ndim)
    footprint = sitk.GetImageFromArray(footprint.astype(np.uint8))
    conv = sitk.ConvolutionImageFilter()
    conv.SetBoundaryCondition(
        BoundaryCondition=1
    )  # ZeroFluxNeumannBoundaryCondition enum
    conv.NormalizeOff()
    mask = sitk.GetArrayFromImage(conv.Execute(A2, footprint)) < np.sum(
        footprint
    )

    padf = sitk.ConstantPadImageFilter()
    padf.SetConstant(0)
    padf.SetPadLowerBound((1, 1))
    padf.SetPadUpperBound((1, 1))
    B2p = padf.Execute(B2)

    f = sitk.SignedMaurerDistanceMapImageFilter()
    f.SetUseImageSpacing(True)
    f.SetBackgroundValue(1)
    f.InsideIsPositiveOn()
    f.SetSquaredDistance(False)

    D = -sitk.GetArrayFromImage(
        f.Execute(sitk.InvertIntensity(B2p, maximum=1))
    )[1:-1, :][:, 1:-1]
    D[D < 0] = 0

    return D[(sitk.GetArrayFromImage(A2) == 1) & mask]


@pytest.mark.parametrize(
    "A,B",
    [
        [
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
            np.random.randint(0, 2, (6, 6), dtype=np.bool),
        ]
        for _ in range(20)
    ],
)
@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
def test_directed_contour_distance(A, B, voxelspace):
    sd = stats.__directed_contour_distances(A, B, voxelspace)
    sd2 = stats.__directed_contour_distances(B, A, voxelspace)
    sd3 = sitk_directed_contour_distance(A, B, voxelspace)
    sd4 = sitk_directed_contour_distance(B, A, voxelspace)

    assert len(sd) == len(sd3) and np.allclose(sd, sd3)
    assert len(sd2) == len(sd4) and np.allclose(sd2, sd4)
    assert len(sd) != len(sd2) or not np.allclose(sd, sd2)


@pytest.mark.parametrize(
    "A,B",
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
def test_surface_distance(A, B, voxelspace, connectivity):
    sd = stats.__surface_distances(A, B, voxelspace, connectivity=connectivity)
    sd2 = stats.__surface_distances(
        B, A, voxelspace, connectivity=connectivity
    )
    sd3 = sitk_surface_distance(A, B, voxelspace, connectivity)
    sd4 = sitk_surface_distance(B, A, voxelspace, connectivity)

    assert len(sd) == len(sd3) and np.allclose(sd, sd3)
    assert len(sd2) == len(sd4) and np.allclose(sd2, sd4)
    assert len(sd) != len(sd2) or not np.allclose(sd, sd2)


@pytest.mark.parametrize(
    "A", [np.random.randint(0, 2, (30, 20, 10)).astype(np.bool)]
)
@pytest.mark.parametrize(
    "B", [np.random.randint(0, 2, (30, 20, 10)).astype(np.bool)]
)
@pytest.mark.parametrize("voxelspace", [None])  # , (0.3, 0.8, 1.2)])
@pytest.mark.parametrize("connectivity", [0, 1, 2])
def test_hd_and_contour_functions(A, B, voxelspace, connectivity):
    r1 = stats.hausdorff_distance(A, B, voxelspace, connectivity)
    r2 = stats.percentile_hausdorff_distance(A, B, 1, voxelspace, connectivity)
    r3 = stats.modified_hausdorff_distance(A, B, voxelspace, connectivity)
    r4 = stats.percentile_hausdorff_distance(
        A, B, 0.95, voxelspace, connectivity
    )
    r5 = stats.relative_absolute_volume_difference(A, B)
    r6 = stats.relative_absolute_volume_difference(B, A)
    r7 = stats.mean_contour_distance(A, B, voxelspace)
    r8 = stats.mean_contour_distance(B, A, voxelspace)
    r = stats.hausdorff_distance_measures(
        A, B, voxelspace, connectivity, percentile=0.95
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


@pytest.mark.parametrize("A", [np.random.randint(0, 3, (30, 20, 10))])
@pytest.mark.parametrize("B", [np.random.randint(0, 3, (30, 20, 10))])
@pytest.mark.parametrize("classes", [[0, 1, 2]])
def test_cm_functions(A, B, classes):
    cm = stats.calculate_confusion_matrix(A, B, classes)
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
