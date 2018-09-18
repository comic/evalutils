import pytest
import numpy as np
import evalutils.stats as stats
import sklearn
import SimpleITK as sitk


@pytest.mark.parametrize("Y_true", [np.random.randint(0,2,(30,20,10)).astype(np.int)])
@pytest.mark.parametrize("Y_pred", [np.random.randint(0,2,(30,20,10)).astype(np.int)])
@pytest.mark.parametrize("labels", [[0], [0, 1], [0, 2], [0, 1, 2]])
def test_calculate_confusion_matrix(Y_true, Y_pred, labels):
    result = stats.calculate_confusion_matrix(Y_true, Y_pred, labels)
    result2 = sklearn.metrics.confusion_matrix(Y_true.flatten(), Y_pred.flatten(), labels)
    assert result.shape[0] == len(labels) and result.shape[1] == len(labels)
    assert np.equal(result, result2).all()


def test_accuracies_from_cm():
    cm = np.array([[5, 2, 1],
                   [1, 2, 1],
                   [1, 0, 4]])
    expected = np.array([5 + 2 + 1 + 4, 2 + 5 + 4 + 1 + 1, 4 + 2 + 1 + 2 + 5],
                        dtype=np.float) // float(np.sum(cm))
    accs = stats.accuracies_from_cm(cm)
    assert np.equal(expected, accs).all()


def test_jaccard_from_cm():
    cm = np.array([[5, 2, 1],
                   [1, 2, 1],
                   [1, 0, 4]])
    expected = np.array([5, 2, 4], dtype=np.float) / \
               np.array([2 + 1 + 1 + 1 + 5, 2 + 1 + 1 + 2, 4 + 1 + 1 + 1], dtype=np.float)
    accs = stats.jaccard_from_cm(cm)
    assert np.allclose(expected, accs)


def test_dice_from_cm():
    cm = np.array([[5, 2, 1],
                   [1, 2, 1],
                   [1, 0, 4]])
    expected = np.array([5, 2, 4], dtype=np.float) * 2 / \
               np.array([2 + 1 + 1 + 1 + 5 + 5, 2 + 1 + 1 + 2 + 2, 4 + 4 + 1 + 1 + 1], dtype=np.float)
    accs = stats.dice_from_cm(cm)
    assert np.allclose(expected, accs)


def test_ravd():
    A = np.array([[1, 0, 0],
                  [1, 1, 1],
                  [0, 1, 1]], dtype=np.bool)
    B = np.array([[1, 0, 1],
                  [1, 0, 1],
                  [0, 0, 1]], dtype=np.bool)
    r1 = stats.ravd(A, B)
    r2 = stats.ravd(B, A)
    assert r1 != r2
    assert r1 == (6.0 - 5.0) / 6.0
    assert r2 == (6.0 - 5.0) / 5.0


@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
def test_avd(voxelspace):
    A = np.array([[1, 0, 0],
                  [1, 1, 1],
                  [0, 1, 1]], dtype=np.bool)
    B = np.array([[1, 0, 1],
                  [1, 0, 1],
                  [0, 0, 1]], dtype=np.bool)
    r1 = stats.avd(A, B, voxelspace)
    r2 = stats.avd(B, A, voxelspace)
    assert r1 == r2
    assert r1 == (6.0 - 5.0) * (1 if voxelspace is None else np.prod(voxelspace))


@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8), (12.0, 4.0)])
@pytest.mark.parametrize("connectivity", [0, 1, 2])
def test_hd(voxelspace, connectivity):
    A = np.array([[1, 0, 0, 1],
                  [1, 1, 1, 0],
                  [0, 1, 1, 0]], dtype=np.bool)
    B = np.array([[1, 0, 1, 1],
                  [1, 0, 1, 0],
                  [0, 0, 1, 1]], dtype=np.bool)
    hd = stats.hd(A, B, voxelspace, connectivity=connectivity)
    hd2 = stats.hd(B, A, voxelspace, connectivity=connectivity)

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

    assert hd == hd3
    assert hd2 == hd4


@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
@pytest.mark.parametrize("connectivity", [0, 1, 2])
def test_modified_hd(voxelspace, connectivity):
    A = np.array([[1, 0, 0, 1],
                  [0, 1, 1, 0],
                  [0, 0, 1, 0]], dtype=np.bool)
    B = np.array([[1, 0, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 1, 1]], dtype=np.bool)
    dA = sitk_surface_distance(A, B, voxelspace, connectivity)
    dB = sitk_surface_distance(B, A, voxelspace, connectivity)
    mhd = max(dA.mean(), dB.mean())

    mhd2 = stats.modified_hd(A, B, voxelspace, connectivity)
    mhd3 = stats.modified_hd(B, A, voxelspace, connectivity)

    assert np.isclose(mhd, mhd2)
    assert np.isclose(mhd, mhd3)
    assert mhd2 == mhd3



@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
def test_percentile_hd(voxelspace):
    assert True


@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
def test_mean_contour_distance(voxelspace):
    assert True


@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
def test_directed_contour_distance(voxelspace):
    assert True


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
        eroder.SetKernelType(sitk.sitkCross if connectivity <= 1 else sitk.sitkBox)
        eroder.SetKernelRadius(1)
        eroder.BoundaryToForegroundOff()

        A2 = sitk.And(A2, sitk.InvertIntensity(eroder.Execute(A2), maximum=1))
        B2 = sitk.And(B2, sitk.InvertIntensity(eroder.Execute(B2), maximum=1))


        # BE2 = eroder.Execute(B2)
        # from scipy.ndimage.morphology import binary_erosion, generate_binary_structure, distance_transform_edt
        # footprint = generate_binary_structure(A.ndim, connectivity)
        # Bb = B & ~binary_erosion(B, structure=footprint, iterations=1)
        # Ab = A & ~binary_erosion(A, structure=footprint, iterations=1)
        # print(voxelspace, connectivity)
        # print(footprint)
        #
        # print(B.astype(np.int))
        # print(binary_erosion(B, structure=footprint, iterations=1).astype(np.int))
        # print(sitk.GetArrayFromImage(BE2))
        #
        # # print(Bb.astype(np.int))
        # # print(sitk.GetArrayFromImage(B2))
        # print(abs(Bb.astype(np.int) - sitk.GetArrayFromImage(B2)))

    padf = sitk.ConstantPadImageFilter()
    padf.SetConstant(0)
    padf.SetPadLowerBound((1,1))
    padf.SetPadUpperBound((1,1))
    B2p = padf.Execute(B2)

    f = sitk.SignedMaurerDistanceMapImageFilter()
    f.SetUseImageSpacing(True)
    f.SetBackgroundValue(1)
    f.InsideIsPositiveOn()
    f.SetSquaredDistance(False)

    D = -sitk.GetArrayFromImage(f.Execute(sitk.InvertIntensity(B2p, maximum=1)))[1:-1,:][:,1:-1]
    D[D < 0] = 0

    # from scipy.ndimage.morphology import distance_transform_edt
    # aa = D
    # bb = distance_transform_edt(~(B.astype(np.bool)), sampling=voxelspace)
    # if (np.abs(aa - bb) > 0.001).any():
    #     np.set_printoptions(suppress=True)
    #     print(voxelspace, connectivity)
    #     print(aa)
    #     print(bb) #[A.astype(np.bool)]
    #     print(aa - bb)

    return D[sitk.GetArrayFromImage(A2) == 1]


@pytest.mark.parametrize("A,B", [
                                # [np.array([[1, 0, 0, 1],
                                #          [1, 1, 1, 0],
                                #          [0, 1, 1, 0]], dtype=np.bool),
                                #   np.array([[1, 0, 0, 1],
                                #             [1, 0, 0, 0],
                                #             [0, 0, 0, 1]], dtype=np.bool)],
                                 [np.random.randint(0, 2, (6, 6), dtype=np.bool),
                                  np.random.randint(0, 2, (6, 6), dtype=np.bool),
                                  ] for _ in range(100)])
@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8)])
@pytest.mark.parametrize("connectivity", [0, 1, 2])
def test_surface_distance(A, B, voxelspace, connectivity):
    sd = stats.__surface_distances(A, B, voxelspace, connectivity=connectivity)
    sd2 = stats.__surface_distances(B, A, voxelspace, connectivity=connectivity)
    sd3 = sitk_surface_distance(A, B, voxelspace, connectivity)
    sd4 = sitk_surface_distance(B, A, voxelspace, connectivity)

    assert len(sd) == len(sd3) and np.allclose(sd, sd3)
    assert len(sd2) == len(sd4) and np.allclose(sd2, sd4)
    assert len(sd) != len(sd2) or not np.allclose(sd, sd2)


@pytest.mark.parametrize("A", [np.random.randint(0,2,(30,20,10)).astype(np.bool)])
@pytest.mark.parametrize("B", [np.random.randint(0,2,(30,20,10)).astype(np.bool)])
@pytest.mark.parametrize("voxelspace", [None]) #, (0.3, 0.8, 1.2)])
@pytest.mark.parametrize("connectivity", [0, 1, 2])
def test_hd_and_contour_functions(A, B, voxelspace, connectivity):
    r1 = stats.hd(A, B, voxelspace, connectivity)
    r2 = stats.percentile_hd(A, B, 1, voxelspace,connectivity)
    r3 = stats.modified_hd(A, B, voxelspace, connectivity)
    r4 = stats.percentile_hd(A, B, 0.95, voxelspace, connectivity)
    r5 = stats.ravd(A, B)
    r6 = stats.ravd(B, A)
    r7 = stats.mean_contour_distance(A, B, voxelspace)
    r8 = stats.mean_contour_distance(B, A, voxelspace)
    r = stats.hd_measures(A, B, voxelspace, connectivity, percentile=0.95)
    r9 = r["hd"]
    r10 = r["percentile_hd"]
    r11 = r["modified_hd"]
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


@pytest.mark.parametrize("A", [np.random.randint(0,3,(30,20,10))])
@pytest.mark.parametrize("B", [np.random.randint(0,3,(30,20,10))])
@pytest.mark.parametrize("classes", [[0, 1, 2]])
def test_cm_functions(A, B, classes):
    cm = stats.calculate_confusion_matrix(A, B, classes)
    r1 = stats.dice_from_cm(cm)
    r2 = stats.jaccard_from_cm(cm)
    r3 = stats.jaccard_to_dice(r2)
    r4 = stats.dice_to_jaccard(r1)
    r5 = stats.accuracies_from_cm(cm)
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
