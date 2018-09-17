import pytest
import numpy as np
import evalutils.stats as stats
import sklearn


@pytest.mark.parametrize("Y_true", [np.random.randint(0,2,(30,20,10)).astype(np.int)])
@pytest.mark.parametrize("Y_pred", [np.random.randint(0,2,(30,20,10)).astype(np.int)])
@pytest.mark.parametrize("labels", [[0], [0, 1], [0, 2], [0, 1, 2]])
def test_calculate_confusion_matrix(Y_true, Y_pred, labels):
    result = stats.calculate_confusion_matrix(Y_true, Y_pred, labels)
    result2 = sklearn.metrics.confusion_matrix(Y_true.flatten(), Y_pred.flatten(), labels)
    assert result.shape[0] == len(labels) and result.shape[1] == len(labels)
    assert np.equal(result, result2).all()


@pytest.mark.parametrize("A", [np.random.randint(0,2,(30,20,10)).astype(np.bool)])
@pytest.mark.parametrize("B", [np.random.randint(0,2,(30,20,10)).astype(np.bool)])
@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8, 1.2)])
def test_avd(A, B, voxelspace):
    r1 = stats.avd(A, B, voxelspace)
    r2 = stats.avd(B, A, voxelspace)
    assert r1 == r2


@pytest.mark.parametrize("A", [np.random.randint(0,2,(30,20,10)).astype(np.bool)])
@pytest.mark.parametrize("B", [np.random.randint(0,2,(30,20,10)).astype(np.bool)])
@pytest.mark.parametrize("voxelspace", [None, (0.3, 0.8, 1.2)])
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


@pytest.mark.parametrize("A", [np.random.randint(0,3,(30,20,10)).astype(np.int)])
@pytest.mark.parametrize("B", [np.random.randint(0,3,(30,20,10)).astype(np.int)])
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
