# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure, distance_transform_edt
from scipy.ndimage.filters import convolve
from typing import Union, Dict


def calculate_confusion_matrix(Y_true, Y_pred, labels) -> np.ndarray:
    """Efficient confusion matrix calculation, based on sklearn interface

    Parameters
    ----------
    Y_true : array_like
             Target multi-object segmentation mask
    Y_pred : array_like
             Predicted multi-object segmentation mask
    labels : List of integers
             Inclusive list of N labels to compute the confusion matrix for.

    Returns
    -------
    N x N confusion matrix for Y_pred w.r.t. Y_true

    Notes
    -----
    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` but
    predicted to be in group :math:`j`.

    """
    cm = np.zeros((len(labels), len(labels)), dtype=np.int)
    for i, x in enumerate(labels):
        for j, y in enumerate(labels):
            cm[i, j] = np.sum((Y_true == x) & (Y_pred == y))
    return cm


def jaccard_to_dice(jacc) -> Union[int, float, np.ndarray]:
    """Conversion computation from Jaccard to Dice

    Parameters
    ----------
    jacc : array_like or float
           1 or N Jaccard values within [0 .. 1]

    Returns
    -------
    1 or N Dice values within [0 .. 1]
    """
    assert all(jacc >= 0) and all(jacc <= 1)
    return (jacc * 2.0) / (1.0 + jacc)


def dice_to_jaccard(dice) -> Union[int, float, np.ndarray]:
    """Conversion computation from Dice to Jaccard

    Parameters
    ----------
    dice : array_like or float
           1 or N Dice values within [0 .. 1]

    Returns
    -------
    1 or N Jaccard values within [0 .. 1]
    """
    assert all(dice >= 0) and all(dice <= 1)
    return dice / (2.0 - dice)


def accuracies_from_cm(cm) -> np.ndarray:
    """Computes accuracy scores from a confusion matrix

    Parameters
    ----------
    cm : array_like (2D)
         N x N Input confusion matrix

    Returns
    -------
    1d ndarray containing accuracy scores for all N classes
    """
    results = np.zeros((len(cm)), dtype=np.float32)
    for i in range(len(cm)):
        filter = np.ones((len(cm)), dtype=np.bool)
        filter[i] = 0
        results[i] = (cm[i, i] + np.sum(cm[filter, filter])) / float(np.sum(cm))
    return results


def jaccard_from_cm(cm) -> np.ndarray:
    """Computes Jaccard scores from a confusion matrix
    a.k.a. intersection over union (IoU)

    Parameters
    ----------
    cm : array_like (2D)
         N x N Input confusion matrix

    Returns
    -------
    1d ndarray containing Jaccard scores for all N classes
    """
    assert(cm.ndim == 2)
    assert(cm.shape[0] == cm.shape[1])
    jaccs = np.zeros((cm.shape[0]), dtype=np.float32)
    for i in range(cm.shape[0]):
        jaccs[i] = cm[i, i] / float(np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
    return jaccs


def dice_from_cm(cm) -> np.ndarray:
    """Computes Dice scores from a confusion matrix

    Parameters
    ----------
    cm : array_like (2D)
         N x N Input confusion matrix

    Returns
    -------
    1d ndarray containing Dice scores for all N classes
    """
    assert(cm.ndim == 2)
    assert(cm.shape[0] == cm.shape[1])
    dices = np.zeros((cm.shape[0]), dtype=np.float32)
    for i in range(cm.shape[0]):
        dices[i] = 2 * cm[i, i] / float(np.sum(cm[i, :]) + np.sum(cm[:, i]))
    return dices


def icc(M) -> float:
    """
    Computes intra class correlation

    Parameters
    ----------
    M : array_like (2D, observations x classes)
        Input data represents a design matrix. Should be numerical.
        First dimension contains observations, second dimension contains classes.

    Returns
    -------
    icc : float
        Returns the intra class correlation between classes over the observations.
    """
    assert M.ndims == 2
    result = 1.0 / (len(M) * np.var(M)) * np.sum(np.prod(M - np.mean(M), axis=1))
    return result


def __surface_distances(A, B, voxelspacing=None, connectivity=1) -> np.ndarray:
    """
    Computes set of surface distances.

    Retrieve set of distances for all elements from set A to B
    With a connectivity of 1 or higher only the distances between
    the contours of A and B are used.

    Parameters
    ----------
    A : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    B : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int, optional
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.

    Returns
    -------
    distances : 1d ndarray
        The distances from all non-zero object(s) in ```A``` to the nearest
        non zero object(s) in ```B```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This function is not symmetric.
    """
    Ab = np.atleast_1d(A.astype(np.bool))
    Bb = np.atleast_1d(B.astype(np.bool))
    if connectivity > 0:
        footprint = generate_binary_structure(A.ndim, connectivity)
        Bb = Bb & ~binary_erosion(Bb, structure=footprint, iterations=1)
        Ab = Ab & ~binary_erosion(Ab, structure=footprint, iterations=1)
    return distance_transform_edt(~Bb, sampling=voxelspacing)[Ab]


def hd(A, B, voxelspacing=None, connectivity=1) -> float:
    """
    Hausdorff Distance.

    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    A : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    B : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int, optional
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```A``` and the
        object(s) in ```B```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric.
    Implementation inspired by medpy.metric.binary
    http://pythonhosted.org/MedPy/_modules/medpy/metric/binary.html
    """
    dA = __surface_distances(A, B, voxelspacing, connectivity)
    dB = __surface_distances(B, A, voxelspacing, connectivity)
    return max(dA.max(), dB.max())


def percentile_hd(A, B, percentile=0.95, voxelspacing=None, connectivity=1) -> float:
    """
    Nth Percentile Hausdorff Distance.

    Computes a percentile for the (symmetric) Hausdorff Distance between the binary objects in two
    images. It is defined as the maximum surface distance between the objects at the nth percentile.

    Parameters
    ----------
    A : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    B : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    percentile : float between 0 and 1
        The percentile to perform the comparison on the two sorted distance sets
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int, optional
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.

    Returns
    -------
    hd : float
        The maximum Percentile Hausdorff Distance between the object(s) in ```A``` and the
        object(s) in ```B``` at the ```percentile``` percentile.
        The distance unit is the same as for the spacing of elements along each dimension,
        which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric.
    """
    dA = __surface_distances(A, B, voxelspacing, connectivity)
    dB = __surface_distances(B, A, voxelspacing, connectivity)
    dA.sort()
    dB.sort()
    return max(dA[int((len(dA)-1) * percentile)], dB[int((len(dB)-1) * percentile)])


def modified_hd(A, B, voxelspacing=None, connectivity=1) -> float:
    """
    Hausdorff Distance.

    Computes the (symmetric) Modified Hausdorff Distance (MHD) between the binary objects in two
    images. It is defined as the maximum average surface distance between the objects.

    Parameters
    ----------
    A : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    B : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int, optional
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.

    Returns
    -------
    hd : float
        The symmetric Modified Hausdorff Distance between the object(s) in ```A```
        and the object(s) in ```B```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric.
    """
    dA = __surface_distances(A, B, voxelspacing, connectivity).mean()
    dB = __surface_distances(B, A, voxelspacing, connectivity).mean()
    return max(dA.mean(), dB.mean())


def ravd(A, B) -> float:
    """
    Calculate relative absolute volume difference from B to A

    Parameters
    ----------
    A : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else. A is taken
        to be the reference.
    B : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    ravd : float
        The relative absolute volume difference between the object(s) in ``input1``
        and the object(s) in ``input2``. This is a percentage value in the range
        :math:`[0, +inf]` for which a :math:`0` denotes an ideal score.

    Notes
    -----
    This is not a real metric! it is asymmetric.
    """
    A, B = A != 0, B != 0
    return abs(np.sum(B) - np.sum(A)) / float(np.sum(A))


def avd(A, B, voxelspacing) -> float:
    """
    Calculate absolute volume difference from B to A

    Parameters
    ----------
    A : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else. A is taken
        to be the reference.
    B : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.

    Returns
    -------
    avd : float
        The absolute volume difference between the object(s) in ``input1``
        and the object(s) in ``input2``. This is a percentage value in the range
        :math:`[0, +inf]` for which a :math:`0` denotes an ideal score.

    Notes
    -----
    This is a real metric
    """
    A, B = A != 0, B != 0
    if voxelspacing is None:
        voxelspacing = [1] * A.ndim
    if isinstance(voxelspacing, float) or isinstance(voxelspacing, int):
        voxelspacing = [voxelspacing] * A.ndim
    assert(len(voxelspacing) == A.ndim)
    assert(A.ndim == B.ndim)
    volume_per_voxel = np.prod(voxelspacing)
    return np.abs(np.sum(B) - np.sum(A)) * volume_per_voxel


def __directed_contour_distances(A, B, voxelspacing=None) -> np.ndarray:
    """
    Computes set of surface contour distances.
    This function always explicitly calculates the contour-set of A.

    Retrieve set of distances for all elements from set A to B
    The elements of the contour of A are determined by:
    1) whether elements in A are fully enclosed by other voxels.
          (in a 3x3x3 neighborhood)
    2) whether elements in A are ON.

     Parameters
    ----------
    A : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    B : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.

    Returns
    -------
    distances : 1d ndarray
        The distances from all non-zero object(s) in ```A``` to the nearest
        non zero object(s) in ```B```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This function is not symmetric.

    For determing the contours, the border handling from ITK relies on
    `ZeroFluxNeumannBoundaryCondition`, which equals `nearest` mode in scipy.

    """
    Ab = np.atleast_1d(A.astype(np.bool))
    Bb = np.atleast_1d(B.astype(np.bool))
    # all elements in neighborhood are fully checked! equals np.ones((3,3,3)) for A.ndim == 3
    footprint = generate_binary_structure(A.ndim, A.ndim)
    df = distance_transform_edt(~Bb, sampling=voxelspacing)
    # generate mask for elements not entirly enclosed by mask Ab (contours & non-zero elements)
    # convolve mode ITK relies on ZeroFluxNeumannBoundaryCondition == nearest
    mask = convolve(Ab.astype(np.int), footprint, mode='nearest') < np.sum(footprint)
    # return distance to contours only
    return df[mask & Ab]


def mean_contour_distance(A, B, voxelspacing=None) -> float:
    """
    Mean Contour Distance.

    Computes the (symmetric) Mean Contour Distance between the binary objects in two
    images. It is defined as the maximum average surface distance between the objects.

    Parameters
    ----------
    A : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    B : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.

    Returns
    -------
    hd : float
        The symmetric Mean Contour Distance between the object(s) in ```A```
        and the object(s) in ```B```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric that mimics the ITK MeanContourDistanceFilter.
    """
    dA = __directed_contour_distances(A, B, voxelspacing)
    dB = __directed_contour_distances(B, A, voxelspacing)
    return max(dA.mean(), dB.mean())


def hd_measures(A, B, voxelspacing=None, connectivity=1, percentile=0.95) -> Dict:
    """
    Returns multiple Hausdorff measures - (hd, modified_hd, percentile_hd)
    Since measures share common calculations,
    together the measures can be calculated more efficiently

    Parameters
    ----------
    A : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    B : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int, optional
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.

    Returns
    -------
    hd : dict
        The symmetric Modified Hausdorff Distance between the object(s) in ```A```
        and the object(s) in ```B```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This returns real metrics.
    """
    Ab = np.atleast_1d(A.astype(np.bool))
    Bb = np.atleast_1d(B.astype(np.bool))

    if connectivity > 0:
        footprint = generate_binary_structure(A.ndim, connectivity)
        Bb = Bb & ~binary_erosion(Bb, structure=footprint, iterations=1)
        Ab = Ab & ~binary_erosion(Ab, structure=footprint, iterations=1)

    dA = distance_transform_edt(~Bb, sampling=voxelspacing)[Ab]
    dB = distance_transform_edt(~Ab, sampling=voxelspacing)[Bb]
    dA.sort()
    dB.sort()

    # calculate all hausdorff statistics
    hdv = max(dA.max(), dB.max())
    modified_hdv = max(dA.mean(), dB.mean())
    percentile_hdv = max(dA[int((len(dA) - 1) * percentile)], dB[int((len(dB) - 1) * percentile)])

    return dict(hd=hdv, modified_hd=modified_hdv, percentile_hd=percentile_hdv)
