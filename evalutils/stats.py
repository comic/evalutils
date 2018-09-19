# -*- coding: utf-8 -*-
from collections import namedtuple
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import (
    binary_erosion,
    generate_binary_structure,
    distance_transform_edt,
)


def calculate_confusion_matrix(
    y_true: ndarray, y_pred: ndarray, labels: List[int]
) -> ndarray:
    """
    Efficient confusion matrix calculation, based on sklearn interface

    Parameters
    ----------
    y_true
             Target multi-object segmentation mask
    y_pred
             Predicted multi-object segmentation mask
    labels
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
            cm[i, j] = np.sum((y_true == x) & (y_pred == y))

    return cm


def jaccard_to_dice(jacc: ndarray) -> Union[int, float, ndarray]:
    """
    Conversion computation from Jaccard to Dice

    Parameters
    ----------
    jacc
           1 or N Jaccard values within [0 .. 1]

    Returns
    -------
    1 or N Dice values within [0 .. 1]
    """
    assert all(jacc >= 0) and all(jacc <= 1)
    return (jacc * 2.0) / (1.0 + jacc)


def dice_to_jaccard(dice: ndarray) -> Union[int, float, ndarray]:
    """
    Conversion computation from Dice to Jaccard

    Parameters
    ----------
    dice
           1 or N Dice values within [0 .. 1]

    Returns
    -------
    1 or N Jaccard values within [0 .. 1]
    """
    assert all(dice >= 0) and all(dice <= 1)

    return dice / (2.0 - dice)


def accuracies_from_confusion_matrix(cm: ndarray) -> ndarray:
    """
    Computes accuracy scores from a confusion matrix

    Parameters
    ----------
    cm
         N x N Input confusion matrix

    Returns
    -------
    1d ndarray containing accuracy scores for all N classes
    """
    results = np.zeros((len(cm)), dtype=np.float32)

    for i in range(len(cm)):
        mask = np.ones((len(cm)), dtype=np.bool)
        mask[i] = False
        results[i] = cm[i, i] + np.sum(cm[mask, :][:, mask])

    return results // float(np.sum(cm))


def jaccard_from_confusion_matrix(cm: ndarray) -> ndarray:
    """
    Computes Jaccard scores from a confusion matrix a.k.a. intersection over
    union (IoU)

    Parameters
    ----------
    cm
         N x N Input confusion matrix

    Returns
    -------
    1d ndarray containing Jaccard scores for all N classes
    """
    assert cm.ndim == 2
    assert cm.shape[0] == cm.shape[1]

    jaccs = np.zeros((cm.shape[0]), dtype=np.float32)

    for i in range(cm.shape[0]):
        jaccs[i] = cm[i, i] / float(
            np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i]
        )

    return jaccs


def dice_from_confusion_matrix(cm: ndarray) -> ndarray:
    """
    Computes Dice scores from a confusion matrix

    Parameters
    ----------
    cm
         N x N Input confusion matrix

    Returns
    -------
    1d ndarray containing Dice scores for all N classes
    """
    assert cm.ndim == 2
    assert cm.shape[0] == cm.shape[1]

    dices = np.zeros((cm.shape[0]), dtype=np.float32)

    for i in range(cm.shape[0]):
        dices[i] = 2 * cm[i, i] / float(np.sum(cm[i, :]) + np.sum(cm[:, i]))

    return dices


def __surface_distances(
    s1: ndarray,
    s2: ndarray,
    voxelspacing: Optional[Tuple[float, float]] = None,
    connectivity: int = 1,
) -> ndarray:
    """
    Computes set of surface distances.

    Retrieve set of distances for all elements from set s1 to s2
    With a connectivity of 1 or higher only the distances between
    the contours of s1 and s2 are used.

    Parameters
    ----------
    s1
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    s2
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually
        be :math:`> 1`.

    Returns
    -------
        The distances from all non-zero object(s) in ```s1``` to the nearest
        non zero object(s) in ```s2```. The distance unit is the same as for
        the spacing of elements along each dimension, which is usually given in
        mm.

    Notes
    -----
    This function is not symmetric.
    """
    s1_b = np.atleast_1d(s1.astype(np.bool))
    s2_b = np.atleast_1d(s2.astype(np.bool))

    if connectivity > 0:
        footprint = generate_binary_structure(s1.ndim, connectivity)
        s2_b = s2_b & ~binary_erosion(s2_b, structure=footprint, iterations=1)
        s1_b = s1_b & ~binary_erosion(s1_b, structure=footprint, iterations=1)

    return distance_transform_edt(~s2_b, sampling=voxelspacing)[s1_b]


def hausdorff_distance(
    s1: ndarray,
    s2: ndarray,
    voxelspacing: Optional[Tuple[float, float]] = None,
    connectivity: int = 1,
) -> float:
    """
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects
    in two images. It is defined as the maximum surface distance between the
    objects.

    Parameters
    ----------
    s1
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    s2
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually
        be :math:`> 1`.

    Returns
    -------
        The symmetric Hausdorff Distance between the object(s) in ```s1``` and
        the object(s) in ```s2```. The distance unit is the same as for the
        spacing of elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric.
    Implementation inspired by medpy.metric.binary
    http://pythonhosted.org/MedPy/_modules/medpy/metric/binary.html
    """
    s1_dist = __surface_distances(s1, s2, voxelspacing, connectivity)
    s2_dist = __surface_distances(s2, s1, voxelspacing, connectivity)

    return max(s1_dist.max(), s2_dist.max())


def percentile_hausdorff_distance(
    s1: ndarray,
    s2: ndarray,
    percentile: Union[int, float] = 0.95,
    voxelspacing: Optional[Tuple[float, float]] = None,
    connectivity: int = 1,
) -> float:
    """
    Nth Percentile Hausdorff Distance.

    Computes a percentile for the (symmetric) Hausdorff Distance between the
    binary objects in two images. It is defined as the maximum surface distance
    between the objects at the nth percentile.

    Parameters
    ----------
    s1
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    s2
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    percentile
        The percentile to perform the comparison on the two sorted distance
        sets
    voxelspacing
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually
        be :math:`> 1`.

    Returns
    -------
        The maximum Percentile Hausdorff Distance between the object(s) in
        ```s1``` and the object(s) in ```s2``` at the ```percentile```
        percentile.
        The distance unit is the same as for the spacing of elements along each
        dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric.
    """
    s1_dist = __surface_distances(s1, s2, voxelspacing, connectivity)
    s2_dist = __surface_distances(s2, s1, voxelspacing, connectivity)

    s1_dist.sort()
    s2_dist.sort()

    return max(
        s1_dist[int((len(s1_dist) - 1) * percentile)],
        s2_dist[int((len(s2_dist) - 1) * percentile)],
    )


def modified_hausdorff_distance(
    s1: ndarray,
    s2: ndarray,
    voxelspacing: Optional[Tuple[float, float]] = None,
    connectivity: int = 1,
) -> float:
    """
    Computes the (symmetric) Modified Hausdorff Distance (MHD) between the
    binary objects in two images. It is defined as the maximum average surface
    distance between the objects.

    Parameters
    ----------
    s1
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    s2
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually
        be :math:`> 1`.

    Returns
    -------
        The symmetric Modified Hausdorff Distance between the object(s) in
        ```s1``` and the object(s) in ```s2```. The distance unit is the same
        as for the spacing of elements along each dimension, which is usually
        given in mm.

    Notes
    -----
    This is a real metric.
    """
    s1_dist = __surface_distances(s1, s2, voxelspacing, connectivity)
    s2_dist = __surface_distances(s2, s1, voxelspacing, connectivity)

    return max(s1_dist.mean(), s2_dist.mean())


def relative_absolute_volume_distance(s1: ndarray, s2: ndarray) -> float:
    """
    Calculate relative absolute volume difference from s2 to s1

    Parameters
    ----------
    s1
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else. s1 is taken
        to be the reference.
    s2
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
        The relative absolute volume difference between the object(s) in
        ``input1`` and the object(s) in ``input2``. This is a percentage value
        in the range :math:`[0, +inf]` for which a :math:`0` denotes an ideal
        score.

    Notes
    -----
    This is not a real metric! it is asymmetric.
    """
    s1, s2 = s1 != 0, s2 != 0

    return abs(np.sum(s2) - np.sum(s1)) / float(np.sum(s1))


def absolute_volume_distance(
    s1: ndarray, s2: ndarray, voxelspacing: Optional[Tuple[float, float]]
) -> float:
    """
    Calculate absolute volume difference from s2 to s1

    Parameters
    ----------
    s1
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else. s1 is taken
        to be the reference.
    s2
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.

    Returns
    -------
        The absolute volume difference between the object(s) in ``input1``
        and the object(s) in ``input2``. This is a percentage value in the
        range :math:`[0, +inf]` for which a :math:`0` denotes an ideal score.

    Notes
    -----
    This is a real metric
    """
    s1, s2 = s1 != 0, s2 != 0

    if voxelspacing is None:
        voxelspacing = [1] * s1.ndim

    if isinstance(voxelspacing, float) or isinstance(voxelspacing, int):
        voxelspacing = [voxelspacing] * s1.ndim

    assert len(voxelspacing) == s1.ndim
    assert s1.ndim == s2.ndim

    volume_per_voxel = np.prod(voxelspacing)

    return np.abs(np.sum(s2) - np.sum(s1)) * volume_per_voxel


def __directed_contour_distances(
    s1: ndarray,
    s2: ndarray,
    voxelspacing: Optional[Tuple[float, float]] = None,
) -> ndarray:
    """
    Computes set of surface contour distances.
    This function always explicitly calculates the contour-set of s1.

    Retrieve set of distances for all elements from set s1 to s2
    The elements of the contour of s1 are determined by:
    1) whether elements in s1 are fully enclosed by other voxels.
          (in a 3x3x3 neighborhood)
    2) whether elements in s1 are ON.

     Parameters
    ----------
    s1
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    s2
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.

    Returns
    -------
        The distances from all non-zero object(s) in ```s1``` to the nearest
        non zero object(s) in ```s2```. The distance unit is the same as for
        the spacing of elements along each dimension, which is usually given in
        mm.

    Notes
    -----
    This function is not symmetric.

    For determining the contours, the border handling from ITK relies on
    `ZeroFluxNeumannBoundaryCondition`, which equals `nearest` mode in scipy.

    """
    s1_b = np.atleast_1d(s1.astype(np.bool))
    s2_b = np.atleast_1d(s2.astype(np.bool))

    # all elements in neighborhood are fully checked! equals np.ones((3,3,3))
    # for s1.ndim == 3
    footprint = generate_binary_structure(s1.ndim, s1.ndim)
    df = distance_transform_edt(~s2_b, sampling=voxelspacing)

    # generate mask for elements not entirly enclosed by mask s1_b
    # (contours & non-zero elements)
    # convolve mode ITK relies on ZeroFluxNeumannBoundaryCondition == nearest
    mask = convolve(s1_b.astype(np.int), footprint, mode="nearest") < np.sum(
        footprint
    )

    # return distance to contours only
    return df[mask & s1_b]


def mean_contour_distance(
    s1: ndarray,
    s2: ndarray,
    voxelspacing: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Computes the (symmetric) Mean Contour Distance between the binary objects
    in two images. It is defined as the maximum average surface distance
    between the objects.

    Parameters
    ----------
    s1
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    s2
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.

    Returns
    -------
        The symmetric Mean Contour Distance between the object(s) in ```s1```
        and the object(s) in ```s2```. The distance unit is the same as for the
        spacing of elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric that mimics the ITK MeanContourDistanceFilter.
    """
    s1_c_dist = __directed_contour_distances(s1, s2, voxelspacing)
    s2_c_dist = __directed_contour_distances(s2, s1, voxelspacing)

    return max(s1_c_dist.mean(), s2_c_dist.mean())


HausdorffMeasures = namedtuple(
    "HausdorffMeasures",
    ["distance", "modified_distance", "percentile_distance"],
)


def hausdorff_distance_measures(
    s1: ndarray,
    s2: ndarray,
    voxelspacing: None = None,
    connectivity: int = 1,
    percentile: float = 0.95,
) -> HausdorffMeasures:
    """
    Returns multiple Hausdorff measures - (hd, modified_hd, percentile_hd)
    Since measures share common calculations,
    together the measures can be calculated more efficiently

    Parameters
    ----------
    s1
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    s2
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually
        be :math:`> 1`.
    percentile
        The percentile at which to calculate the Hausdorff Distance

    Returns
    -------
        The hausdorff distance, modified hausdorff distance and percentile
        hausdorff distance

    Notes
    -----
    This returns real metrics.
    """
    s1_b = np.atleast_1d(s1.astype(np.bool))
    s2_b = np.atleast_1d(s2.astype(np.bool))

    if connectivity > 0:
        footprint = generate_binary_structure(s1.ndim, connectivity)
        s2_b = s2_b & ~binary_erosion(s2_b, structure=footprint, iterations=1)
        s1_b = s1_b & ~binary_erosion(s1_b, structure=footprint, iterations=1)

    s1_dist = distance_transform_edt(~s2_b, sampling=voxelspacing)[s1_b]
    s2_dist = distance_transform_edt(~s1_b, sampling=voxelspacing)[s2_b]

    s1_dist.sort()
    s2_dist.sort()

    # calculate all hausdorff statistics
    distance = max(s1_dist.max(), s2_dist.max())
    modified_distance = max(s1_dist.mean(), s2_dist.mean())
    percentile_distance = max(
        s1_dist[int((len(s1_dist) - 1) * percentile)],
        s2_dist[int((len(s2_dist) - 1) * percentile)],
    )

    return HausdorffMeasures(
        distance=distance,
        modified_distance=modified_distance,
        percentile_distance=percentile_distance,
    )
