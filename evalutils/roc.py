from collections import namedtuple

import numpy as np
from numpy import ndarray
from sklearn import metrics

bootstrapped_roc_ci_curves = namedtuple(
    "bootstrapped_roc_ci_curves",
    [
        "fpr_vals",
        "mean_tpr_vals",
        "low_tpr_vals",
        "high_tpr_vals",
        "low_az_val",
        "high_az_val",
    ],
)


def get_bootstrapped_roc_ci_curves(
    y_pred: ndarray,
    y_true: ndarray,
    num_bootstraps: int = 100,
    ci_to_use: float = 0.95,
) -> bootstrapped_roc_ci_curves:
    """
    Produces Confidence-Interval Curves to go alongside a regular ROC curve
    This is done by using boostrapping.
    Bootstrapping is done by selecting len(y_pred) samples randomly
    (with replacement) from y_pred and y_true.
    This is done num_boostraps times.

    Parameters
    ----------
    y_pred
        The predictions (scores) produced by the system being evaluated
    y_true
        The true labels (1 or 0) which are the reference standard being used
    num_bootstraps
        How many times to make a random sample with replacement
    ci_to_use
        Which confidence interval is required.

    Returns
    -------
    fpr_vals
        An equally spaced set of fpr vals between 0 and 1
    mean_tpr_vals
        The mean tpr vals (one per fpr_val) obtained by boostrapping
    low_tpr_vals
        The tpr vals (one per fpr_val) representing lower curve for CI
    high_tpr_vals
        The tpr vals (one per fpr_val) representing the upper curve for CI
    low_Az_val
        The lower Az (AUC) val for the given CI_to_use
    high_Az_val
        The higher Az (AUC) val for the given CI_to_use
    """

    rng_seed = 40  # control reproducibility
    bootstrapped_az_scores = []

    tprs_list = []
    base_fpr = np.linspace(0, 1, 101)
    rng = np.random.RandomState(rng_seed)

    while len(bootstrapped_az_scores) < num_bootstraps:
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        # get the fpr and tpr for this bootstrap
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true[indices], y_pred[indices]
        )
        # get values at fixed fpr locations
        tpr_b = np.interp(base_fpr, fpr, tpr)
        tpr_b[0] = 0.0
        # append to list for all bootstraps
        tprs_list.append(tpr_b)

        # Get the Az score
        az_score = metrics.auc(fpr, tpr)
        bootstrapped_az_scores.append(az_score)

    # Get the mean of the boostrapped tprs (at each fpr location)
    tprs_array = np.array(tprs_list)
    mean_tprs = tprs_array.mean(axis=0)

    # half the error margin allowed
    one_sided_ci = (1 - ci_to_use) / 2

    tprs_upper = []
    tprs_lower = []
    # Now get upper CI and lower CI values for each distinct fpr location
    for b_fpr in range(len(base_fpr)):
        tprs = tprs_array[:, b_fpr]
        tprs.sort()

        tpr_upper = tprs[int((1 - one_sided_ci) * len(tprs))]
        tprs_upper.append(tpr_upper)
        tpr_lower = tprs[int(one_sided_ci * len(tprs))]
        tprs_lower.append(tpr_lower)

    sorted_az_scores = np.array(bootstrapped_az_scores)
    sorted_az_scores.sort()

    az_ci_lower = sorted_az_scores[int(one_sided_ci * len(sorted_az_scores))]
    az_ci_upper = sorted_az_scores[
        int((1 - one_sided_ci) * len(sorted_az_scores))
    ]

    return bootstrapped_roc_ci_curves(
        fpr_vals=base_fpr,
        mean_tpr_vals=mean_tprs,
        low_tpr_vals=np.asarray(tprs_lower),
        high_tpr_vals=np.asarray(tprs_upper),
        low_az_val=az_ci_lower,
        high_az_val=az_ci_upper,
    )


bootstrapped_ci_point_error = namedtuple(
    "bootstrapped_ci_point_error",
    [
        "mean_fprs",
        "mean_tprs",
        "low_tpr_vals",
        "high_tpr_vals",
        "low_fpr_vals",
        "high_fpr_vals",
    ],
)


def get_bootstrapped_ci_point_error(
    y_score: ndarray,
    y_true: ndarray,
    num_bootstraps: int = 100,
    ci_to_use: float = 0.95,
    exclude_first_last: bool = True,
) -> bootstrapped_ci_point_error:
    """
    Produces Confidence-Interval errors for individual points from ROC
    Useful when only few ROC points exist so they will be plotted individually
    e.g. when range of score values in y_score is very small
    (e.g. manual observer scores)

    Note that this method only works by analysing the cloud of boostrapped
    points generatedfor a particular threshold value.  A fixed number of
    threshold values is essential. Therefore the scores in y_score must be
    from a fixed discrete set of values, eg. [1,2,3,4,5]

    Bootstrapping is done by selecting len(y_score) samples randomly
    (with replacement) from y_score and y_true.
    This is done num_boostraps times.

    Parameters
    ----------
    y_score
        The scores produced by the system being evaluated. A discrete set of
        possible scores must be used.
    y_true
        The true labels (1 or 0) which are the reference standard being used
    num_bootstraps: integer
        How many times to make a random sample with replacement
    ci_to_use
        Which confidence interval is required.
    exclude_first_last
        The first and last ROC point (0,0 and 1,1) are usually irrelevant
        in these scenarios where only a few ROC points will be
        individually plotted.
        Set this to true to ignore these first and last points.

    Returns
    -------
    mean_fprs
        The array of mean fpr values (1 per possible ROC point)
    mean_tprs
        The array of mean tpr values (1 per possible ROC point)
    low_tpr_vals
        The tpr vals (one per ROC point) representing lowest val in CI
    high_tpr_vals
        The tpr vals (one per ROC point) representing the highest val in CI
    low_fpr_vals
        The fpr vals (one per ROC point) representing lowest val in CI_to_use
    high_fpr_vals
        The fpr vals (one per ROC point) representing the highest val in CI
    """
    rng_seed = 40  # control reproducibility
    tprs_list = []
    fprs_list = []
    rng = np.random.RandomState(rng_seed)

    num_possible_scores = len(np.unique(y_score))

    while len(tprs_list) < num_bootstraps:
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_score) - 1, len(y_score))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        # get ROC data this boostrap
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true[indices], y_score[indices]
        )
        if len(fpr) < num_possible_scores + 1:
            # if all scores are not represented in this selection then a
            # different number of ROC thresholds will be defined.
            # This causes problems.
            continue

        # remove first and last items - these are just end points of the ROC
        if exclude_first_last:
            fpr = fpr[1:-1]
            tpr = tpr[1:-1]

        # append these boostrap values to the list
        tprs_list.append(tpr)
        fprs_list.append(fpr)

    # Get the mean values for fpr and tpr at each ROC location
    tprs_array = np.array(tprs_list)
    fprs_array = np.array(fprs_list)

    mean_tprs = tprs_array.mean(axis=0)
    mean_fprs = fprs_array.mean(axis=0)

    # half the error margin allowed
    one_sided_ci = (1 - ci_to_use) / 2

    tprs_upper = []
    tprs_lower = []
    fprs_upper = []
    fprs_lower = []

    # for each of the ROC points
    for boot_strap_point in range(tprs_array.shape[1]):
        # sort all of the tpr values we calculated at this point
        tprs = tprs_array[:, boot_strap_point]
        tprs.sort()
        # sort all of the fpr values we calculated at this point
        fprs = fprs_array[:, boot_strap_point]
        fprs.sort()

        tpr_upper = tprs[int((1 - one_sided_ci) * len(tprs))]
        tprs_upper.append(tpr_upper)
        tpr_lower = tprs[int(one_sided_ci * len(tprs))]
        tprs_lower.append(tpr_lower)

        fpr_upper = fprs[int((1 - one_sided_ci) * len(fprs))]
        fprs_upper.append(fpr_upper)
        fpr_lower = fprs[int(one_sided_ci * len(fprs))]
        fprs_lower.append(fpr_lower)

    tprs_lower = np.asarray(tprs_lower)
    tprs_upper = np.asarray(tprs_upper)
    fprs_lower = np.asarray(fprs_lower)
    fprs_upper = np.asarray(fprs_upper)

    return bootstrapped_ci_point_error(
        mean_fprs=mean_fprs,
        mean_tprs=mean_tprs,
        low_tpr_vals=tprs_lower,
        high_tpr_vals=tprs_upper,
        low_fpr_vals=fprs_lower,
        high_fpr_vals=fprs_upper,
    )
