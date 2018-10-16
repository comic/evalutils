# -*- coding: utf-8 -*-
import numpy as np

from evalutils import roc


def test_get_bootstrapped_roc_ci_curves():
    y_true = np.random.randint(0, 2, 500).astype(np.int)
    y_pred = np.random.random_sample(500)
    (
        base_fpr,
        mean_tprs_95,
        tprs_lower_95,
        tprs_upper_95,
        az_ci_lower_95,
        az_ci_upper_95,
    ) = roc.get_bootstrapped_roc_ci_curves(
        y_pred, y_true, num_bootstraps=3, ci_to_use=0.95
    )
    # check that all tpr/fpr values are between 0 and 1
    assert (base_fpr >= 0).all() and (base_fpr <= 1).all()
    assert (mean_tprs_95 >= 0).all() and (mean_tprs_95 <= 1).all()
    assert (tprs_lower_95 >= 0).all() and (tprs_lower_95 <= 1).all()
    assert (tprs_upper_95 >= 0).all() and (tprs_upper_95 <= 1).all()

    # check that the lower is always <= the higher curve
    assert (tprs_lower_95 <= tprs_upper_95).all()
    # check that the mean is never outside the CI values
    assert (tprs_lower_95 <= mean_tprs_95).all() and (
        tprs_upper_95 >= mean_tprs_95
    ).all()
    # check that the az values are both in range 0-1
    assert 0 <= az_ci_lower_95 <= 1
    assert 0 <= az_ci_upper_95 <= 1
    # check that the lower az value is <= the upper az value
    assert az_ci_lower_95 <= az_ci_upper_95

    # repeat with a different CI
    (
        base_fpr,
        mean_tprs_65,
        tprs_lower_65,
        tprs_upper_65,
        az_CI_lower_65,
        az_CI_upper_65,
    ) = roc.get_bootstrapped_roc_ci_curves(
        y_pred, y_true, num_bootstraps=3, ci_to_use=0.65
    )

    # check that the lower_95 curve is always <= lower_65 curve
    assert (tprs_lower_95 <= tprs_lower_65).all()
    # check that the upper_95 curve is always >= upper_65 curve
    assert (tprs_upper_95 >= tprs_upper_65).all()

    # check that the az_lower_95 is always <= az_lower_65
    assert az_ci_lower_95 <= az_CI_lower_65
    # check that the az_upper_95 is always >= az_upper_65
    assert az_ci_upper_95 >= az_CI_upper_65


def test_get_bootstrapped_ci_point_error():
    y_true = np.random.randint(0, 2, 500).astype(np.int)
    y_pred = np.random.randint(1, 10, 500).astype(np.int)
    (
        mean_fprs_95,
        mean_tprs_95,
        tprs_lower_95,
        tprs_upper_95,
        fprs_lower_95,
        fprs_upper_95,
    ) = roc.get_bootstrapped_ci_point_error(
        y_pred, y_true, num_bootstraps=100, ci_to_use=0.95
    )

    # check that all tpr/fpr values are between 0 and 1
    assert (mean_fprs_95 >= 0).all() and (mean_fprs_95 <= 1).all()
    assert (mean_tprs_95 >= 0).all() and (mean_tprs_95 <= 1).all()
    assert (tprs_lower_95 >= 0).all() and (tprs_lower_95 <= 1).all()
    assert (tprs_upper_95 >= 0).all() and (tprs_upper_95 <= 1).all()
    assert (fprs_lower_95 >= 0).all() and (fprs_lower_95 <= 1).all()
    assert (fprs_upper_95 >= 0).all() and (fprs_upper_95 <= 1).all()

    # check that the lower values are  <= the higher ones
    assert (tprs_lower_95 <= tprs_upper_95).all()
    assert (fprs_lower_95 <= fprs_upper_95).all()

    # check that the mean is never outside the CI values
    assert (tprs_lower_95 <= mean_tprs_95).all() and (
        tprs_upper_95 >= mean_tprs_95
    ).all()
    assert (fprs_lower_95 <= mean_fprs_95).all() and (
        fprs_upper_95 >= mean_fprs_95
    ).all()

    # repeat with a different CI
    (
        mean_fprs_65,
        mean_tprs_65,
        tprs_lower_65,
        tprs_upper_65,
        fprs_lower_65,
        fprs_upper_65,
    ) = roc.get_bootstrapped_ci_point_error(
        y_pred, y_true, num_bootstraps=100, ci_to_use=0.65
    )

    # check that the lower_95 val is always <= lower_65 val
    assert (tprs_lower_95 <= tprs_lower_65).all()
    assert (fprs_lower_95 <= fprs_lower_65).all()

    # check that the upper_95 val is always >= upper_65 val
    assert (tprs_upper_95 >= tprs_upper_65).all()
    assert (fprs_upper_95 >= fprs_upper_65).all()
