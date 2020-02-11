import numpy as np

from evalutils import roc


def test_get_bootstrapped_roc_ci_curves():
    y_true = np.random.randint(0, 2, 500).astype(np.int)
    y_pred = np.random.random_sample(500)
    roc_95 = roc.get_bootstrapped_roc_ci_curves(
        y_pred, y_true, num_bootstraps=3, ci_to_use=0.95
    )
    # check that all tpr/fpr values are between 0 and 1
    assert (roc_95.fpr_vals >= 0).all() and (roc_95.fpr_vals <= 1).all()
    assert (roc_95.mean_tpr_vals >= 0).all() and (
        roc_95.mean_tpr_vals <= 1
    ).all()
    assert (roc_95.low_tpr_vals >= 0).all() and (
        roc_95.low_tpr_vals <= 1
    ).all()
    assert (roc_95.high_tpr_vals >= 0).all() and (
        roc_95.high_tpr_vals <= 1
    ).all()

    # check that the lower is always <= the higher curve
    assert (roc_95.low_tpr_vals <= roc_95.high_tpr_vals).all()
    # check that the mean is never outside the CI values
    assert (roc_95.low_tpr_vals <= roc_95.mean_tpr_vals).all() and (
        roc_95.high_tpr_vals >= roc_95.mean_tpr_vals
    ).all()
    # check that the az values are both in range 0-1
    assert 0 <= roc_95.low_az_val <= 1
    assert 0 <= roc_95.high_az_val <= 1
    # check that the lower az value is <= the upper az value
    assert roc_95.low_az_val <= roc_95.high_az_val

    # repeat with a different CI
    roc_65 = roc.get_bootstrapped_roc_ci_curves(
        y_pred, y_true, num_bootstraps=3, ci_to_use=0.65
    )

    # check that the lower_95 curve is always <= lower_65 curve
    assert (roc_95.low_tpr_vals <= roc_65.low_tpr_vals).all()
    # check that the upper_95 curve is always >= upper_65 curve
    assert (roc_95.high_tpr_vals >= roc_65.high_tpr_vals).all()

    # check that the az_lower_95 is always <= az_lower_65
    assert roc_95.low_az_val <= roc_65.low_az_val
    # check that the az_upper_95 is always >= az_upper_65
    assert roc_95.high_az_val >= roc_65.high_az_val


def test_get_bootstrapped_ci_point_error():
    y_true = np.random.randint(0, 2, 500).astype(np.int)
    y_pred = np.random.randint(1, 10, 500).astype(np.int)
    err_95 = roc.get_bootstrapped_ci_point_error(
        y_pred, y_true, num_bootstraps=100, ci_to_use=0.95
    )

    # check that all tpr/fpr values are between 0 and 1
    assert (err_95.mean_fprs >= 0).all() and (err_95.mean_fprs <= 1).all()
    assert (err_95.mean_tprs >= 0).all() and (err_95.mean_tprs <= 1).all()
    assert (err_95.low_tpr_vals >= 0).all() and (
        err_95.low_tpr_vals <= 1
    ).all()
    assert (err_95.high_tpr_vals >= 0).all() and (
        err_95.high_tpr_vals <= 1
    ).all()
    assert (err_95.low_fpr_vals >= 0).all() and (
        err_95.low_fpr_vals <= 1
    ).all()
    assert (err_95.high_fpr_vals >= 0).all() and (
        err_95.high_fpr_vals <= 1
    ).all()

    # check that the lower values are  <= the higher ones
    assert (err_95.low_tpr_vals <= err_95.high_tpr_vals).all()
    assert (err_95.low_fpr_vals <= err_95.high_fpr_vals).all()

    # check that the mean is never outside the CI values
    assert (err_95.low_tpr_vals <= err_95.mean_tprs).all() and (
        err_95.high_tpr_vals >= err_95.mean_tprs
    ).all()
    assert (err_95.low_fpr_vals <= err_95.mean_fprs).all() and (
        err_95.high_fpr_vals >= err_95.mean_fprs
    ).all()

    # repeat with a different CI
    err_65 = roc.get_bootstrapped_ci_point_error(
        y_pred, y_true, num_bootstraps=100, ci_to_use=0.65
    )

    # check that the lower_95 val is always <= lower_65 val
    assert (err_95.low_tpr_vals <= err_65.low_tpr_vals).all()
    assert (err_95.low_fpr_vals <= err_65.low_fpr_vals).all()

    # check that the upper_95 val is always >= upper_65 val
    assert (err_95.high_tpr_vals >= err_65.high_tpr_vals).all()
    assert (err_95.high_fpr_vals >= err_65.high_fpr_vals).all()
