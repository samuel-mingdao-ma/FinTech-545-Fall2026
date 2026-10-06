import numpy as np

from risk545.common import (
    effective_sample_size,
    exponential_weights,
    historical_var_es,
    normal_var,
)


def test_exponential_weights_are_normalized_and_recent_heaviest():
    weights = exponential_weights(100, 0.94)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[-1] > weights[0]
    assert effective_sample_size(weights) > 1.0


def test_historical_var_and_es_are_positive_losses():
    pnl = np.array([-10.0, -5.0, 1.0, 2.0, 3.0])
    var, es = historical_var_es(pnl, alpha=0.2)
    assert var > 0.0
    assert es >= var


def test_normal_var_increases_with_volatility():
    assert normal_var(0.0, 2.0) > normal_var(0.0, 1.0)
