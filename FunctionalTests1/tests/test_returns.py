"""Independent return checks: known values, dates, identities and invalid inputs."""
import numpy as np
import pandas as pd
import pytest
from risk545 import calculate_returns


def test_hand_calculated_returns_and_dates():
    prices = pd.DataFrame({"Date": ["d1", "d2", "d3"], "B": [100., 110., 99.], "A": [50., 50., 100.]})
    arithmetic = calculate_returns(prices)
    assert list(arithmetic.columns) == list(prices.columns)
    assert arithmetic.Date.tolist() == ["d2", "d3"]
    np.testing.assert_allclose(arithmetic[["B", "A"]], [[.1, 0], [-.1, 1]], atol=1e-15)
    np.testing.assert_allclose(calculate_returns(prices, "log")[["B", "A"]], np.log([[1.1, 1], [.9, 2]]))
    pd.testing.assert_frame_equal(prices, pd.DataFrame({"Date": ["d1", "d2", "d3"], "B": [100., 110., 99.], "A": [50., 50., 100.]}))


def test_log_arithmetic_identity_and_scaling():
    rng = np.random.default_rng(61)
    p = pd.DataFrame(np.exp(rng.normal(size=(12, 3))), columns=["C", "B", "A"])
    p.insert(0, "Date", range(12))
    a = calculate_returns(p).drop(columns="Date")
    log = calculate_returns(p, "log").drop(columns="Date")
    np.testing.assert_allclose(np.expm1(log), a, atol=1e-13)
    scaled = p.copy(); scaled[["C", "B", "A"]] *= [2, 3, 4]
    np.testing.assert_allclose(calculate_returns(scaled).drop(columns="Date"), a, atol=1e-13)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_prices(value):
    with pytest.raises(ValueError):
        calculate_returns(pd.DataFrame({"Date": [1, 2], "A": [100, value]}))


def test_invalid_method():
    with pytest.raises(ValueError):
        calculate_returns(pd.DataFrame({"Date": [1, 2], "A": [100, 110]}), "unknown")
