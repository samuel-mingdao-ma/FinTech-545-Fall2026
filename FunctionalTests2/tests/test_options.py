"""Independent numerical and financial checks for the option library."""
from math import exp, sqrt

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from risk545_ft2.options import (
    build_cases, cash_dividend_price, greeks, price, tree_greeks, tree_price,
)


@pytest.mark.parametrize("is_call", [True, False])
def test_european_price_against_risk_neutral_payoff_integral(is_call):
    """Integrate the lognormal payoff rather than reusing the closed form."""
    s, k, t, r, b, sigma = 103., 98., .8, .041, .015, .27
    sign = 1 if is_call else -1
    exercise_z = (np.log(k/s)-(b-.5*sigma*sigma)*t)/(sigma*sqrt(t))
    lower, upper = (exercise_z, 12.) if is_call else (-12., exercise_z)
    integrand = lambda z: max(sign*(s*exp((b-.5*sigma*sigma)*t+sigma*sqrt(t)*z)-k), 0.)*exp(-z*z/2)/sqrt(2*np.pi)
    expected = exp(-r*t)*quad(integrand, lower, upper, epsabs=1e-10)[0]
    assert price(s, k, t, r, b, sigma, is_call) == pytest.approx(expected, abs=2e-10)


def test_generalized_put_call_parity_and_vectorization():
    s = np.array([85., 100., 125.])
    k, t, r, b, sigma = 101., 1.2, .045, .012, .31
    difference = price(s, k, t, r, b, sigma)-price(s, k, t, r, b, sigma, False)
    np.testing.assert_allclose(difference, s*np.exp((b-r)*t)-k*np.exp(-r*t), atol=3e-14)
    np.testing.assert_array_equal(price(s, k, 0., r, b, sigma), np.maximum(s-k, 0.))
    assert price(100., 90., .5, .04, .02, 0.) == pytest.approx(exp(-.02)*max(100*exp(.01)-90, 0.))
    assert price(0., 100., 1., .04, .02, .2, False) == pytest.approx(100*exp(-.04))


@pytest.mark.parametrize("is_call", [True, False])
def test_european_greeks_against_independent_repricing(is_call):
    parameters = np.array([107., 100., .7, .037, .012, .29])
    actual = greeks(*parameters, is_call)

    def derivative(index, h):
        high, low = parameters.copy(), parameters.copy()
        high[index] += h
        low[index] -= h
        return (price(*high, is_call)-price(*low, is_call))/(2*h)

    assert actual["delta"] == pytest.approx(derivative(0, .001), abs=1e-9)
    assert actual["vega"] == pytest.approx(derivative(5, .00001), abs=1e-7)
    assert actual["theta"] == pytest.approx(-derivative(2, .00001), abs=1e-7)
    assert actual["carry_rho"] == pytest.approx(derivative(4, .00001), abs=1e-7)
    assert actual["rho_fixed_carry"] == pytest.approx(derivative(3, .00001), abs=1e-7)
    # Stock-option rho changes both r and b, holding q=r-b constant.
    h = .00001
    high, low = parameters.copy(), parameters.copy()
    high[3:5] += h
    low[3:5] -= h
    rho = (price(*high, is_call)-price(*low, is_call))/(2*h)
    assert actual["rho"] == pytest.approx(rho, abs=1e-7)
    spot_h = .02
    gamma = (price(parameters[0]+spot_h, *parameters[1:], is_call)
             +price(parameters[0]-spot_h, *parameters[1:], is_call)
             -2*actual["value"])/spot_h**2
    assert actual["gamma"] == pytest.approx(gamma, abs=5e-10)
    assert actual["rho"]-actual["carry_rho"] == pytest.approx(actual["rho_fixed_carry"], abs=1e-13)


@pytest.mark.parametrize("is_call", [True, False])
def test_european_crr_converges_to_gbsm(is_call):
    parameters = (100., 103., .75, .04, .015, .3)
    analytic = price(*parameters, is_call)
    lattice = tree_price(*parameters, n=1600, is_call=is_call, american=False)
    assert lattice == pytest.approx(analytic, abs=.003)


def test_american_exercise_and_no_dividend_call_identity():
    arguments = (100., 100., 1., .06, .06, .2)
    assert tree_price(*arguments, n=200) == pytest.approx(
        tree_price(*arguments, n=200, american=False), abs=1e-12)
    american_put = tree_price(*arguments, n=200, is_call=False)
    european_put = tree_price(*arguments, n=200, is_call=False, american=False)
    assert american_put > european_put+.1
    assert tree_price(50., 100., 1., .08, .08, .1, n=200, is_call=False) >= 50.


def test_tree_rho_holds_carry_fixed_and_theta_is_maturity_sensitivity():
    parameters = (101., 99., .6, .04, .04, .25)
    result = tree_greeks(*parameters, n=120)
    # For this no-dividend call exercise is never optimal. With b fixed, only
    # discounting changes when r changes, giving the exact rho=-T*V identity.
    assert result["rho"] == pytest.approx(-parameters[2]*result["value"], abs=1e-6)
    maturity_h = 1e-4
    future = tree_price(parameters[0], parameters[1], parameters[2]+maturity_h, *parameters[3:], n=120)
    earlier = tree_price(parameters[0], parameters[1], parameters[2]-maturity_h, *parameters[3:], n=120)
    assert result["theta"] == pytest.approx((future-earlier)/(2*maturity_h), abs=2e-6)
    assert result["theta"] > 0
    assert greeks(*parameters)["theta"] < 0
    # Repricing a portfolio with the documented $1.50 gamma bump must agree.
    bumped_sum = tree_price(parameters[0]+1.5, *parameters[1:], n=120)+tree_price(parameters[0]-1.5, *parameters[1:], n=120)
    assert bumped_sum-2*result["value"] == pytest.approx(1.5**2*result["gamma"], abs=1e-13)


def _full_path_value(s, k, t, r, sigma, n, dividends, is_call, american):
    """Exhaust every binary history, with no recombination or batched arrays."""
    up = exp(sigma*sqrt(t/n))
    down = 1/up
    probability = (exp(r*t/n)-down)/(up-down)
    discount = exp(-r*t/n)
    sign = 1 if is_call else -1
    payments = {}
    for step, cash in dividends:
        payments[step] = payments.get(step, 0.)+cash

    def node(spot, step):
        before = max(sign*(spot-k), 0.)
        after = max(spot-payments.get(step, 0.), 0.)
        if step == n:
            return max(sign*(after-k), 0.)
        continuation = discount*(probability*node(after*up, step+1)
                                 +(1-probability)*node(after*down, step+1))
        if not american:
            return continuation
        return max(before, max(sign*(after-k), 0.), continuation)

    return node(s, 0)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("american", [True, False])
def test_cash_dividends_against_full_binary_history_tree(is_call, american):
    arguments = (103., 100., .75, .05, .3, 7)
    payments = [(2, 3.), (5, 6.)]
    expected = _full_path_value(*arguments, payments, is_call, american)
    actual = cash_dividend_price(*arguments, payments, is_call, american)
    assert actual == pytest.approx(expected, abs=5e-13)


def test_cash_call_can_exercise_immediately_before_payment():
    arguments = (150., 100., 1., .05, .15, 4)
    payment = [(1, 40.)]
    actual = cash_dividend_price(*arguments, payment)
    expected = _full_path_value(*arguments, payment, True, True)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert actual > cash_dividend_price(*arguments, payment, american=False)+20


def test_cash_zero_payments_equivalent_grid_and_stock_floor():
    arguments = (100., 102., .8, .04, .25, 12)
    empty = cash_dividend_price(*arguments, is_call=False)
    assert empty == pytest.approx(tree_price(100., 102., .8, .04, .04, .25, 12, False), abs=1e-12)
    combined = cash_dividend_price(*arguments, [(6, 3.), (3, 1.), (6, 2.)], is_call=False)
    assert combined == pytest.approx(cash_dividend_price(*arguments, [(3, 1.), (6, 5.)], is_call=False), abs=1e-12)
    floor_arguments = (100., 100., .5, .03, .2, 4)
    assert cash_dividend_price(*floor_arguments, [(1, 200.)], False) == pytest.approx(
        _full_path_value(*floor_arguments, [(1, 200.)], False, True), abs=1e-12)
    vector = cash_dividend_price(np.array([90., 100.]), 102., .8, .04, .25, 12, [(6, 2.)])
    assert vector.shape == (2,)
    assert vector[1] > vector[0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_invalid_european_inputs(bad):
    with pytest.raises(ValueError):
        price(bad, 100., 1., .04, .04, .2)
    with pytest.raises(ValueError):
        greeks(100., 100., 0., .04, .04, .2)


@pytest.mark.parametrize("payments", [[(1.5, 1.)], [(0, 1.)], [(10, 1.)], [(2, -1.)], [(True, 1.)]])
def test_invalid_cash_dividend_schedule(payments):
    with pytest.raises(ValueError):
        cash_dividend_price(100., 100., 1., .04, .2, 10, payments)


def test_invalid_tree_probability_and_flags():
    with pytest.raises(ValueError, match="probability"):
        tree_price(100., 100., 1., .04, 2., .01, n=1)
    with pytest.raises(ValueError):
        tree_price(100., 100., 1., .04, .04, .2, n=True)
    with pytest.raises(ValueError):
        price(100., 100., 1., .04, .04, .2, is_call="Put")


def test_case_builder_uses_only_inputs_and_records_greek_conventions(tmp_path):
    pd.DataFrame([{
        "ID": 1, "Option Type": "Call", "Underlying": 100., "Strike": 100.,
        "DaysToMaturity": 50, "DayPerYear": 365, "RiskFreeRate": .045,
        "DividendRate": 0., "ImpliedVol": .2,
    }]).to_csv(tmp_path/"test12_1.csv", index=False)
    pd.DataFrame([{
        "ID": 1, "Option Type": "Put", "Underlying": 100., "Strike": 100.,
        "DaysToMaturity": 4, "DayPerYear": 365, "RiskFreeRate": .045,
        "ImpliedVol": .2, "DividendDates": "1,3", "DividendAmts": ".1,.2",
    }]).to_csv(tmp_path/"test12_3.csv", index=False)
    # The directory contains no expected-output files; results must be computed.
    cases, diagnostic = build_cases(tmp_path, simulations=1, seed=1)
    assert set(cases) == {"12.1", "12.2", "12.3"}
    assert list(cases["12.1"].columns) == ["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]
    assert cases["12.1"].Theta.iloc[0] < 0 < cases["12.2"].Theta.iloc[0]
    expected = _full_path_value(100., 100., 4/365, .045, .2, 8, [(2, .1), (6, .2)], False, True)
    assert cases["12.3"].Value.iloc[0] == pytest.approx(expected, abs=1e-12)
    assert "holding cost of carry b fixed" in diagnostic["american_greeks"]["rho"]
    assert diagnostic["cash_dividends"]["steps_per_day"] == 2
