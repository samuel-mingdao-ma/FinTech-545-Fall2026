"""Independent pricing identities and full Q1/Q3 output coverage."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assignment3.pricing import greeks, implied_volatility, price, price_bounds
from assignment3 import q1, q3

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("b", [.035, 0., -.02])
def test_gbsm_matches_discounted_payoff_integral(call, b):
    s, k, t, r, vol = 103., 100., .6, .045, .26
    d = stats.lognorm(vol*np.sqrt(t), scale=s*np.exp((b-.5*vol*vol)*t))
    sign = 1 if call else -1
    expected = np.exp(-r*t)*integrate.quad(
        lambda terminal: max(sign*(terminal-k), 0.)*d.pdf(terminal),
        k if call else 0., np.inf if call else k, epsabs=1e-10,
    )[0]
    assert price(s, k, t, r, b, vol, call) == pytest.approx(expected, abs=2e-8)


def test_put_call_parity_and_vega_equality():
    s, k, t, r, b, vol = 101., 98., .7, .04, .027, .23
    call, put = [greeks(s, k, t, r, b, vol, kind) for kind in (True, False)]
    assert call["price"]-put["price"] == pytest.approx(s*np.exp((b-r)*t)-k*np.exp(-r*t))
    assert call["delta"]-put["delta"] == pytest.approx(np.exp((b-r)*t))
    assert call["gamma"] == pytest.approx(put["gamma"])
    assert call["vega"] == pytest.approx(put["vega"])


@pytest.mark.parametrize("call", [True, False])
def test_greeks_match_central_differences(call):
    s, k, t, r, b, vol = 101., 98., .7, .04, .027, .23
    g = greeks(s, k, t, r, b, vol, call)
    h, h_time, h_vol = .002, 1e-5, 1e-5
    up, down = [price(s+sign*h, k, t, r, b, vol, call) for sign in (1, -1)]
    assert g["delta"] == pytest.approx((up-down)/(2*h), rel=1e-7)
    assert g["gamma"] == pytest.approx((up-2*g["price"]+down)/(h*h), rel=1e-5)
    time_derivative = (price(s, k, t+h_time, r, b, vol, call)-price(s, k, t-h_time, r, b, vol, call))/(2*h_time)
    vega = (price(s, k, t, r, b, vol+h_vol, call)-price(s, k, t, r, b, vol-h_vol, call))/(2*h_vol)
    assert g["theta"] == pytest.approx(-time_derivative, rel=1e-8)
    assert g["vega"] == pytest.approx(vega, rel=1e-8)


def test_vectorized_prices_and_limits():
    spots = np.array([0., 80., 100., 120.])
    np.testing.assert_allclose(price(spots, 100., 0., .04, .04, .2, True), [0., 0., 0., 20.])
    np.testing.assert_allclose(price(spots, 100., 0., .04, .04, .2, False), [100., 20., 0., 0.])
    actual = price(spots, 100., .5, .04, .04, .2, True)
    expected = [price(float(s), 100., .5, .04, .04, .2, True) for s in spots]
    np.testing.assert_allclose(actual, expected)
    assert price(120., 100., .5, .04, .02, 0., True) == pytest.approx(
        120*np.exp(-.02*.5)-100*np.exp(-.04*.5))


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("vol", [.05, .2, .65])
def test_iv_recovers_prices_and_rejects_bounds(call, vol):
    params = (102., 100., .5, .04, .03)
    quote = price(*params, vol, call)
    fitted, status = implied_volatility(quote, *params, call)
    assert status == "ok" and fitted == pytest.approx(vol, abs=1e-10)
    lower, upper = price_bounds(*params, call)
    assert implied_volatility(lower-.01, *params, call)[0] is None
    assert implied_volatility(upper+.01, *params, call)[0] is None
    assert implied_volatility(upper, *params, call)[0] is None
    assert implied_volatility(lower, *params, call)[0] == 0.


def test_q1_actual_required_tables_and_root_repricing(tmp_path):
    out, figures = tmp_path/"results", tmp_path/"figures"
    summary = q1.run(ROOT/"data", out, figures)
    parity = pd.read_csv(out/"q1_parity.csv")
    fitted = pd.read_csv(out/"q1_quotes_iv.csv")
    assert len(parity) == 18 and len(fitted) == 36
    assert set(parity.loc[parity.flag, "Strike"]) == {175.}
    assert len(summary["missing_IV_roots"]) > 0
    assert (out/"q1_predictions.json").exists()
    assert (figures/"q1_smile.png").stat().st_size > 10000
    assert (figures/"q1_vega_sensitivity.png").stat().st_size > 10000
    for source in ("bid", "mid", "ask"):
        assert fitted[f"status_{source}"].notna().all()
        assert fitted.loc[fitted[f"status_{source}"] == "ok", f"IV_{source}"].notna().all()


def test_book_contract_multiplier_and_short_signs():
    positions = pd.DataFrame([
        dict(Portfolio="Synthetic", Type="Stock", Strike=np.nan, DaysToExpiry=np.nan, ImpliedVol=np.nan, Holding=2.),
        dict(Portfolio="Synthetic", Type="Call", Strike=100., DaysToExpiry=30., ImpliedVol=.2, Holding=-3.),
    ])
    s = 101.
    expected = 2*s-3*price(s, 100., 30/255, .0425, .0425, .2, True)
    assert q3.book_values(positions, s)["Synthetic"] == pytest.approx(expected)
    table = q3.position_greeks(positions, s)
    assert table.gamma.sum() < 0 and table.Holding.tolist() == [2., -3.]


def test_q3_actual_both_families_and_all_approximations(tmp_path):
    out, figures = tmp_path/"results", tmp_path/"figures"
    summary = q3.run(ROOT/"data", out, figures, n_sim=5000, seed=545)
    risk = pd.read_csv(out/"q3_full_revaluation_risk.csv")
    approximate = pd.read_csv(out/"q3_approximation_risk.csv")
    dn = pd.read_csv(out/"q3_delta_normal_risk.csv")
    assert len(risk) == 16 and len(approximate) == 48 and len(dn) == 8
    assert set(risk.family) == {"normal", "t"}
    assert set(risk.alpha) == {.05, .01}
    assert (risk.ES >= risk.VaR).all()
    assert set(approximate.approximation) == {"delta_only", "delta_gamma", "delta_gamma_theta"}
    assert summary["parameters"]["option_contract_multiplier"] == 1
    assert (figures/"q3_pnl_curves.png").stat().st_size > 10000
