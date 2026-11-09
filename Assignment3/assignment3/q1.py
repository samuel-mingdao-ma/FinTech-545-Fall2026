"""Problem 1(a-h): quote diagnostics and implied-volatility inversion."""
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import write_json, write_predictions
from .pricing import greeks, implied_volatility, price, price_bounds


def run(data_dir, output_dir, figure_dir, **kwargs):
    data_dir, out, figures = map(Path, (data_dir, output_dir, figure_dir))
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    s, r, q = 182.40, .0425, .0060
    today, expiry = date(2026, 11, 13), date(2026, 12, 18)
    t, b = (expiry-today).days/365, r-q
    quotes = pd.read_csv(data_dir / "problem1.csv")
    if set(quotes.Type) != {"Call", "Put"} or quotes.duplicated(["Type", "Strike"]).any():
        raise ValueError("Expected unique call and put quotes at every strike")
    if (quotes.Bid < 0).any() or (quotes.Ask < quotes.Bid).any():
        raise ValueError("Invalid bid/ask quote")
    quotes["Mid"] = (quotes.Bid+quotes.Ask)/2
    quotes["Spread"] = quotes.Ask-quotes.Bid
    calls, puts = [quotes.loc[quotes.Type == kind].set_index("Strike").sort_index() for kind in ("Call", "Put")]
    if not calls.index.equals(puts.index):
        raise ValueError("Call and put strike sets differ")

    # Problem 1(a): parity is checked BEFORE any IV is inverted.
    parity = pd.DataFrame(index=calls.index)
    parity["call_mid"], parity["put_mid"] = calls.Mid, puts.Mid
    parity["theoretical_call_minus_put"] = s*np.exp(-q*t)-parity.index*np.exp(-r*t)
    parity["gap"] = parity.call_mid-parity.put_mid-parity.theoretical_call_minus_put
    parity["band"] = calls.Spread+puts.Spread
    parity["flag"] = parity.gap.abs() > parity.band
    parity.to_csv(out / "q1_parity.csv")
    flagged = [float(k) for k in parity.index[parity.flag]]

    # Problem 1(b/c): preserve the first analytical prediction before inversion.
    write_predictions(out / "q1_predictions.json", dict(
        a="The midpoint parity gaps above the complete call-plus-put spread band identify inconsistent quotes, not an independently tradable arbitrage guarantee.",
        b="Under a common carry/rate and a unique IV root, exact put-call parity implies equal call and put IV at the same strike: both price functions differ by the fixed parity term.",
        c="Low-vega far-wing quotes amplify a one-cent price error into a large IV error. The identical 0.05/0.10 call quotes at strikes above 210 suggest a quote floor and an unreliable apparent wing before any IV fit.",
        parity_flagged_strikes=flagged,
    ))

    # Problem 1(d/e): invert bid/mid/ask independently, preserving missing roots.
    rows = []
    for row in quotes.itertuples(index=False):
        is_call = row.Type == "Call"
        lower, upper = price_bounds(s, row.Strike, t, r, b, is_call)
        item = dict(Type=row.Type, Strike=row.Strike, Bid=row.Bid, Mid=row.Mid, Ask=row.Ask,
                    lower_bound=lower, upper_bound=upper)
        for source in ("Bid", "Mid", "Ask"):
            iv, status = implied_volatility(getattr(row, source), s, row.Strike, t, r, b, is_call)
            item[f"IV_{source.lower()}"] = iv
            item[f"status_{source.lower()}"] = status
            if iv is not None:
                np.testing.assert_allclose(price(s, row.Strike, t, r, b, iv, is_call),
                                           getattr(row, source), atol=3e-8, rtol=0)
        iv = item["IV_mid"]
        if iv is not None and iv > 0:
            vega = greeks(s, row.Strike, t, r, b, iv, is_call)["vega"]
            item["vega_per_unit_sigma"] = vega
            item["vega_per_vol_point"] = vega/100
            item["cent_dIV_linear"] = .01/vega
            item["cent_dIV_vol_points"] = 100*.01/vega
            bumped, status = implied_volatility(row.Mid+.01, s, row.Strike, t, r, b, is_call)
            item["cent_dIV_exact_up"] = None if bumped is None else bumped-iv
            item["cent_up_status"] = status
        rows.append(item)
    fitted = pd.DataFrame(rows)
    fitted.to_csv(out / "q1_quotes_iv.csv", index=False)

    # Problem 1(f): the parity residual is the integral of vega between the IVs.
    translations = []
    for k in parity.index:
        fc = fitted.loc[(fitted.Type == "Call") & (fitted.Strike == k)].iloc[0]
        fp = fitted.loc[(fitted.Type == "Put") & (fitted.Strike == k)].iloc[0]
        if pd.notna(fc.IV_mid) and pd.notna(fp.IV_mid) and fc.IV_mid+fp.IV_mid > 0:
            reference_iv = (fc.IV_mid+fp.IV_mid)/2
            reference_vega = greeks(s, k, t, r, b, reference_iv, True)["vega"]
            actual_gap = fc.IV_mid-fp.IV_mid
            translations.append(dict(Strike=k, flag=bool(parity.loc[k, "flag"]),
                                     parity_gap=parity.loc[k, "gap"], reference_iv=reference_iv,
                                     reference_vega=reference_vega,
                                     estimated_IV_gap=parity.loc[k, "gap"]/reference_vega,
                                     actual_IV_gap=actual_gap,
                                     estimated_gap_vol_points=100*parity.loc[k, "gap"]/reference_vega,
                                     actual_gap_vol_points=100*actual_gap))
    translated = pd.DataFrame(translations)
    translated.to_csv(out / "q1_parity_vol_translation.csv", index=False)

    # Problem 1(g/h): retain an inspectable OTM smile after the two quoted issues.
    clean = fitted.loc[((fitted.Type == "Put") & (fitted.Strike < s)) |
                       ((fitted.Type == "Call") & (fitted.Strike >= s))].copy()
    clean = clean.loc[~clean.Strike.isin(flagged) & ~((clean.Type == "Call") & (clean.Strike > 210))]
    clean = clean.sort_values("Strike")
    clean.to_csv(out / "q1_clean_otm_smile.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for kind, color in (("Call", "#235789"), ("Put", "#bd632f")):
        group = fitted.loc[fitted.Type == kind].sort_values("Strike")
        ax.plot(group.Strike, 100*group.IV_mid, "o-", color=color, label=f"{kind} mid IV", ms=4)
        ax.plot(group.Strike, 100*group.IV_bid, ":", color=color, alpha=.7)
        ax.plot(group.Strike, 100*group.IV_ask, ":", color=color, alpha=.7)
        ax.fill_between(group.Strike.to_numpy(), 100*group.IV_bid.to_numpy(),
                        100*group.IV_ask.to_numpy(), color=color, alpha=.12)
    for k in flagged:
        ax.axvline(k, color="#a71d31", ls="--", lw=1, label=f"Parity flag K={k:g}")
    ax.axvspan(210, 226, color="grey", alpha=.08, label="Call quote-floor wing")
    ax.set(xlabel="Strike ($)", ylabel="Implied volatility (%)", title="Q1: mid IV with bid-ask IV bands")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "q1_smile.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True)
    for kind, color in (("Call", "#235789"), ("Put", "#bd632f")):
        group = fitted.loc[fitted.Type == kind].sort_values("Strike")
        axes[0].plot(group.Strike, group.vega_per_vol_point, "o-", label=kind, color=color, ms=3)
        axes[1].plot(group.Strike, group.cent_dIV_vol_points, "o-", label=kind, color=color, ms=3)
    axes[0].set(ylabel="Vega ($ per vol point)", title="Q1: inverse-price sensitivity")
    axes[1].set(xlabel="Strike ($)", ylabel="Vol points per $0.01")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / "q1_vega_sensitivity.png", dpi=180)
    plt.close(fig)

    missing = []
    for source in ("bid", "mid", "ask"):
        for row in fitted.loc[fitted[f"IV_{source}"].isna()].itertuples(index=False):
            missing.append(dict(Type=row.Type, Strike=row.Strike, source=source,
                                status=getattr(row, f"status_{source}")))
    summary = dict(parameters=dict(spot=s, rate=r, dividend_yield=q, valuation_date=str(today),
                                   expiration_date=str(expiry), calendar_days=(expiry-today).days, t=t),
                   flagged_strikes=flagged,
                   flagged_parity=parity.loc[parity.flag].reset_index().to_dict("records"),
                   flagged_vol_translation=translated.loc[translated.flag].to_dict("records"),
                   missing_IV_roots=missing,
                   call_wing=fitted.loc[(fitted.Type == "Call") & (fitted.Strike > 210)].to_dict("records"),
                   clean_smile=clean[["Type", "Strike", "IV_mid"]].to_dict("records"),
                   all_mid_repricing_checked=True,
                   files=["q1_parity.csv", "q1_quotes_iv.csv", "q1_parity_vol_translation.csv", "q1_clean_otm_smile.csv"],
                   figures=["q1_smile.png", "q1_vega_sensitivity.png"])
    write_json(out / "q1_summary.json", summary)
    return summary
