"""Input validation tests using synthetic observations only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examkit.validation import (
    load_marginal_choices, validate_holdings, validate_missing_frame,
    validate_price_frame, validate_return_frame,
)


@pytest.fixture
def prices():
    return pd.DataFrame({"Date": ["d1", "d2", "d3", "d4"],
                         "X": [100., 101., 99., 102.], "Y": [80., 81., 82., 80.]})


@pytest.fixture
def holdings():
    return pd.DataFrame({"Stock": ["Y", "X"], "Book": ["B", "A"], "Holding": [20., 10.]})


def test_price_validation_returns_only_ordered_numeric_assets(prices):
    result = validate_price_frame(prices, ["Y", "X"])
    assert list(result) == ["Y", "X"]
    assert result.index.equals(prices.index)
    np.testing.assert_allclose(result.to_numpy(), prices[["Y", "X"]].to_numpy())
    assert "Date" in prices and list(prices) == ["Date", "X", "Y"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0., -1., True, "not-a-price", 1+2j])
def test_invalid_prices_are_rejected(prices, value):
    prices["X"] = prices.X.astype(object)
    prices.loc[0, "X"] = value
    with pytest.raises(ValueError):
        validate_price_frame(prices)


@pytest.mark.parametrize("date", [None, "", " ", "d2"])
def test_invalid_dates_are_rejected(prices, date):
    prices.loc[0, "Date"] = date
    with pytest.raises(ValueError, match="Date"):
        validate_price_frame(prices)


def test_price_shape_and_required_asset_errors(prices):
    for frame in [prices.drop(columns="Date"), prices.iloc[:2], prices[["Date"]]]:
        with pytest.raises(ValueError):
            validate_price_frame(frame)
    with pytest.raises(ValueError, match="exactly"):
        validate_price_frame(prices, ["X"])
    with pytest.raises(ValueError, match="unique"):
        validate_price_frame(prices, ["X", "X"])
    repeated = pd.concat([prices, prices[["X"]]], axis=1)
    with pytest.raises(ValueError, match="unique"):
        validate_price_frame(repeated)


def test_return_validation_accepts_finite_nonconstant_columns():
    frame = pd.DataFrame({"X": np.linspace(-.03, .02, 6), "Y": np.linspace(.01, -.01, 6)})
    pd.testing.assert_frame_equal(validate_return_frame(frame), frame)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, "not-a-return"])
def test_invalid_return_values_are_rejected(value):
    frame = pd.DataFrame({"X": np.arange(6, dtype=float)}).astype(object)
    frame.loc[0, "X"] = value
    with pytest.raises(ValueError):
        validate_return_frame(frame)


def test_return_validation_rejects_small_constant_and_empty_inputs():
    for frame in [pd.DataFrame({"X": [1., 2.]}), pd.DataFrame({"X": [1.] * 6}),
                  pd.DataFrame(index=range(6))]:
        with pytest.raises(ValueError):
            validate_return_frame(frame)


def test_missing_validation_preserves_missing_values_and_uses_complete_pairs():
    frame = pd.DataFrame({"X": [1., 2., 3., np.nan], "Y": [2., 4., np.nan, 6.]})
    pd.testing.assert_frame_equal(validate_missing_frame(frame), frame)


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"X": [1., 2.]}),
    pd.DataFrame({"X": [1., 2.], "Y": [2., np.inf]}),
    pd.DataFrame({"X": [1., 2.], "Y": [2., np.nan]}),
    pd.DataFrame({"X": [1., 2.], "Y": [2., 2.]}),
    pd.DataFrame({"X": [1., 2., np.nan], "Y": [np.nan, 2., 3.]}),
    pd.DataFrame({"X": [1., 1., 2.], "Y": [2., 3., np.nan]}),
    pd.DataFrame({"X": [True, False], "Y": [2., 3.]}),
])
def test_unusable_missing_data_inputs_are_rejected(frame):
    with pytest.raises(ValueError):
        validate_missing_frame(frame)


def test_holdings_are_ordered_and_indexed_without_mutation(holdings):
    original = holdings.copy(deep=True)
    result = validate_holdings(holdings, ["X", "Y"])
    assert result.index.name == "Stock" and list(result.index) == ["X", "Y"]
    assert result.Holding.tolist() == [10., 20.]
    pd.testing.assert_frame_equal(holdings, original)


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf, -np.inf, True, "not-a-holding"])
def test_nonpositive_or_invalid_holdings_are_rejected(holdings, value):
    holdings["Holding"] = holdings.Holding.astype(object)
    holdings.loc[0, "Holding"] = value
    with pytest.raises(ValueError):
        validate_holdings(holdings, ["X", "Y"])


def test_holdings_reject_duplicate_missing_extra_and_invalid_books(holdings):
    malformed = [holdings.drop(columns="Holding"), holdings.iloc[:1],
                 pd.concat([holdings, holdings.iloc[:1]], ignore_index=True)]
    for frame in malformed:
        with pytest.raises(ValueError):
            validate_holdings(frame, ["X", "Y"])
    for column, value in [("Stock", None), ("Stock", "Z"),
                          ("Book", "C"), ("Book", None), ("Book", "A")]:
        frame = holdings.copy()
        frame.loc[0, column] = value
        with pytest.raises(ValueError):
            validate_holdings(frame, ["X", "Y"])


def test_choices_none_and_explicit_synthetic_mapping(tmp_path):
    assert load_marginal_choices(None, ["X", "Y"]) is None
    path = tmp_path / "choices.json"
    path.write_text(json.dumps({"Y": "t", "X": "normal"}), encoding="utf-8")
    choices = load_marginal_choices(path, ["X", "Y"])
    assert choices == {"X": "normal", "Y": "t"}
    assert list(choices) == ["X", "Y"]


@pytest.mark.parametrize("text", [
    '{"X":"normal","X":"t","Y":"nig"}',
    '{"X":"normal","Y":null}', '{"X":"normal","Y":""}',
    '{"X":"normal","Y":"invalid"}', '{"X":"normal"}',
    '{"X":"normal","Y":"t","Z":"nig"}',
    '{"X":"normal","Y":true}', '[]', 'null', '{bad json}',
])
def test_choices_reject_duplicates_missing_and_invalid_entries(tmp_path, text):
    path = tmp_path / "choices.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_marginal_choices(path, ["X", "Y"])
