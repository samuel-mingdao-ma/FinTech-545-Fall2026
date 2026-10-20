"""Input validation shared by the calculation entry points."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _columns(frame):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("Input column names must be unique")
    if not len(frame.columns) or any(
        not isinstance(name, str) or not name.strip() for name in frame.columns
    ):
        raise ValueError("Input columns must have nonempty string names")


def _asset_names(assets):
    names = list(assets)
    if (not names or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Asset names must be nonempty, unique strings")
    return names


def _numeric(frame, *, allow_missing=False):
    _columns(frame)
    values = frame.to_numpy(dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in values.flat):
        raise ValueError("Boolean values are not numeric observations")
    if any(isinstance(value, (complex, np.complexfloating)) for value in values.flat):
        raise ValueError("Complex values are not supported")
    try:
        converted = frame.apply(pd.to_numeric, errors="raise")
        array = converted.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Observations must be numeric") from exc
    if np.isinf(array).any():
        raise ValueError("Infinite observations are not allowed")
    if not allow_missing and not np.isfinite(array).all():
        raise ValueError("Observations must be finite and complete")
    return pd.DataFrame(array, index=frame.index, columns=frame.columns)


def validate_price_frame(frame, required_assets=None):
    """Return positive numeric asset prices after checking the Date column."""
    _columns(frame)
    if "Date" not in frame.columns:
        raise ValueError("Price data requires a Date column")
    if len(frame) < 3:
        raise ValueError("Price data requires at least three rows")
    dates = frame["Date"]
    if dates.isna().any() or dates.astype(str).str.strip().eq("").any():
        raise ValueError("Date values must be nonmissing")
    if dates.duplicated().any():
        raise ValueError("Date values must be unique")
    prices = frame.drop(columns="Date")
    if required_assets is not None:
        assets = _asset_names(required_assets)
        if set(prices.columns) != set(assets):
            raise ValueError("Price columns must exactly match the required assets")
        prices = prices.loc[:, assets]
    numeric = _numeric(prices)
    if (numeric <= 0).any().any():
        raise ValueError("Prices must be strictly positive")
    return numeric


def validate_return_frame(frame):
    """Return finite, nonconstant numeric return columns with sufficient rows."""
    if not isinstance(frame, pd.DataFrame) or len(frame) < 6:
        raise ValueError("Return data requires at least six rows")
    numeric = _numeric(frame)
    if (numeric.nunique(dropna=False) < 2).any():
        raise ValueError("Return columns must be nonconstant")
    return numeric


def validate_missing_frame(frame):
    """Validate numeric data with missing entries and usable complete pairs."""
    _columns(frame)
    if len(frame.columns) < 2:
        raise ValueError("Missing-data calculations require at least two series")
    numeric = _numeric(frame, allow_missing=True)
    for name in numeric:
        column = numeric[name].dropna()
        if len(column) < 2 or column.nunique() < 2:
            raise ValueError(f"Series {name} requires two or more nonconstant observations")
    for i, left in enumerate(numeric.columns):
        for right in numeric.columns[i + 1:]:
            pair = numeric[[left, right]].dropna()
            if len(pair) < 2 or (pair.nunique() < 2).any():
                raise ValueError(f"Pair {left}/{right} requires two or more nonconstant paired observations")
    return numeric


def validate_holdings(frame, assets):
    """Return ordered holdings for strictly positive long positions in A/B."""
    _columns(frame)
    assets = _asset_names(assets)
    if not {"Stock", "Book", "Holding"}.issubset(frame.columns):
        raise ValueError("Holdings require Stock, Book, and Holding columns")
    if frame.Stock.isna().any() or frame.Stock.duplicated().any():
        raise ValueError("Holdings stock IDs must be nonmissing and unique")
    if set(frame.Stock) != set(assets):
        raise ValueError("Holdings must exactly match the asset set")
    if set(frame.Book) != {"A", "B"}:
        raise ValueError("Book labels must be A or B, with both books present")
    amounts = _numeric(frame[["Holding"]])["Holding"]
    if (amounts <= 0).any():
        raise ValueError("Holding must be strictly positive; analytic checks support long-only positions")
    ordered = frame.copy()
    ordered["Holding"] = amounts.to_numpy()
    return ordered.set_index("Stock").loc[assets]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_marginal_choices(path, stocks):
    """Read an explicit complete family mapping without defaults or selection."""
    if path is None:
        return None
    stocks = _asset_names(stocks)
    choices = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if (not isinstance(choices, dict) or set(choices) != set(stocks)
            or any(not isinstance(family, str) or family not in ("normal", "t", "nig")
                   for family in choices.values())):
        raise ValueError("Choices must supply normal/t/nig for every stock; null/blank values are not allowed")
    return {stock: choices[stock] for stock in stocks}
