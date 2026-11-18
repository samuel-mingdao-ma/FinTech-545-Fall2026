"""Price-to-return conversions, retaining dates and original asset order."""
import numpy as np
import pandas as pd


def calculate_returns(prices, method="arithmetic", date_column="Date"):
    """Return n-1 observations indexed by each ending observation's date.

    Prices must be complete, finite and strictly positive. No sorting, filling,
    demeaning or percentage rescaling is performed. The first row is omitted.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if date_column not in prices or prices.columns.duplicated().any():
        raise ValueError("Unique columns and the specified date column are required")
    if len(prices) < 2 or prices.shape[1] < 2:
        raise ValueError("Need two price observations and at least one asset")
    if prices[date_column].isna().any():
        raise ValueError("Dates must not be missing")
    assets = prices.columns.drop(date_column)
    values = prices[assets].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite, complete and strictly positive")
    if method.lower() in ("arithmetic", "discrete"):
        result = values[1:] / values[:-1] - 1
    elif method.lower() == "log":
        # The ratio can overflow or underflow even when both prices and the
        # log return are finite. Keep ratio accuracy for ordinary price moves.
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            result = np.log(values[1:] / values[:-1])
        extreme = ~np.isfinite(result)
        result[extreme] = np.log(values[1:][extreme]) - np.log(values[:-1][extreme])
    else:
        raise ValueError("method must be arithmetic or log")
    frame = pd.DataFrame(result, columns=assets)
    frame[date_column] = prices[date_column].iloc[1:].to_numpy()
    return frame.loc[:, prices.columns]
