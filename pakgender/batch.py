"""
batch.py
--------
predict_series(series) — run predict() over a pandas Series of names.
Returns a DataFrame with columns [gender, confidence, source, matched_token].
"""

import pandas as pd
from .predictor import Predictor

_predictor: Predictor | None = None


def predict_series(series: pd.Series, use_ml: bool = True) -> pd.DataFrame:
    """
    Predict gender for every name in a pandas Series.

    Parameters
    ----------
    series : pd.Series
        A Series of name strings.
    use_ml : bool
        Whether to use the ML layer (default True).

    Returns
    -------
    pd.DataFrame
        Columns: gender, confidence, source, matched_token.
        Index matches the input Series index.

    Example
    -------
    >>> df[["gender","confidence"]] = predict_series(df["CNIC_Name"])[["gender","confidence"]]
    """
    global _predictor
    if _predictor is None or _predictor._ml is None and use_ml:
        _predictor = Predictor(use_ml=use_ml)

    results = series.apply(lambda name: _predictor.predict(name))

    return pd.DataFrame(
        {
            "gender":        [r.gender        for r in results],
            "confidence":    [r.confidence    for r in results],
            "source":        [r.source        for r in results],
            "matched_token": [r.matched_token for r in results],
        },
        index=series.index,
    )
