"""
pakgender
~~~~~~~~~
Gender inference for Pakistani names.

Quick start::

    from pakgender import predict, predict_series

    result = predict("Fatima Noor")
    print(result)
    # GenderResult(gender='F', confidence=0.97, source='dict', matched_token='Fatima')

    import pandas as pd
    df = pd.DataFrame({"name": ["Muhammad Ahsan", "Zara Malik", "Noor Ali"]})
    df[["gender", "confidence"]] = predict_series(df["name"])
"""

from .predictor import Predictor
from .batch import predict_series
from ._version import __version__

__all__ = ["predict", "predict_series", "Predictor", "__version__"]

_default_predictor: Predictor | None = None


def predict(name: str) -> "GenderResult":
    """
    Predict gender for a single name string.

    Parameters
    ----------
    name : str
        Any Pakistani name in Roman script, e.g. "Muhammad Ahsan Raza",
        "Fatima", "Begum Zara", "Ayesha".

    Returns
    -------
    GenderResult
        A named result with fields: gender, confidence, source, matched_token.
        gender is 'M', 'F', or 'U' (unknown).
    """
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = Predictor()
    return _default_predictor.predict(name)
