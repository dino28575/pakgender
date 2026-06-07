"""
ml_model.py
-----------
Layer 3: Character n-gram Logistic Regression.
Loaded from the bundled model.pkl if it exists; silently disabled otherwise.

Train with: python train_model.py

IMPORTANT: model was trained with ^ $ boundary padding.
           predict() applies the same padding before calling the pipeline.
"""

import importlib.resources
import pickle
from typing import Optional
from ._version import GenderResult


class MLModel:
    def __init__(self):
        self._pipeline = None
        self._load()

    def _load(self):
        try:
            ref = importlib.resources.files("pakgender.data").joinpath("model.pkl")
            with importlib.resources.as_file(ref) as path:
                with open(path, "rb") as f:
                    self._pipeline = pickle.load(f)
        except Exception:
            self._pipeline = None  # model not yet trained — graceful degradation

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    def predict(self, token: str) -> Optional[GenderResult]:
        if not self.is_ready:
            return None
        try:
            # Apply same ^ $ padding used during training
            padded = f"^{token.lower().strip()}$"
            gender = self._pipeline.predict([padded])[0]
            proba  = self._pipeline.predict_proba([padded])[0]
            confidence = float(max(proba))
            if confidence < 0.60:
                return None   # not confident enough — return None so predictor stays U
            return GenderResult(
                gender=gender,
                confidence=round(confidence, 2),
                source="ml",
                matched_token=token,
            )
        except Exception:
            return None
