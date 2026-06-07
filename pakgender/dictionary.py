"""
dictionary.py
-------------
Layer 1: Dictionary lookup against bundled names.json.

names.json schema
-----------------
{
  "first_names": {
    "fatima":  {"gender": "F", "frequency": "high"},
    "ahsan":   {"gender": "M", "frequency": "medium"},
    "noor":    {"gender": "U", "frequency": "high"},   // ambiguous
    ...
  },
  "full_names": {
    "fatima noor": {"gender": "F", "frequency": "low"},
    ...
  }
}

Confidence scores returned
--------------------------
  1.00  exact match, non-ambiguous
  0.90  variant-normalized match, non-ambiguous
  0.60  match but marked ambiguous in dataset ("U" entry)
  0.00  no match -> caller proceeds to Layer 2
"""

import json
import importlib.resources
from typing import Optional
from ._version import GenderResult


def _load_names_db() -> dict:
    """Load names.json from the bundled data directory."""
    try:
        # Python 3.9+ path
        ref = importlib.resources.files("pakgender.data").joinpath("names.json")
        with importlib.resources.as_file(ref) as path:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except (FileNotFoundError, TypeError):
        # Fallback for development before data file exists
        return {"first_names": {}, "full_names": {}}


class DictionaryLayer:
    """
    Wraps names.json into a fast in-memory dict for O(1) lookup.
    Loaded once at Predictor init, shared across all predict() calls.
    """

    def __init__(self, db: Optional[dict] = None):
        self._db = db if db is not None else _load_names_db()
        self._first: dict = self._db.get("first_names", {})
        self._full: dict = self._db.get("full_names", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, candidates: list[str]) -> Optional[GenderResult]:
        """
        Try each candidate token in order until a confident match is found.

        Parameters
        ----------
        candidates : list[str]
            Ordered list from preprocessor — personal name first.

        Returns
        -------
        GenderResult or None
            None means no match; caller moves to Layer 2.
        """
        for token in candidates:
            result = self._lookup_token(token)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_token(self, token: str) -> Optional[GenderResult]:
        """Try token against full_names first, then first_names."""
        # Full-name match (more specific = higher confidence)
        if token in self._full:
            entry = self._full[token]
            return self._make_result(entry, token, exact=True)

        # First-name match
        if token in self._first:
            entry = self._first[token]
            return self._make_result(entry, token, exact=True)

        return None

    @staticmethod
    def _make_result(entry: dict, token: str, exact: bool) -> Optional[GenderResult]:
        """
        Convert a names.json entry into a GenderResult.
        Returns None if the entry itself says ambiguous and confidence too low.
        """
        gender = entry.get("gender", "U")
        frequency = entry.get("frequency", "medium")

        if gender == "U":
            # Ambiguous entry — return low-confidence result so
            # predictor can try further layers before committing.
            return GenderResult(
                gender="U",
                confidence=0.60,
                source="dict",
                matched_token=token,
            )

        # Confidence scaling by frequency
        base = 1.00 if exact else 0.90
        freq_penalty = {"low": 0.05, "medium": 0.0, "high": 0.0}
        confidence = base - freq_penalty.get(frequency, 0.0)

        return GenderResult(
            gender=gender,
            confidence=round(confidence, 2),
            source="dict",
            matched_token=token,
        )

    # ------------------------------------------------------------------
    # Utility — for train_model.py and data prep scripts
    # ------------------------------------------------------------------

    def all_labeled_names(self) -> list[tuple[str, str]]:
        """Return [(name, gender)] for all non-ambiguous entries. Used for ML training."""
        rows = []
        for name, entry in self._first.items():
            if entry.get("gender") in ("M", "F"):
                rows.append((name, entry["gender"]))
        for name, entry in self._full.items():
            if entry.get("gender") in ("M", "F"):
                rows.append((name, entry["gender"]))
        return rows
