__version__ = "0.1.5"

from dataclasses import dataclass


@dataclass(frozen=True)
class GenderResult:
    """
    Immutable result returned by predict().

    Attributes
    ----------
    gender : str
        'M' (male), 'F' (female), or 'U' (unknown/ambiguous).
    confidence : float
        Score between 0.0 and 1.0. Higher = more certain.
        >= 0.75  dict hit
        >= 0.65  rule hit
        >= 0.50  ML prediction
        <  0.50  unknown
    source : str
        Which layer produced the answer: 'dict', 'rule', or 'ml'.
    matched_token : str
        The specific token (part of the name) that triggered the result.
    """
    gender: str
    confidence: float
    source: str
    matched_token: str

    def __repr__(self) -> str:
        return (
            f"GenderResult(gender={self.gender!r}, "
            f"confidence={self.confidence:.2f}, "
            f"source={self.source!r}, "
            f"matched_token={self.matched_token!r})"
        )

    def is_known(self) -> bool:
        """Return True if gender is not unknown."""
        return self.gender != "U"
