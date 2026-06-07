"""
predictor.py
------------
Orchestrates the full 3-layer pipeline with confidence-based ensemble voting.

Architecture
------------
Old design (waterfall):
    Dict hit → return immediately, ignoring Rule and ML completely.

New design (per-token voting with definitive female override):
    Step 0 — Definitive female check (any token is an unambiguous female word)
              → immediately return F with confidence 1.0, no voting needed.
    Step 1 — For each candidate token, collect results from ALL three layers.
    Step 2 — Combine into a single VoteResult using weighted confidence.
    Step 3 — Pick best token-level result across all candidate tokens.

    KEY IMPROVEMENT (from key_observations):
    In Pakistani naming, males NEVER include feminine words in their names.
    Therefore: if ANY token in the name is a definitive female word
    (bibi, begum, bano, fatima, khatoon, sultana, kanwal, parveen, etc.),
    the ENTIRE name is female — regardless of what the first token says.

    The old fast-path (high-freq dict hit on token[0]) is now gated behind
    the definitive-female check, so "IRSHAD FATIMA" correctly returns F
    even though "irshad" is M in the dictionary.

Confidence weights per source
------------------------------
    dict  (verified human label)  →  weight 1.15 if high-freq, 1.0 otherwise
    rule  (linguistic pattern)    →  weight 0.90
    ml    (learned from data)     →  weight 0.95

Aggregation within one token
-----------------------------
    Collect all layer results for the token.
    Separate into M-votes and F-votes.
    Weighted-average confidence per gender side.
    Winner = gender with higher weighted confidence.
    Final confidence = winner_conf - loser_conf  (margin, not raw prob).
    If margin < 0.10 and no clear linguistic signal → return U.

Token selection across candidates
-----------------------------------
    Personal name token (index 0) gets position_weight = 1.0
    Father name token (index 1) gets position_weight = 0.80
    Remaining tokens   get position_weight = 0.60
    
    Best token = argmax(final_confidence × position_weight)
    Exception: if any token has a dict hit with freq=high → that token wins.
"""

import re
from dataclasses import dataclass
from typing import Optional

from .preprocessor import preprocess
from .dictionary import DictionaryLayer
from .rules import RuleEngine
from .ml_model import MLModel
from ._version import GenderResult


# ── Source weights ─────────────────────────────────────────────────────────
_W_DICT_HIGH   = 1.15   # dict hit, high-frequency name
_W_DICT_OTHER  = 1.00   # dict hit, medium/low frequency
_W_RULE        = 0.90   # rule engine result
_W_ML          = 0.95   # ML model result

# ── Decision thresholds ────────────────────────────────────────────────────
_MARGIN_CERTAIN    = 0.20   # margin above this → high confidence
_MARGIN_MINIMUM    = 0.08   # margin below this → return U (too close to call)
_ML_OVERRIDE_DICT  = 0.80   # ML must exceed this to override a dict M/F result
_DICT_U_THRESHOLD  = 0.61   # dict confidence at or below this = treat as ambiguous

# ── Position weights for multi-token names ─────────────────────────────────
_POS_WEIGHTS = [1.0, 0.80, 0.60, 0.50]  # index 0 = personal name

# ── Name validity for ML ───────────────────────────────────────────────────
_VOWELS      = frozenset("aeiou")
_NAME_PATT   = re.compile(r'^[a-z\-]{2,}$')


@dataclass
class _TokenVote:
    """Intermediate result for one candidate token."""
    token:      str
    gender:     str      # 'M', 'F', or 'U'
    confidence: float    # 0.0 – 1.0, adjusted margin
    source:     str      # dominant source: 'dict', 'rule', 'ml', 'ensemble'
    detail:     dict     # per-layer breakdown for debugging


class Predictor:
    """
    Main entry point. Instantiate once; reuse across many predict() calls.
    All three layers are loaded at init time (lazy-loaded on first predict).
    """

    def __init__(self, use_ml: bool = True):
        self._dict  = DictionaryLayer()
        self._rules = RuleEngine()
        self._ml    = MLModel() if use_ml else None

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, name: str) -> GenderResult:
        """
        Predict gender for a single name string.

        Parameters
        ----------
        name : str
            Raw name in Roman script, e.g. "Muhammad Ahsan Raza".

        Returns
        -------
        GenderResult
            gender: 'M', 'F', or 'U'
            confidence: 0.0 – 1.0
            source: 'dict', 'rule', 'ml', or 'ensemble'
            matched_token: the token that produced the result
        """
        if not name or not str(name).strip():
            return GenderResult(gender="U", confidence=0.0,
                                source="none", matched_token="")

        prep = preprocess(str(name))

        if not prep.candidates:
            # Only honorific signal available — use it at low confidence
            if prep.honorific_signal:
                return GenderResult(
                    gender=prep.honorific_signal, confidence=0.60,
                    source="rule", matched_token="[honorific]",
                )
            return GenderResult(gender="U", confidence=0.0,
                                source="none", matched_token="")

        # ── STEP 0: Definitive female check ──────────────────────────────
        # In Pakistani naming, a male name NEVER contains a feminine word.
        # If ANY token is a definitive female word → entire name is female.
        # This must fire BEFORE the fast path to correctly handle:
        #   IRSHAD FATIMA, MAQSOOD BEGUM, ALTAF BIBI, etc.
        definitive = self._rules.check_definitive_female(prep.tokens)
        if definitive:
            return GenderResult(
                gender="F",
                confidence=1.0,
                source="rule",
                matched_token=definitive.matched_token,
            )

        # ── STEP 0b: Honorific signal shortcut ───────────────────────────
        # If preprocessor detected a strong female honorific (Mst., Begum, etc.)
        # and it hasn't already been caught above, return F immediately.
        if prep.honorific_signal == "F":
            return GenderResult(
                gender="F",
                confidence=0.95,
                source="rule",
                matched_token="[honorific]",
            )

        # ── STEP 1: Fast path for high-frequency dict hits ────────────────
        # Only fires if the personal name token is high-frequency M or F.
        fast = self._try_fast_path(prep.candidates[0])
        if fast:
            return fast

        # ── STEP 2: Full voting across all candidate tokens ────────────────
        token_votes: list[_TokenVote] = []
        for i, token in enumerate(prep.candidates):
            vote = self._vote_on_token(token, position=i)
            token_votes.append(vote)

        # ── STEP 3: Select best token ──────────────────────────────────────
        best = self._select_best_token(token_votes, prep.honorific_signal)
        return best

    def explain(self, name: str) -> list[dict]:
        """
        Return per-token voting breakdown for debugging.

        Example
        -------
        >>> from pakgender import Predictor
        >>> p = Predictor()
        >>> p.explain("Samreen Akhtar")
        [{'token': 'samreen', 'gender': 'F', 'confidence': 0.95, ...}, ...]
        """
        prep = preprocess(str(name))
        results = []

        # Show definitive female check result
        definitive = self._rules.check_definitive_female(prep.tokens)
        if definitive:
            results.append({
                "token":      definitive.matched_token,
                "position":   -1,
                "gender":     "F",
                "confidence": 1.0,
                "source":     "definitive_female_word",
                "layers":     {"note": "Definitive female word — entire name is F"},
            })

        for i, token in enumerate(prep.candidates):
            vote = self._vote_on_token(token, position=i)
            results.append({
                "token":      vote.token,
                "position":   i,
                "gender":     vote.gender,
                "confidence": round(vote.confidence, 3),
                "source":     vote.source,
                "layers":     vote.detail,
            })
        return results

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _try_fast_path(self, token: str) -> Optional[GenderResult]:
        """
        Immediate return for high-frequency dict hits — no voting needed.
        These are the most common Pakistani names and their labels are reliable.
        
        Note: Only fires AFTER the definitive-female check (Step 0), so
        high-freq male names like 'irshad' won't incorrectly dominate
        when 'fatima' appears later in the name.
        """
        dr = self._dict.lookup([token])
        if (dr and dr.gender in ("M", "F")
                and dr.confidence >= 1.0
                and self._dict._first.get(token, {}).get("frequency") == "high"):
            return GenderResult(
                gender=dr.gender,
                confidence=dr.confidence,
                source="dict",
                matched_token=token,
            )
        return None

    def _vote_on_token(self, token: str, position: int) -> _TokenVote:
        """
        Run all three layers on a single token and combine their votes.
        Returns a _TokenVote with the resolved gender and confidence.
        """
        # Collect raw layer results
        dr  = self._dict.lookup([token])
        rr  = self._rules.lookup([token])
        mlr = self._ml.predict(token) if (
                self._ml and self._ml.is_ready
                and self._is_name_like(token)) else None

        detail = {
            "dict": f"{dr.gender}@{dr.confidence:.2f}" if dr else None,
            "rule": f"{rr.gender}@{rr.confidence:.2f}" if rr else None,
            "ml":   f"{mlr.gender}@{mlr.confidence:.2f}" if mlr else None,
        }

        # ── Special case: dict says U (ambiguous) ─────────────────────────
        # Don't let an ambiguous dict entry block good rule/ML signals.
        dict_is_ambiguous = (dr is None or
                             dr.gender == "U" or
                             dr.confidence <= _DICT_U_THRESHOLD)

        # ── Gather weighted votes per gender side ─────────────────────────
        m_score = 0.0
        f_score = 0.0
        dominant_source = "none"

        def add_vote(result, weight, source_name):
            nonlocal m_score, f_score, dominant_source
            if result is None:
                return
            if result.gender == "M":
                m_score += result.confidence * weight
            elif result.gender == "F":
                f_score += result.confidence * weight

        # Dict vote — full weight only if not ambiguous
        if not dict_is_ambiguous and dr:
            freq = self._dict._first.get(token, {}).get("frequency", "medium")
            w = _W_DICT_HIGH if freq == "high" else _W_DICT_OTHER
            add_vote(dr, w, "dict")
        
        # Rule vote
        add_vote(rr, _W_RULE, "rule")

        # ML vote — upweight ML at second+ positions to let it override
        # a male first token when second token is clearly female
        ml_weight = _W_ML if position == 0 else _W_ML * 1.10
        add_vote(mlr, ml_weight, "ml")

        # ── Resolve winner ────────────────────────────────────────────────
        total = m_score + f_score
        if total == 0:
            return _TokenVote(token=token, gender="U", confidence=0.0,
                              source="none", detail=detail)

        if m_score > f_score:
            winner, loser = "M", "F"
            winner_score, loser_score = m_score, f_score
        else:
            winner, loser = "F", "M"
            winner_score, loser_score = f_score, m_score

        margin = (winner_score - loser_score) / total

        # ── Special override rules ────────────────────────────────────────
        # 1. Strong dict hit contradicts ML — keep dict unless ML very confident
        if (not dict_is_ambiguous and dr and
                dr.gender != "U" and dr.gender != winner):
            if mlr and mlr.confidence >= _ML_OVERRIDE_DICT:
                dominant_source = "ensemble"
            else:
                winner = dr.gender
                margin = dr.confidence
                dominant_source = "dict"

        # 2. Determine dominant source for reporting
        if dominant_source == "none":
            if not dict_is_ambiguous and dr and dr.gender == winner:
                dominant_source = "dict"
            elif rr and rr.gender == winner:
                dominant_source = "rule"
            elif mlr and mlr.gender == winner:
                dominant_source = "ml"
            else:
                dominant_source = "ensemble"

        # 3. Too close to call → U
        if margin < _MARGIN_MINIMUM:
            return _TokenVote(token=token, gender="U",
                              confidence=margin, source="ensemble", detail=detail)

        confidence = round(min(max(margin, 0.0), 1.0), 2)

        return _TokenVote(token=token, gender=winner,
                          confidence=confidence,
                          source=dominant_source, detail=detail)

    def _select_best_token(
        self,
        votes: list[_TokenVote],
        honorific_signal: Optional[str],
    ) -> GenderResult:
        """
        Among all token votes, select the one to return as the final answer.

        Priority order:
        1. Any female vote — female signal anywhere in name dominates
           (Pakistani male names never include feminine words).
        2. Among remaining: highest position-weighted confidence.
        3. Honorific signal if all tokens returned U.
        4. U if nothing resolved.
        """
        # First pass: if ANY token came back F, that wins
        # (implements the 'any female indicator = 100% female' rule)
        female_votes = [v for v in votes if v.gender == "F"]
        if female_votes:
            # Pick the highest-confidence female vote
            best_f = max(female_votes, key=lambda v: v.confidence)
            return GenderResult(
                gender="F",
                confidence=best_f.confidence,
                source=best_f.source,
                matched_token=best_f.token,
            )

        # Second pass: normal position-weighted selection for non-female
        best_score  = -1.0
        best_vote   = None

        for i, vote in enumerate(votes):
            if vote.gender == "U":
                continue
            pos_weight = _POS_WEIGHTS[min(i, len(_POS_WEIGHTS) - 1)]
            score = vote.confidence * pos_weight
            if score > best_score:
                best_score = score
                best_vote  = vote

        if best_vote:
            return GenderResult(
                gender=best_vote.gender,
                confidence=best_vote.confidence,
                source=best_vote.source,
                matched_token=best_vote.token,
            )

        # All tokens returned U — fall back to honorific signal
        if honorific_signal in ("M", "F"):
            return GenderResult(
                gender=honorific_signal,
                confidence=0.60,
                source="rule",
                matched_token="[honorific]",
            )

        # Truly unknown
        first_token = votes[0].token if votes else ""
        return GenderResult(
            gender="U", confidence=0.0,
            source="none", matched_token=first_token,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_name_like(token: str) -> bool:
        """
        Return True if token looks like a real name.
        Rejects: tokens with digits/symbols, pure consonant clusters (xyz, zzz).
        Real names always contain at least one vowel.
        """
        if not _NAME_PATT.match(token):
            return False
        return any(c in _VOWELS for c in token)
