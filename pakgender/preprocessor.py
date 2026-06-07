"""
preprocessor.py
---------------
Turns a raw name string into an ordered list of candidate tokens for lookup.

Pipeline:
  raw string
    -> lowercase + strip
    -> normalize Roman Urdu spelling variants  (aisha -> ayesha)
    -> strip honorific prefixes                (Muhammad, Syed, Begum …)
    -> tokenize into parts
    -> rank tokens: personal name first, compound last

Changes from v1:
  - ALLAH is now treated as a neutral prefix (not male honorific),
    because ALLAH RAKHI / ALLAH WASAI / ALLAH BACHAI are female names.
    The token after ALLAH carries the true gender signal.
  - UME / UMME / OMME added as strong female prefixes.
  - BABY, ANGEL, TWINKLE added as female prefix signals.
  - Expanded VARIANT_MAP with more common Roman Urdu alternates.
  - Full compound string candidate always added last for compound-dict hits
    (e.g. 'naseem akhtar' as a single dict key).
"""

import re
import unicodedata
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Honorific / relational prefixes — stripped before tokenisation
# ---------------------------------------------------------------------------
HONORIFICS: set[str] = {
    # Islamic / religious — NOTE: Muhammad IS stripped (too common as prefix)
    "muhammad", "mohammed", "mohammad", "mehmed",
    "m", "md",
    "hafiz", "hafez",
    "haji", "haj",
    "syed", "sayyid", "syeda",          # syeda gives female signal first
    "maulana", "maulvi",
    "sheikh", "shaikh",
    # Tribal / caste
    "ch", "chaudhry", "chaudhary", "choudhry",
    "raja", "rana",
    "malik",
    "khan",
    "mirza",
    "baig", "beg",
    "mian",
    # Female honorifics / relational — handled separately below
    "miss", "mrs", "ms", "dr", "mr",
    "mst", "mst.",
    # NOTE: 'allah' is intentionally NOT in HONORIFICS here —
    # it is handled specially: the engine strips it as a prefix token
    # but continues evaluating the remaining tokens (which carry gender signal).
}

# Honorifics that are themselves strong female gender signals
FEMALE_HONORIFICS: set[str] = {
    "begum", "bibi", "bano", "mst", "mst.",
    "syeda", "hafiza",
    "sultana", "batool", "khatoon",
    # Female prefix names (not traditional honorifics but act as one)
    "baby", "angel", "twinkle",
    "ume", "umme", "omme",              # 'mother of' — always female
}

# Honorifics that are strong male gender signals
MALE_HONORIFICS: set[str] = {
    "muhammad", "mohammed", "mohammad", "mehmed", "md", "hafiz",
    "haji", "haj", "maulana", "maulvi",
}

# Words used as name-prefixes that should be STRIPPED but whose following
# tokens carry the gender signal (ALLAH-names: the second token decides)
NEUTRAL_PREFIXES: set[str] = {
    "allah",   # ALLAH RAKHI=F, ALLAH DIN=M — evaluate remainder
    "ghulam",  # GHULAM FATIMA=F edge case — rare but possible
    # NOTE: 'noor' intentionally removed — NOOR is a female name in its own right
    # (KARAM NOOR = F). NOOR UL AIN / NOOR UL HUDA compounds are caught by
    # compound rules applied to the full_compound string.
}


# ---------------------------------------------------------------------------
# Roman Urdu spelling variant map
# Key = non-canonical form(s) -> value = canonical form used in names.json
# ---------------------------------------------------------------------------
VARIANT_MAP: dict[str, str] = {
    # A-sounds
    "aisha":    "ayesha",
    "anfaal":   "anfal",
    "aesha":    "ayesha",
    "aysha":    "ayesha",
    "aayesha":  "ayesha",
    "aaysha":   "ayesha",
    "amena":    "amina",
    "ameena":   "amina",
    "aamina":   "amina",
    "aaminah":  "amina",
    "fatimah":  "fatima",
    "fatemah":  "fatima",
    "fatema":   "fatima",
    "khadijah": "khadija",
    "khadeeja": "khadija",
    "zeinab":   "zainab",
    "zaynab":   "zainab",
    "mariam":   "maryam",
    "marium":   "maryam",
    "maryum":   "maryam",
    "mahanoor": "mahnoor",
    "saana":    "sana",
    # U-sounds
    "osman":    "usman",
    "osama":    "usama",
    # H-sounds
    "hassan":   "hasan",
    "husain":   "husain",
    "hussain":  "husain",
    "hussein":  "husain",
    # I/Y alternation
    "umran":    "imran",
    "erfan":    "irfan",
    # Double-letter simplification
    "ehsan":    "ahsan",
    "ihsan":    "ahsan",
    "nouman":   "noman",
    "numaan":   "noman",
    # Z/J alternation
    "zehra":    "zahra",
    # Noor/Nur
    "nur":      "noor",
    "nour":     "noor",
    # Khatoon variants
    "khatun":   "khatoon",
    "khatoun":  "khatoon",
    # Kalsoom variants
    "kulsoom":  "kalsoom",
    "kulsum":   "kalsoom",
    "kulsoum":  "kalsoom",
    "gulsoom":  "kalsoom",   # GULSOOM is a kalsoom variant
    "gulsoum":  "kalsoom",
    # Ume/Umme/Omme (mother of) variants
    "umme":     "ume",
    "omme":     "ume",
    "um":       "ume",
    # Parveen variants
    "parvinn":  "parveen",
    "perveen":  "parveen",
    # Akhter variants
    "akhtar":   "akhter",
    "akhtar":   "akhter",
    # Naseem variants
    "nasim":    "naseem",
    # Tabassum variant
    "tabasum":  "tabassum",
    # Shakeel variant
    "shakil":   "shakeel",
}


class PreprocessedName(NamedTuple):
    """Result of preprocessing a raw name string."""
    tokens: list[str]             # cleaned, lowercased tokens (honorifics stripped)
    candidates: list[str]         # ordered list to try in dictionary/rules (best first)
    honorific_signal: str | None  # 'M', 'F', or None — from prefix alone
    original: str                 # raw input, preserved
    full_compound: str            # full lowercased token string (for compound dict lookup)


def normalize_spelling(token: str) -> str:
    """Map spelling variant to canonical form. Identity if unknown."""
    return VARIANT_MAP.get(token, token)


def _clean(raw: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation except hyphens."""
    s = raw.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s\-]", "", s)   # keep word chars, spaces, hyphens
    s = re.sub(r"\s+", " ", s)
    return s


def preprocess(raw: str) -> "PreprocessedName":
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    raw : str
        Raw name string from the database, e.g. "Mst. Fatima Noor".

    Returns
    -------
    PreprocessedName
        tokens     : cleaned parts after stripping honorifics
        candidates : ordered search list — personal name first
        honorific_signal : gender hint from honorific alone ('M', 'F', None)
        original   : unchanged input
        full_compound : full joined token string for compound dict lookup
    """
    cleaned = _clean(raw)
    parts = cleaned.split()

    honorific_signal: str | None = None
    tokens: list[str] = []

    for part in parts:
        part_norm = normalize_spelling(part)

        # --- Female honorifics ---
        if part_norm in FEMALE_HONORIFICS or part in FEMALE_HONORIFICS:
            honorific_signal = "F"
            # Keep female honorifics as tokens too — they ARE the gender signal
            # (e.g. "ume" in "Ume Kalsoom" should be in candidates)
            tokens.append(part_norm if part_norm != part else part_norm)
            continue

        # --- Male honorifics ---
        if part_norm in MALE_HONORIFICS or part in MALE_HONORIFICS:
            if honorific_signal is None:
                honorific_signal = "M"
            continue   # strip prefix, don't add to tokens

        # --- Neutral honorifics ---
        if part_norm in HONORIFICS or part in HONORIFICS:
            continue   # neutral — strip silently

        # --- Neutral prefixes (like ALLAH) — strip but keep rest ---
        if part_norm in NEUTRAL_PREFIXES or part in NEUTRAL_PREFIXES:
            # Do NOT add to tokens; the following tokens carry the signal.
            # This prevents ALLAH from triggering the -allah suffix rule.
            continue

        tokens.append(normalize_spelling(part_norm))

    # Build candidate search order:
    #   Pakistani naming: FIRST_NAME FATHER_NAME [FAMILY]
    #   The personal/given name is almost always token index 0 after stripping.
    #   But some records are stored as FATHER GIVEN — so we try both orderings.
    candidates: list[str] = []
    if tokens:
        candidates.append(tokens[0])          # most likely personal name
        if len(tokens) > 1:
            candidates.append(tokens[1])      # father's name / second part
        if len(tokens) > 2:
            candidates.extend(tokens[2:])     # remaining parts
        # Add the full normalized string as last resort (catches compound dict entries)
        full = " ".join(tokens)
        if full not in candidates:
            candidates.append(full)

    full_compound = " ".join(tokens)

    return PreprocessedName(
        tokens=tokens,
        candidates=candidates,
        honorific_signal=honorific_signal,
        original=raw,
        full_compound=full_compound,
    )
