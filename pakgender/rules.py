"""
rules.py
--------
Layer 2: Rule-based gender inference using suffix patterns, prefix signals,
and phoneme rules specific to Pakistani/Urdu/Arabic names.

Rules fire in priority order. First match wins.

Confidence levels
-----------------
  0.98  definitive — cultural/linguistic marker, near-zero exceptions
  0.90  strong suffix (nearly always one gender)
  0.82  moderate suffix (usually one gender, occasional exceptions)
  0.72  weak suffix / phoneme rule (tendency, not certainty)
"""

import re
from typing import Optional
from ._version import GenderResult


# ---------------------------------------------------------------------------
# Definitive female compound suffixes / words — any part of name containing
# these means the WHOLE name is female. No exceptions in Pakistani naming.
# ---------------------------------------------------------------------------
DEFINITIVE_FEMALE_WORDS: set[str] = {
    # Honorific / relational suffixes
    "bibi", "begum", "bano", "banu", "khatoon", "sultana",
    # Religious female names used as second-word markers
    "fatima", "fatimah", "khadija", "maryam", "zainab", "batool",
    # ALLAH-name female second words (user-specified: bachai, rakhi, wasai, mafi)
    "bachai",   # ALLAH BACHAI — female
    "rakhi",    # ALLAH RAKHI — female
    "wasai",    # ALLAH WASAI — female
    "mafi",     # ALLAH MAFI — female (forgiveness/pardon — female name)
    # User-specified definitive female words
    "mai",      # KAREEM MAI — 'mai' is a female honorific/name
    "mia",      # MIA — female name
    "dua",      # DUA — female name (prayer)
    "huda",     # NOOR UL HUDA, MAHNOOR — 'guidance', exclusively female name
    "noor",     # KARAM NOOR — noor by itself in a compound = female
    # Common female name suffixes used standalone as second word
    "nisa", "nisaa",      # 'women' — Noor un Nisa, Quamer un Nisa
    "ain",                # 'eye' — Noor ul Ain, Qurat ul Ain
    "kanwal",             # lotus — almost exclusively female in Pakistan
    "parveen", "parvinn", # star — female
    "rubab",              # musical instrument name — female
    "kalsoom", "kulsoom", # female name
    "naz",                # female suffix (Gul Naz, Falak Naz)
    "zahra", "zehra",     # female (Flower)
    "zadi",               # NAWAB ZADI — princess/daughter suffix
    "tabassum",           # exclusively female name in Pakistan
    "tabasum",            # variant spelling
    "jannat",             # heaven — exclusively female name in Pakistan
    "muqaddas",           # sacred — female name in Pakistan
    "resham",             # silk — female name
    "reshman",            # variant — female
    "raisa",              # female name
    "ruby",               # female name
    "kashaf",             # female name (revelation)
    "aiman", "ayman",     # female name (blessed/lucky)
    "gulsoom",            # variant of kalsoom — female
    "sahar",              # dawn — female name in Pakistan
    "mahnoor",            # moonlight — female name
    "nayyab", "nayab",    # rare/precious — predominantly female
}

# Female prefix words — when a name STARTS with these, it is female
DEFINITIVE_FEMALE_PREFIXES: set[str] = {
    "bibi",    # Bibi Gul
    "ume",     # Ume Kalsoom — 'mother of'
    "umme",    # variant
    "omme",    # variant
    "baby",    # Baby Saba, Baby Noor — virtually always female prefix
    "angel",   # Angel John, Twinkle Angel
    "twinkle", # Twinkle Angel
    "dua",     # Dua Malik — female prefix name
}

# Male prefix words — when a name STARTS with these, it is confirmed male
DEFINITIVE_MALE_PREFIXES: set[str] = {
    "kaka",    # KAKA — male honorific/prefix in Pakistani naming
}

# Words that are ambiguous AS FIRST TOKEN but become definitive
# female signals as SECOND (or later) tokens
FEMALE_AS_SECOND_TOKEN: set[str] = {
    "bibi", "begum", "bano", "banu", "khatoon", "sultana", "batool",
    "fatima", "fatimah", "khadija", "maryam", "zainab",
    "nisa", "ain", "kanwal", "parveen", "rubab",
    "kalsoom", "kulsoom", "naz", "zahra", "zehra",
    "bachai", "rakhi", "wasai", "mafi", "mai", "mia", "dua", "huda",
    "zadi", "tabassum", "tabasum", "jannat", "muqaddas",
    "resham", "reshman", "raisa", "ruby", "kashaf",
    "aiman", "ayman", "gulsoom", "sahar", "mahnoor",
    "nayyab", "nayab", "noor",
}

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

# Each entry: (pattern, gender, confidence, description)
# pattern is a regex applied to the END of the token (suffix match)
# Applied in order — first match wins for each token.

SUFFIX_RULES: list[tuple[re.Pattern, str, float, str]] = [
    # -----------------------------------------------------------------------
    # CRITICAL: longest/most-specific patterns FIRST
    # A suffix like -ullah must appear before -ah or -ul to win first-match.
    # -----------------------------------------------------------------------

    # Strong MALE — compound suffixes (must precede their sub-patterns)
    (re.compile(r"ullah$"),     "M", 0.98, "suffix -ullah (e.g. Saifullah)"),
    (re.compile(r"allah$"),     "M", 0.98, "suffix -allah (e.g. Bismillah)"),
    (re.compile(r"uddin$"),     "M", 0.98, "suffix -uddin (e.g. Salahuddin)"),
    (re.compile(r"din$"),       "M", 0.98, "suffix -din (e.g. Nooruddin)"),
    (re.compile(r"eem$"),       "M", 0.80, "suffix -eem (e.g. Kareem, Raheem)"),
    (re.compile(r"aan$"),       "M", 0.75, "suffix -aan (e.g. Ramzan)"),
    (re.compile(r"an$"),        "M", 0.70, "suffix -an (e.g. Imran, Adnan — weak)"),
    (re.compile(r"oor$"),       "F", 0.60, "suffix -oor (Noor is predominantly female)"),
    (re.compile(r"ur$"),        "M", 0.85, "suffix -ur (e.g. Zia-ur)"),
    (re.compile(r"ud$"),        "M", 0.90, "suffix -ud (e.g. Mahmud, Masud)"),
    # NOTE: -ul removed as standalone suffix — too greedy (fires on 'noor ul')
    # It is handled as a COMPOUND rule instead.
    (re.compile(r"ab$"),        "M", 0.72, "suffix -ab (e.g. Nawab, Shuaib)"),
    (re.compile(r"ub$"),        "M", 0.82, "suffix -ub (e.g. Ayyub)"),

    # Strong FEMALE — compound suffixes first
    (re.compile(r"afzaa$"),     "F", 0.98, "suffix -afzaa"),
    (re.compile(r"afza$"),      "F", 0.98, "suffix -afza (e.g. Gulafza)"),
    (re.compile(r"bano$"),      "F", 0.98, "suffix -bano (e.g. Shabano)"),
    (re.compile(r"banu$"),      "F", 0.98, "suffix -banu"),
    (re.compile(r"bibi$"),      "F", 0.98, "suffix -bibi"),
    (re.compile(r"begum$"),     "F", 0.98, "suffix -begum"),
    (re.compile(r"naz$"),       "F", 0.98, "suffix -naz (e.g. Dilnaz, Shehnaz, Gul Naz)"),
    (re.compile(r"ara$"),       "F", 0.88, "suffix -ara (e.g. Gulara, Dilara)"),
    (re.compile(r"een$"),       "F", 0.82, "suffix -een (e.g. Shaheen, Parveen)"),
    (re.compile(r"ine$"),       "F", 0.98, "suffix -ine"),
    (re.compile(r"oon$"),       "F", 0.68, "suffix -oon"),
    (re.compile(r"iya$"),       "F", 0.99, "suffix -iya (e.g. Ruqaiya)"),
    (re.compile(r"ia$"),        "F", 0.98, "suffix -ia (e.g. Sofia, Nadia)"),
    (re.compile(r"abila$"),     "F", 0.98, "suffix -abila (e.g. Nabila, Sabila)"),
    (re.compile(r"yeda$"),      "F", 0.98, "suffix -yeda (e.g. Syeda)"),
    (re.compile(r"rakhi$"),     "F", 0.98, "suffix -rakhi (e.g. Allah Rakhi)"),
    (re.compile(r"wasai$"),     "F", 0.98, "suffix -wasai (e.g. Allah Wasai)"),
    (re.compile(r"abrin$"),     "F", 0.98, "suffix -abrin (e.g. Samrin)"),
    (re.compile(r"abreen$"),    "F", 0.98, "suffix -abreen (e.g. Sambreen)"),
    (re.compile(r"eeba$"),      "F", 0.98, "suffix -eeba (e.g. Aneeba)"),
    (re.compile(r"eeqa$"),      "F", 0.98, "suffix -eeqa (e.g. Aneeqa)"),
    (re.compile(r"iqa$"),       "F", 0.98, "suffix -iqa (e.g. Aniqa)"),
    (re.compile(r"inah$"),      "F", 0.98, "suffix -inah (e.g. Aminah)"),
    (re.compile(r"ilya$"),      "F", 0.98, "suffix -ilya (e.g. Aliya)"),
    (re.compile(r"asra$"),      "F", 0.98, "suffix -asra (e.g. Nasra)"),
    (re.compile(r"rah$"),       "F", 0.98, "suffix -rah (e.g. Bushrah)"),
    (re.compile(r"ni$"),        "F", 0.95, "suffix -ni (e.g. Roshni)"),
    (re.compile(r"zadi$"),      "F", 0.98, "suffix -zadi (e.g. Nawab Zadi)"),
    (re.compile(r"tana$"),      "F", 0.92, "suffix -tana (e.g. Sultana)"),
    (re.compile(r"khatoon$"),   "F", 0.98, "suffix -khatoon"),
    (re.compile(r"kalsoom$"),   "F", 0.98, "suffix -kalsoom / kulsoom"),
    (re.compile(r"kulsoom$"),   "F", 0.98, "suffix -kulsoom"),
    # -ah is LAST among female rules — very general
    (re.compile(r"ah$"),        "F", 0.72, "suffix -ah (e.g. Farah, Leylah)"),
]

# Compound masculine patterns — applied to the FULL token, not just suffix
COMPOUND_RULES: list[tuple[re.Pattern, str, float, str]] = [
    # Male Islamic compound patterns
    (re.compile(r"\bur\s+rahman\b"),   "M", 0.98, "compound -ur Rahman"),
    (re.compile(r"\bul\s+haq\b"),      "M", 0.98, "compound -ul Haq"),
    (re.compile(r"\bul\s+islam\b"),    "M", 0.98, "compound -ul Islam"),
    (re.compile(r"\bul\s+din\b"),      "M", 0.98, "compound -ul Din"),
    # Female Islamic compound patterns (ADDED — previously missing)
    (re.compile(r"\bul\s+ain\b"),      "F", 0.98, "compound -ul Ain (Noor ul Ain)"),
    (re.compile(r"\bun\s+nisa\b"),     "F", 0.98, "compound -un Nisa (women)"),
    (re.compile(r"\bun\s+nissa\b"),    "F", 0.98, "compound -un Nissa variant"),
    (re.compile(r"\bul\s+huda\b"),     "F", 0.98, "compound -ul Huda (guidance — female name)"),
    (re.compile(r"\bul\s+nisa\b"),     "F", 0.98, "compound -ul Nisa"),
    # Standalone male names
    (re.compile(r"\bali$"),            "M", 0.98, "standalone Ali"),
    (re.compile(r"\busman$"),          "M", 0.90, "standalone Usman"),
    (re.compile(r"\bomar$"),           "M", 0.90, "Omar"),
    (re.compile(r"\bumar$"),           "M", 0.90, "Umar"),
    (re.compile(r"\beeb$"),            "M", 0.90, "nuneeb"),
]

# Phoneme / vowel-pattern rules — last resort within Layer 2
PHONEME_RULES: list[tuple[re.Pattern, str, float, str]] = [
    # Names ending in open 'a' vowel (not -ara/-ia covered above) tend female in Urdu
    (re.compile(r"[^aeiou][aeiou]a$"),  "F", 0.65, "terminal open -a vowel pattern"),
]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class RuleEngine:
    """
    Stateless rule engine. No data files needed.
    Instantiate once; call lookup() per name.
    """

    def check_definitive_female(
        self,
        tokens: list[str],
    ) -> Optional["GenderResult"]:
        """
        Check whether ANY token in the full name is a definitive gender signal.

        KEY INSIGHT:
        In Pakistani naming convention, males NEVER include feminine words
        in their names. So if ANY part of a multi-word name is a definitive
        female word, the entire name is female with certainty.
        Likewise, certain prefix honorifics confirm male (e.g. KAKA).

        This should be called BEFORE per-token lookup.

        Parameters
        ----------
        tokens : list[str]
            All name tokens (after honorific stripping, lowercased).

        Returns
        -------
        GenderResult with confidence=1.0 if a definitive signal found,
        else None.
        """
        if not tokens:
            return None

        # ── Definitive MALE prefix check ─────────────────────────────────
        if tokens[0] in DEFINITIVE_MALE_PREFIXES:
            return GenderResult(
                gender="M",
                confidence=1.0,
                source="rule",
                matched_token=tokens[0],
            )

        # ── Definitive FEMALE prefix check ───────────────────────────────
        if tokens[0] in DEFINITIVE_FEMALE_PREFIXES:
            return GenderResult(
                gender="F",
                confidence=1.0,
                source="rule",
                matched_token=tokens[0],
            )

        # ── Check ALL tokens for definitive female words ──────────────────
        # (ANY token anywhere = whole name is female, no exceptions)
        for token in tokens:
            if token in DEFINITIVE_FEMALE_WORDS:
                return GenderResult(
                    gender="F",
                    confidence=1.0,
                    source="rule",
                    matched_token=token,
                )

        # ── Check second+ tokens for "female as second token" words ───────
        if len(tokens) > 1:
            for token in tokens[1:]:
                if token in FEMALE_AS_SECOND_TOKEN:
                    return GenderResult(
                        gender="F",
                        confidence=0.95,
                        source="rule",
                        matched_token=token,
                    )

        return None

    def lookup(
        self,
        candidates: list[str],
        honorific_signal: Optional[str] = None,
    ) -> Optional["GenderResult"]:
        """
        Try each candidate token against all rule sets.

        Parameters
        ----------
        candidates : list[str]
            Preprocessed candidate tokens (personal name first).
        honorific_signal : str or None
            'M', 'F', or None from the preprocessor honorific detection.
            If set and no suffix rule fires, return this as a low-confidence result.

        Returns
        -------
        GenderResult or None
            None means no rule matched; caller moves to Layer 3.
        """
        # Try suffix + compound rules on each candidate
        for token in candidates:
            result = self._apply_suffix_rules(token)
            if result:
                return result
            result = self._apply_compound_rules(token)
            if result:
                return result

        # Phoneme rules (weaker — only on first candidate)
        if candidates:
            result = self._apply_phoneme_rules(candidates[0])
            if result:
                return result

        # Fall back to honorific signal alone (lowest confidence)
        if honorific_signal in ("M", "F"):
            return GenderResult(
                gender=honorific_signal,
                confidence=0.65,
                source="rule",
                matched_token="[honorific]",
            )

        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_suffix_rules(token: str) -> Optional["GenderResult"]:
        for pattern, gender, confidence, _ in SUFFIX_RULES:
            if pattern.search(token):
                return GenderResult(
                    gender=gender,
                    confidence=confidence,
                    source="rule",
                    matched_token=token,
                )
        return None

    @staticmethod
    def _apply_compound_rules(token: str) -> Optional["GenderResult"]:
        for pattern, gender, confidence, _ in COMPOUND_RULES:
            if pattern.search(token):
                return GenderResult(
                    gender=gender,
                    confidence=confidence,
                    source="rule",
                    matched_token=token,
                )
        return None

    @staticmethod
    def _apply_phoneme_rules(token: str) -> Optional["GenderResult"]:
        for pattern, gender, confidence, _ in PHONEME_RULES:
            if pattern.search(token):
                return GenderResult(
                    gender=gender,
                    confidence=confidence,
                    source="rule",
                    matched_token=token,
                )
        return None

    # ------------------------------------------------------------------
    # Utility — for debugging / rule development
    # ------------------------------------------------------------------

    def explain(self, token: str) -> list[dict]:
        """Return all rules that match a token, for debugging."""
        matches = []
        for pattern, gender, confidence, description in SUFFIX_RULES:
            if pattern.search(token):
                matches.append({
                    "layer": "suffix",
                    "gender": gender,
                    "confidence": confidence,
                    "rule": description,
                    "pattern": pattern.pattern,
                })
        for pattern, gender, confidence, description in COMPOUND_RULES:
            if pattern.search(token):
                matches.append({
                    "layer": "compound",
                    "gender": gender,
                    "confidence": confidence,
                    "rule": description,
                    "pattern": pattern.pattern,
                })
        return matches
