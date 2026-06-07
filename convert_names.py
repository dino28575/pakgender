"""
convert_names.py
----------------
Converts your Excel name-gender dataset into pakgender/data/names.json.

Usage:
    python convert_names.py Guessed_Genders.xlsx

The script:
  1. Reads the Excel file (expects columns: Full_Name/First_Name, Guessed_Gender)
  2. Strips honorific prefixes / connector tokens
  3. Normalizes Roman Urdu spelling variants
  4. Detects and logs conflicts (same name, different gender in two rows)
  5. Applies MANUAL_OVERRIDES to fix known labeling errors
  6. Writes pakgender/data/names.json

Re-run this script whenever you expand your dataset.
"""

import sys
import re
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────

INPUT_COLS = {"name": "Full_Name/First_Name", "gender": "Guessed_Gender"}
OUTPUT_PATH = Path(__file__).parent / "pakgender" / "data" / "names.json"

STRIP_TOKENS: set[str] = {
    # Islamic / religious prefixes
    "muhammad", "mohammed", "mohammad", "mehmed", "md", "m",
    "hafiz", "hafez", "haji", "haj",
    "syed", "sayyid", "syeda",
    "sheikh", "shaikh",
    "maulana", "maulvi",
    # Tribal / caste / titles
    "ch", "chaudhry", "chaudhary", "choudhry",
    "raja", "rana", "malik", "khan", "mirza",
    "baig", "beg", "mian",
    "dr", "mr", "miss", "mrs", "ms", "prof",
    # Connectors (not names on their own)
    "bin", "bint", "al", "ul", "ud", "ur", "ibn",
    # Female honorifics (stripped because they appear as SECOND part, not personal name)
    "begum", "bibi", "bano", "khatoon", "sultana",
    "mst",
}

VARIANT_MAP: dict[str, str] = {
    "aisha": "ayesha",   "aesha": "ayesha",   "aysha": "ayesha",
    "aayesha": "ayesha", "aaysha": "ayesha",
    "amena": "amina",    "ameena": "amina",    "aamina": "amina",   "aaminah": "amina",
    "fatimah": "fatima", "fatemah": "fatima",  "fatema": "fatima",
    "khadijah": "khadija", "khadeeja": "khadija",
    "mariam": "maryam",  "marium": "maryam",   "maryum": "maryam",
    "zeinab": "zainab",  "zaynab": "zainab",
    "hussain": "husain", "hussein": "husain",  "husayn": "husain",
    "hassan": "hasan",
    "osman": "usman",
    "osama": "usama",
    "umran": "imran",
    "erfan": "irfan",
    "ehsan": "ahsan",    "ihsan": "ahsan",
    "nouman": "noman",   "numaan": "noman",
    "zehra": "zahra",
    "nur": "noor",       "nour": "noor",
}

# ── Manual overrides ───────────────────────────────────────────────────────
# Use this dict to fix known labeling errors in the source Excel.
# Format: "canonical_name": "M" | "F" | "U"
#
# How to identify what to add here:
#   Run the script once, read the CONFLICT LOG at the bottom of the output.
#   For each conflict, decide: is it genuinely ambiguous (U) or a mislabel?
#   If mislabel → add the correct gender here.

MANUAL_OVERRIDES: dict[str, str] = {
    # khadija = unambiguously female Islamic name; conflict was from
    # "KHADIJAH TUL KUBRA" which was mislabeled M in the source Excel.
    "khadija": "F",

    # kousar = female (Kosar/Kausar is a river in paradise, female name)
    # conflict was "KOUSAR PERVEEN" labeled M — likely a data entry error
    "kousar": "F",

    # kaniz = female (means 'maidservant', a female name)
    "kaniz": "F",

    # kaneez = female variant of kaniz
    "kaneez": "F",

    # kishwar = genuinely used for both genders in Pakistan — keep U
    # "kishwar": "U",   # already U from conflict detection, no change needed

    # batool = female (Batool = pure/virgin, an Islamic female title)
    # conflict from "BATOOL BUTT" mislabeled M
    "batool": "F",

    # ── Systematic mislabels found by suffix analysis ──────────────────
    # All have clear female suffixes (-naz, -bano, -een) — mislabeled M in Excel.
    "shehnaz":   "F",   # unambiguously female
    "shahnaz":   "F",   # variant of shehnaz
    "shahbano":  "F",   # female (-bano suffix)
    "noorbano":  "F",   # female (-bano suffix)
    "qurbano":   "F",   # female (-bano suffix)
    "noureen":   "F",   # female (-een suffix)
    "tahseen":   "F",   # predominantly female in Pakistan

    # Add more overrides here as you find errors in your dataset.
    # Format: "name_in_lowercase": "M" or "F" or "U"
}


# ── Helper functions ───────────────────────────────────────────────────────

def normalize(token: str) -> str:
    """Strip punctuation, lowercase, apply variant map."""
    clean = re.sub(r"[^\w]", "", token.lower())
    return VARIANT_MAP.get(clean, clean)


def extract_personal_tokens(raw_name: str) -> list[str]:
    """
    Return the personal name tokens from a full name string.
    Strips honorifics and connector words; normalizes spelling variants.
    """
    parts = raw_name.strip().split()
    tokens = []
    for p in parts:
        n = normalize(p)
        if n and n not in STRIP_TOKENS:
            tokens.append(n)
    return tokens


# ── Main conversion ────────────────────────────────────────────────────────

def convert(excel_path: str) -> None:
    # Load
    df = pd.read_excel(excel_path)
    df = df.rename(columns={
        INPUT_COLS["name"]: "name",
        INPUT_COLS["gender"]: "gender",
    })
    df = df[["name", "gender"]].dropna()
    df["name"] = df["name"].str.strip()
    df = df[df["gender"].isin(["M", "F"])]
    print(f"Loaded {len(df)} rows from {excel_path}")

    # Collect all personal-token → gender mappings, tracking conflicts
    first_name_genders: dict[str, str] = {}           # canonical → gender
    first_name_conflicts: dict[str, set[str]] = defaultdict(set)
    full_name_map: dict[str, dict] = {}
    token_counts: Counter = Counter()

    for _, row in df.iterrows():
        raw, gender = row["name"], row["gender"]
        tokens = extract_personal_tokens(raw)
        if not tokens:
            continue

        # Count for frequency tagging
        token_counts[tokens[0]] += 1

        # Full-name entry (2+ personal tokens)
        if len(tokens) >= 2:
            full_key = f"{tokens[0]} {tokens[1]}"
            if full_key not in full_name_map:
                full_name_map[full_key] = {"gender": gender, "source_name": raw}

        # First-name entry
        personal = tokens[0]
        if personal in first_name_genders:
            if first_name_genders[personal] != gender:
                first_name_conflicts[personal].add(first_name_genders[personal])
                first_name_conflicts[personal].add(gender)
        else:
            first_name_genders[personal] = gender

    # Mark conflicts as ambiguous
    for name in first_name_conflicts:
        first_name_genders[name] = "U"

    # Apply manual overrides
    for name, gender in MANUAL_OVERRIDES.items():
        if name in first_name_genders:
            old = first_name_genders[name]
            first_name_genders[name] = gender
            if old != gender:
                print(f"  [override] '{name}': {old} → {gender}")

    # Assign frequency
    first_names_out: dict[str, dict] = {}
    for name, gender in first_name_genders.items():
        count = token_counts.get(name, 1)
        freq = "high" if count >= 5 else "medium" if count >= 2 else "low"
        first_names_out[name] = {"gender": gender, "frequency": freq}

    # Write output
    output = {"first_names": first_names_out, "full_names": full_name_map}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Summary
    total_M = sum(1 for v in first_names_out.values() if v["gender"] == "M")
    total_F = sum(1 for v in first_names_out.values() if v["gender"] == "F")
    total_U = sum(1 for v in first_names_out.values() if v["gender"] == "U")

    print(f"\n── names.json written to {OUTPUT_PATH} ──")
    print(f"  First names : {len(first_names_out)}  (M={total_M}, F={total_F}, U={total_U})")
    print(f"  Full names  : {len(full_name_map)}")
    print(f"  Conflicts   : {len(first_name_conflicts)}")

    if first_name_conflicts:
        print("\n── Conflict log (names seen with both M and F) ──")
        print("  Review these in your Excel and either fix the label or add to MANUAL_OVERRIDES.")
        for name in sorted(first_name_conflicts):
            override = MANUAL_OVERRIDES.get(name, "→ still U")
            print(f"  {name:<20} {override}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Guessed_Genders.xlsx"
    convert(path)
