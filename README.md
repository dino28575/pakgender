# pakgender

**Gender inference for Pakistani and Arabic names — built for South Asian data pipelines.**

[![PyPI version](https://img.shields.io/pypi/v/pakgender.svg)](https://pypi.org/project/pakgender/)
[![Python](https://img.shields.io/pypi/pyversions/pakgender.svg)](https://pypi.org/project/pakgender/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Most gender inference libraries (like `genderize.io`) perform poorly on Pakistani names or require expensive API subscriptions. `pakgender` is a free, offline alternative—standard tools often misclassify names like `Shehnaz`, `Maryam`, and `Saifullah` because they are trained almost exclusively on Western data.

`pakgender` was built specifically to bridge this gap. It uses a three-layer pipeline — dictionary lookup, rule-based suffix/prefix analysis, and a character n-gram ML model — all tuned for Urdu, Arabic, and Persian-origin names written in Roman script.

---

## Features

- **5,100+ name dictionary** covering Pakistani names records and Arabic names
- **Spelling variant normalisation** — `Aisha`, `Ayesha`, `Aesha` all resolve to the same entry
- **Honorific stripping** — handles `Muhammad`, `Mst.`, `Ch.`, `Syed`, `Begum`, and 30+ other prefixes
- **Rule engine** — 25+ suffix patterns (`-ullah`, `-bano`, `-een`, `-naz`, `-uddin`) with confidence scores
- **ML fallback** — character n-gram Logistic Regression (82% accuracy, ±0.6% CV variance) for names outside the dictionary
- **Batch processing** — native `pandas` Series support for large datasets
- **Fully offline** — no API calls, no internet required; model ships with the package (~185 KB)

---

## Installation

```bash
pip install pakgender
```

No extra dependencies beyond `scikit-learn`.

---

## Quick start

```python
from pakgender import predict

result = predict("Fatima Noor")
print(result)
# GenderResult(gender='F', confidence=0.95, source='dict', matched_token='fatima')

print(result.gender)      # 'F'
print(result.confidence)  # 0.95
print(result.source)      # 'dict'  — which layer answered: dict / rule / ml
print(result.is_known())  # True
```

### Handles real-world messy names

```python
from pakgender import predict

predict("MUHAMMAD AHSAN RAZA")   # strips Muhammad prefix → GenderResult(gender='M', ...)
predict("Mst. Zara Bibi")        # Mst. signals female → GenderResult(gender='F', ...)
predict("Ch. Imran Khan")        # strips Ch. and Khan → GenderResult(gender='M', ...)
predict("Aisha")                 # normalises Aisha → Ayesha → dict lookup
predict("Saifullah")             # rule: -ullah suffix → GenderResult(gender='M', confidence=0.95, source='rule', ...)
predict("xyz123")                # GenderResult(gender='U', confidence=0.0, source='none', ...)
```

### Batch processing with pandas

```python
import pandas as pd
from pakgender import predict_series

df = pd.read_excel("cbs_data.xlsx")

result = predict_series(df["Account_Title"])
df["gender"]      = result["gender"]       # 'M', 'F', or 'U'
df["confidence"]  = result["confidence"]   # 0.0 – 1.0
df["gender_src"]  = result["source"]       # 'dict', 'rule', or 'ml'
```

### Understanding the output

| Field | Values | Meaning |
|---|---|---|
| `gender` | `'M'`, `'F'`, `'U'` | Male, Female, Unknown/ambiguous |
| `confidence` | `0.0` – `1.0` | How certain the prediction is |
| `source` | `'dict'`, `'rule'`, `'ml'` | Which layer answered |
| `matched_token` | e.g. `'fatima'` | The specific name token that triggered the result |

A confidence of `0.0` with `source='none'` means all three layers failed — treat this as `U`.

---

## How it works

```
Input: "Mst. Fatima Noor"
         │
         ▼
   ┌─────────────┐
   │ Preprocessor│  strips Mst. (→ F signal), normalises spelling,
   └──────┬──────┘  tokenises → candidates: ['fatima', 'noor']
          │
          ▼
   ┌─────────────┐
   │  Layer 1    │  looks up 'fatima' in 5,100+ name dictionary
   │  Dictionary │  → hit: gender=F, confidence=1.0
   └──────┬──────┘
          │  (if miss or ambiguous)
          ▼
   ┌─────────────┐
   │  Layer 2    │  checks suffix patterns: -bano, -naz, -een, -ullah,
   │  Rules      │  -uddin, -ara, -ul, and 20+ more
   └──────┬──────┘
          │  (if still ambiguous)
          ▼
   ┌─────────────┐
   │  Layer 3    │  character n-gram (2–4) Logistic Regression
   │  ML model   │  trained on 5,100+ Pakistani + Arabic names
   └─────────────┘
          │
          ▼
   GenderResult(gender='F', confidence=0.95, source='dict', matched_token='fatima')
```

---

## Supported name formats

| Format | Example | Handled |
|---|---|---|
| First name only | `Fatima` | ✓ |
| Full name | `Muhammad Ahsan Raza` | ✓ |
| With honorific prefix | `Mst. Zara`, `Ch. Imran`, `Syed Ali` | ✓ |
| With female honorific | `Begum Nusrat`, `Bibi Zulaikha` | ✓ |
| Spelling variants | `Aisha` / `Ayesha` / `Aesha` / `Aysha` | ✓ |
| Uppercase | `FATIMA MALIK` | ✓ |
| Abbreviated prefix | `M. Usman`, `Md. Tariq` | ✓ |
| Compound Islamic names | `Saifullah`, `Salahuddin`, `Nooruddin` | ✓ |

---

## Accuracy

Evaluated on a held-out 20% test split from the training dictionary:

| Metric | Value |
|---|---|
| Test accuracy | 82.0% |
| 5-fold CV accuracy | 81.4% ± 0.6% |
| Female precision / recall | 0.84 / 0.83 |
| Male precision / recall | 0.80 / 0.81 |

The low CV variance (±0.6%) means the model generalises consistently — it is not sensitive to which names end up in the test split.

The dictionary layer (Layer 1) handles the majority of common Pakistani names with confidence ≥ 0.95. The 82% figure reflects the ML fallback layer only, which activates for names outside the dictionary.

---

## Expanding the dictionary

The dictionary is a plain JSON file bundled with the package. You can add your own names without retraining:

```python
# add_names.py — run once, then reinstall or point to custom path
import json
from pathlib import Path
import importlib.resources

ref = importlib.resources.files("pakgender.data").joinpath("names.json")
with importlib.resources.as_file(ref) as path:
    with open(path) as f:
        db = json.load(f)

# Add new entries
db["first_names"]["bakhtawar"] = {"gender": "F", "frequency": "medium"}
db["first_names"]["zulaikha"]  = {"gender": "F", "frequency": "low"}

with open(path, "w", encoding="utf-8") as f:
    json.dump(db, f, ensure_ascii=False, indent=2)
```

To retrain the ML model after a large dictionary update:

```bash
python train_model.py
```

---

## Honorifics and prefixes recognised

The preprocessor strips these automatically before lookup:

**Neutral** (stripped silently): `Muhammad`, `Mohammed`, `Md`, `M.`, `Syed`, `Sayyid`, `Sheikh`, `Hafiz`, `Haji`, `Ch`, `Chaudhry`, `Raja`, `Rana`, `Malik`, `Khan`, `Mirza`, `Baig`, `Mian`, `Dr`, `Mr`

**Female signal** (stripped + gender hint set to F): `Begum`, `Bibi`, `Bano`, `Mst`, `Mst.`

**Male signal** (stripped + gender hint set to M): `Muhammad`, `Hafiz`, `Haji`, `Maulana`

---

## Spelling variants normalised

A sample of the variant map built into the preprocessor:

| Input variant | Normalised to |
|---|---|
| Aisha / Aesha / Aysha | Ayesha |
| Fatimah / Fatema | Fatima |
| Khadijah / Khadeeja | Khadija |
| Mariam / Marium / Maryum | Maryam |
| Hussain / Hussein | Husain |
| Hassan | Hasan |
| Nouman / Numaan | Noman |
| Nur / Nour | Noor |
| Zehra | Zahra |
| Ehsan / Ihsan | Ahsan |

---

## Limitations

- **Roman script only** — Urdu/Arabic script (`فاطمہ`) is not supported. Transliterate first if needed.
- **Ambiguous names** — names like `Noor`, `Akhtar`, `Gul` are used for both genders in Pakistan. These return `gender='U'` with a note in `source`. Add your own override rules using the dictionary.
- **Gulf vs South Asian conventions** — some names differ in gender between Pakistani and Gulf Arab usage (e.g. `Aman`, `Hani`). The library is tuned for Pakistani convention.
- **Post-processing recommended** — for high-stakes applications, filter on `confidence >= 0.75` and manually review rows where `source='ml'` and `confidence < 0.80`.

---

## Contributing

Contributions welcome — especially:
- Additional Pakistani name entries with correct gender labels
- Urdu/Persian-origin names missing from the dictionary
- Corrections to mislabeled entries

Please open an issue or pull request on [GitHub](https://github.com/dino28575/pakgender).

---

## License

MIT License. See `LICENSE` for details.

---

## Author

**Sahib Dino** — Data Analyst, Monitoring & Internal Control  
[dino28575.github.io](https://dino28575.github.io) · [GitHub](https://github.com/dino28575)

Built to solve a real problem: automated gender verification in Pakistani names records for data quality checks and error resolution.
