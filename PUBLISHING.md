# Publishing pakgender to PyPI

This guide covers everything needed to go from local code to
`pip install pakgender` working for anyone in the world.

---

## Overview — what actually happens

When someone runs `pip install pakgender`, pip connects to
**PyPI** (Python Package Index — pypi.org), downloads a `.whl`
(wheel) file that you uploaded, and installs it.

Your job is to: build that `.whl`, create a PyPI account, and upload it.
After the first upload, every future update is just rebuild → upload.

```
Your code  →  build  →  .whl file  →  upload to PyPI  →  pip install pakgender
```

---

## Step 1 — One-time setup (do this once ever)

### 1a. Install the build tools

```bash
pip install build twine
```

- `build` — converts your project into a `.whl` file
- `twine` — uploads that `.whl` to PyPI securely

### 1b. Create a PyPI account

Go to **https://pypi.org/account/register/** and create an account.
Use the same email as your GitHub account — keeps things consistent.

### 1c. Create a TestPyPI account (for safe testing)

Go to **https://test.pypi.org/account/register/**

TestPyPI is an identical copy of PyPI for testing uploads.
You always test there first before the real PyPI.

### 1d. Create an API token on PyPI

1. Log in to pypi.org
2. Go to Account Settings → API tokens → Add API token
3. Name it `pakgender-upload`, scope: "Entire account" for first upload
4. Copy the token — it starts with `pypi-` — **save it somewhere safe, it shows only once**

Do the same on test.pypi.org for a TestPyPI token.

### 1e. Save your tokens in a credentials file

Create the file `~/.pypirc` on your computer:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-YOUR_REAL_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

Replace `pypi-YOUR_REAL_TOKEN_HERE` with your actual token.
This file is read automatically by twine — you never type your password again.

---

## Step 2 — Prepare the package for upload

### 2a. Make sure these files exist

Your project should look like this:

```
pakgender/                    ← root folder (the GitHub repo)
├── README.md                 ← required — shown on PyPI page
├── LICENSE                   ← required
├── pyproject.toml            ← required — package metadata
├── train_model.py            ← not included in pip package (root level)
├── convert_names.py          ← not included in pip package (root level)
└── pakgender/                ← the actual Python package
    ├── __init__.py
    ├── _version.py
    ├── predictor.py
    ├── preprocessor.py
    ├── dictionary.py
    ├── rules.py
    ├── ml_model.py
    ├── batch.py
    ├── cli.py                ← create this when ready
    └── data/
        ├── names.json        ← bundled in pip package ✓
        └── model.pkl         ← bundled in pip package ✓
```

### 2b. Verify pyproject.toml is correct

Key fields to check — open `pyproject.toml` and confirm:

```toml
[project]
name = "pakgender"          # must be unique on PyPI
version = "0.1.0"           # bump this for every new release
readme = "README.md"
```

The name `pakgender` — check it is not already taken on PyPI
before your first upload: https://pypi.org/project/pakgender/
If it shows "404 Not Found" you're clear to use it.

### 2c. Confirm data files are declared

This line in `pyproject.toml` tells the build system to include
your `names.json` and `model.pkl` inside the wheel:

```toml
[tool.setuptools.package-data]
pakgender = ["data/*.json", "data/*.pkl"]
```

Without this, users would get the Python code but no dictionary
or model — the library would silently fail.

---

## Step 3 — Build the package

From inside the `pakgender/` root folder (where `pyproject.toml` is):

```bash
cd pakgender
python -m build
```

This creates a `dist/` folder with two files:

```
dist/
├── pakgender-0.1.0.tar.gz      ← source distribution
└── pakgender-0.1.0-py3-none-any.whl   ← wheel (what pip installs)
```

The `.whl` is the important one. It is a zip file containing your
code + data bundled together.

---

## Step 4 — Test on TestPyPI first

### 4a. Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

Twine reads your `~/.pypirc` for credentials automatically.
You should see output like:

```
Uploading distributions to https://test.pypi.org/legacy/
Uploading pakgender-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 312.4 kB
View at: https://test.pypi.org/project/pakgender/0.1.0/
```

### 4b. Install from TestPyPI and verify it works

In a new terminal (or new virtual environment to be clean):

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            pakgender
```

The `--extra-index-url` is needed because TestPyPI doesn't have
`scikit-learn` — pip fetches that from the real PyPI.

Then test it:

```python
from pakgender import predict
print(predict("Fatima Noor"))
# Should print: GenderResult(gender='F', confidence=0.95, ...)
```

If this works, you're ready for the real upload.

---

## Step 5 — Upload to the real PyPI

```bash
twine upload dist/*
```

That's it. Your package is now live at:
**https://pypi.org/project/pakgender/**

And anyone anywhere can install it with:

```bash
pip install pakgender
```

---

## Step 6 — Releasing updates (every future version)

Every time you improve the library:

### 6a. Bump the version number

Open `pyproject.toml` and `pakgender/_version.py` and change:
```
version = "0.1.0"  →  version = "0.2.0"
```

Use **semantic versioning**:
- `0.1.0` → `0.1.1` — small bug fix
- `0.1.0` → `0.2.0` — new feature (e.g. added CLI, expanded dictionary)
- `0.1.0` → `1.0.0` — major change (e.g. breaking API change)

### 6b. Delete the old dist/ folder

```bash
rm -rf dist/
```

You must do this — twine will refuse to upload a version that
already exists on PyPI, and having old files in `dist/` causes confusion.

### 6c. Rebuild and upload

```bash
python -m build
twine upload dist/*
```

---

## Setting up the GitHub repository

PyPI and GitHub are separate but should be linked.

### Repo setup

```bash
cd pakgender          # your project root
git init
git add .
git commit -m "Initial release v0.1.0"
```

Create a new repo on GitHub named `pakgender`:
- Go to github.com/dino28575 → New repository
- Name: `pakgender`
- Description: `Gender inference for Pakistani and Arabic names`
- Public (important for portfolio visibility)
- Do NOT initialise with README — you already have one

Then push:

```bash
git remote add origin https://github.com/dino28575/pakgender.git
git branch -M main
git push -u origin main
```

### Linking PyPI badge to your repo

The README already has badge URLs set up. Once the package is on PyPI,
these shields will auto-populate with the real version number and
Python compatibility info.

### GitHub releases (optional but professional)

After each `git push`, go to your GitHub repo →
Releases → Draft a new release → tag it `v0.1.0` → publish.
This creates a clean version history visible to anyone viewing the repo.

---

## Full first-time checklist

```
[ ] pip install build twine
[ ] Created pypi.org account
[ ] Created test.pypi.org account
[ ] Saved API tokens in ~/.pypirc
[ ] Verified name 'pakgender' is available on PyPI
[ ] pyproject.toml has correct name, version, and package-data
[ ] README.md exists
[ ] LICENSE exists
[ ] python -m build  → dist/ folder created
[ ] twine upload --repository testpypi dist/*  → test upload works
[ ] pip install --index-url test.pypi.org  → test install works
[ ] predict("Fatima") returns correct result
[ ] twine upload dist/*  → live on PyPI
[ ] git push to GitHub
[ ] Verified https://pypi.org/project/pakgender/ shows correct page
```

---

## Common errors and fixes

**`File already exists` from twine**
You tried to upload a version number that already exists on PyPI.
PyPI never allows overwriting. Bump the version number and rebuild.

**`Invalid classifier` from twine**
A classifier in `pyproject.toml` has a typo. Run
`twine check dist/*` to see the exact error before uploading.

**`data/*.json not found` after pip install**
The `package-data` declaration in `pyproject.toml` is missing or wrong.
Confirm this block exists exactly:
```toml
[tool.setuptools.package-data]
pakgender = ["data/*.json", "data/*.pkl"]
```

**`ModuleNotFoundError: No module named 'pakgender.data'`**
The `data/` folder is missing an `__init__.py` — but actually
`importlib.resources.files()` does not require one in Python 3.9+.
If you see this error, check that `model.pkl` and `names.json`
are actually present in `pakgender/data/` before building.

**Model loads as None (ML layer inactive)**
The `.pkl` file was not included in the wheel. Verify
`pakgender/data/model.pkl` exists and `package-data` is declared.
Run `python -m build` and then open the `.whl` (it is a zip file)
to confirm the file is inside: `unzip -l dist/*.whl | grep pkl`
