# Contributing to pakgender

Thank you for wanting to make `pakgender` better! This project thrives on community data and insights to fix Western bias in data pipelines.

## How You Can Help

### 1. Adding Names to the Dictionary
The easiest way to contribute is to add names to `pakgender/data/names.json`.
* Ensure names are in **lowercase**.
* Assign the correct gender (`M`, `F`, or `U` for ambiguous/neutral names).

### 2. Improving Rules
If you notice a consistent pattern in Pakistani names (e.g., specific suffixes), you can suggest updates to the rule engine in `pakgender/rules.py`.

### 3. Reporting Bugs
If the library misclassifies a common name, please open an Issue with:
* The input name.
* The incorrect result you got.
* The expected result.

## Pull Request Guidelines
1. Fork the repo and create your branch (`git checkout -b feature/amazing-feature`).
2. Run existing tests to ensure nothing is broken.
3. Commit your changes with clear messages.
4. Push to the branch and open a Pull Request!
