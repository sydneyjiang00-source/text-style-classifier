# Text Style Classifier

A from-scratch text classification model in pure Python — **no external ML libraries** — that builds a statistical "fingerprint" of a body of text and uses a log-likelihood comparison to decide which of two known authors/sources an unknown text most resembles. A lightweight take on authorship attribution.

## How it works

Each body of text is modeled across **five features**:

| Feature | What it captures |
|---|---|
| Word frequencies | Vocabulary and word choice |
| Word lengths | Distribution of short vs. long words |
| Word stems | Root forms (a simple rule-based stemmer) |
| Sentence lengths | Sentence complexity / rhythm |
| Adjacent word pairs | Local phrasing patterns |

To classify an unknown text, the model computes a **log-likelihood score** for each feature against two known source models, then takes a majority vote across the five features to pick the more likely source.

## Files

- `text_style_classifier.py` — the `TextModel` class and all supporting functions.

## Usage

```python
from text_style_classifier import TextModel

# Build models from known sources
source1 = TextModel('Author A')
source1.add_file('author_a.txt')

source2 = TextModel('Author B')
source2.add_file('author_b.txt')

# Classify an unknown text
unknown = TextModel('unknown')
unknown.add_file('mystery.txt')
unknown.classify(source1, source2)
```

Running the file directly executes a small built-in sanity check:

```bash
python text_style_classifier.py
```

The `run_tests()` function shows a fuller example: it builds source models from the speeches of two companies' management teams (Cleveland-Cliffs and Delta) and classifies several new texts by writing style. It expects the corresponding `.txt` files in the working directory.

## Notes

- Implemented entirely with the Python standard library (`math`, `ast`) to demonstrate the mechanics of a naive-Bayes-style classifier without relying on packaged tools.
- The rule-based `stem()` function is recursive and handles common English suffixes (`-ing`, `-ed`, `-ly`, `-tion`, etc.) plus a few irregular forms.
- Saved models are read back with `ast.literal_eval`, which safely parses a Python literal rather than executing arbitrary code.
