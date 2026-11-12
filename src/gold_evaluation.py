"""Evaluate VADER and the trained model against hand-written (human) labels.

Step 1: python -m src.gold_evaluation sample     -> writes data/gold_labels.csv
Step 2: fill the empty human_label column by hand (positive / neutral / negative)
Step 3: python -m src.gold_evaluation evaluate   -> prints the comparison table
"""

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.model_trainer import (
    CLASS_LABELS,
    DEFAULT_INPUT_PATH,
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    load_split,
)
from src.text_preprocessing import clean_and_preprocess, label_sentiment

DEFAULT_GOLD_PATH = PROJECT_ROOT / "data" / "gold_labels.csv"
SAMPLE_SIZE = 200


def sample_for_labelling(
    input_path: Path = DEFAULT_INPUT_PATH,
    gold_path: Path = DEFAULT_GOLD_PATH,
    sample_size: int = SAMPLE_SIZE,
) -> pd.DataFrame:
    """Write random held-out posts with an empty human_label column."""
    # Never overwrite labels someone has already spent time writing.
    if gold_path.exists():
        raise FileExistsError(f"{gold_path} already exists; delete it to resample.")
    # Sample only test rows so the model has never seen the posts being judged.
    _, test_rows = load_split(input_path)
    sample = test_rows.sample(n=min(sample_size, len(test_rows)), random_state=RANDOM_STATE)
    # VADER's label is left out on purpose so it cannot bias the human labeller.
    gold = pd.DataFrame({"text": sample["text"].to_numpy(), "human_label": ""})
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(gold_path, index=False)
    print(f"Wrote {len(gold)} posts to {gold_path}. Fill in the human_label column.")
    return gold


def score_against_gold(gold: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Return accuracy and macro F1 of each labeller versus the human labels."""
    labels = gold["human_label"].astype(str).str.strip().str.lower()
    invalid = sorted(set(labels) - set(CLASS_LABELS))
    if invalid:
        raise ValueError(f"human_label must be one of {CLASS_LABELS}; found {invalid}.")

    predictions = {
        "Majority class (always positive)": ["positive"] * len(gold),
        "VADER (rule-based)": gold["text"].apply(label_sentiment).tolist(),
        "TF-IDF + logistic regression": model.predict(
            gold["text"].apply(clean_and_preprocess).tolist()
        ),
    }
    rows = [
        {
            "method": name,
            "accuracy": accuracy_score(labels, predicted),
            "macro_f1": f1_score(
                labels, predicted, labels=CLASS_LABELS, average="macro", zero_division=0
            ),
        }
        for name, predicted in predictions.items()
    ]
    return pd.DataFrame(rows).round(3)


def evaluate(
    gold_path: Path = DEFAULT_GOLD_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Score every labelled row of the gold file and print the comparison."""
    gold = pd.read_csv(gold_path, keep_default_na=False)
    gold = gold.loc[gold["human_label"].str.strip().ne("")]
    if gold.empty:
        raise ValueError(f"No rows in {gold_path} have a human_label yet.")
    results = score_against_gold(gold, joblib.load(model_path))
    print(f"Scored on {len(gold)} hand-labelled posts:\n")
    print(results.to_string(index=False))
    return results


def main() -> None:
    """Run the sample or evaluate step from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["sample", "evaluate"])
    arguments = parser.parse_args()
    if arguments.step == "sample":
        sample_for_labelling()
    else:
        evaluate()


if __name__ == "__main__":
    main()
