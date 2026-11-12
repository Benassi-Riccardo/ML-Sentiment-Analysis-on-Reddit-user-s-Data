"""Train, evaluate, and log the VADER-labeled sentiment classifier."""

import argparse
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.text_preprocessing import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "processed_dataset.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "sentiment_model.pkl"
DEFAULT_CONFUSION_MATRIX_PATH = PROJECT_ROOT / "confusion_matrix.png"
CLASS_LABELS = ["negative", "neutral", "positive"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)


def load_split(
    input_path: Path = DEFAULT_INPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the processed CSV, drop unusable rows, and return (train, test) rows."""
    dataframe = pd.read_csv(input_path)
    required_columns = {"clean_text", "sentiment"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Processed CSV is missing required columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.dropna(subset=["clean_text", "sentiment"])
    dataframe = dataframe.loc[dataframe["clean_text"].str.strip().ne("")].copy()
    # Different raw posts can clean to identical text; keep one so test rows stay unseen.
    dataframe = dataframe.drop_duplicates(subset="clean_text")
    if dataframe.empty:
        raise ValueError("No usable rows remain in the processed dataset.")
    if set(dataframe["sentiment"].unique()) != set(CLASS_LABELS):
        raise ValueError(
            "Training requires positive, neutral, and negative examples."
        )

    train_rows, test_rows = train_test_split(
        dataframe,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataframe["sentiment"],
    )
    return train_rows, test_rows


def train_model(
    input_path: Path = DEFAULT_INPUT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    confusion_matrix_path: Path = DEFAULT_CONFUSION_MATRIX_PATH,
) -> Pipeline:
    """Fit and evaluate a text-classification pipeline, then save and log it."""
    train_rows, test_rows = load_split(input_path)
    X_train, y_train = train_rows["clean_text"], train_rows["sentiment"]
    X_test, y_test = test_rows["clean_text"], test_rows["sentiment"]

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=MAX_FEATURES,
                    ngram_range=NGRAM_RANGE,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=500,
                    solver="lbfgs",
                    # Neutral is ~10% of posts; without reweighting it is never predicted.
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    _, _, macro_f1_score, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    print(classification_report(y_test, predictions, zero_division=0))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    confusion = confusion_matrix(y_test, predictions, labels=CLASS_LABELS)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("True Sentiment")
    plt.savefig(confusion_matrix_path, bbox_inches="tight")
    plt.close()

    mlflow.set_experiment("Reddit_Sentiment_Analysis")
    with mlflow.start_run(run_name="Logistic_Regression_VADER_Baseline"):
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "C": 1.0,
                "max_iter": 500,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "deduplicated": True,
                "tfidf_max_features": MAX_FEATURES,
                "tfidf_ngram_range": str(NGRAM_RANGE),
                "label_method": "VADER compound score",
                "vader_positive_threshold": POSITIVE_THRESHOLD,
                "vader_negative_threshold": NEGATIVE_THRESHOLD,
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "macro_f1_score": macro_f1_score,
            }
        )
        mlflow.log_artifact(str(confusion_matrix_path))
        mlflow.sklearn.log_model(model, name="sentiment_pipeline")

    print(f"Saved trained pipeline to {model_path}.")
    return model


def main() -> None:
    """Train the model from the processed CSV using optional path overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--confusion-matrix-output",
        type=Path,
        default=DEFAULT_CONFUSION_MATRIX_PATH,
    )
    arguments = parser.parse_args()
    train_model(
        arguments.input,
        arguments.model_output,
        arguments.confusion_matrix_output,
    )


if __name__ == "__main__":
    main()
