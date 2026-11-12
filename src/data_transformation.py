"""Generate VADER labels and cleaned text from the raw dataset CSV."""

import argparse
from pathlib import Path

import pandas as pd

from src.text_preprocessing import clean_and_preprocess, label_sentiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "Dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "processed_dataset.csv"


def transform_data(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Label and clean each Reddit post, then save the processed CSV."""
    dataframe = pd.read_csv(input_path)
    if "text" not in dataframe.columns:
        raise ValueError("The input CSV must contain a 'text' column.")

    # The source repeats ~96% of posts; duplicates would leak across the train/test split.
    row_count = len(dataframe)
    dataframe = dataframe.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"Dropped {row_count - len(dataframe)} duplicate texts.")

    dataframe["sentiment"] = dataframe["text"].apply(label_sentiment)
    dataframe["clean_text"] = dataframe["text"].apply(clean_and_preprocess)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"Transformed {len(dataframe)} rows and wrote {output_path}.")
    print(dataframe["sentiment"].value_counts())
    return dataframe


def main() -> None:
    """Run transformation with optional input and output path overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    transform_data(arguments.input, arguments.output)


if __name__ == "__main__":
    main()
