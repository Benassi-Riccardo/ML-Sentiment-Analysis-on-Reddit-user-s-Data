"""Load the source Reddit pickle dataset and export it as CSV."""

import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "redditDataset.pkl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "Dataset.csv"


def ingest_data(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Load the pickle dataset, validate its text field, and write a raw CSV."""
    with input_path.open("rb") as dataset_file:
        raw_data: Any = pickle.load(dataset_file)

    dataframe = pd.DataFrame(raw_data)
    if "text" not in dataframe.columns:
        raise ValueError("The source dataset must contain a 'text' column.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"Loaded {len(dataframe)} rows and wrote raw data to {output_path}.")
    return dataframe


def main() -> None:
    """Run data ingestion with optional input and output path overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    ingest_data(arguments.input, arguments.output)


if __name__ == "__main__":
    main()
