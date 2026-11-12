"""Tests for the shared sentiment labeling and text-cleaning functions."""

import unittest

from src.text_preprocessing import clean_and_preprocess, label_sentiment


class TextPreprocessingTests(unittest.TestCase):
    """Check the labels and text normalization used by training and serving."""

    def test_label_sentiment_returns_all_three_classes(self) -> None:
        self.assertEqual(label_sentiment("I absolutely love this!"), "positive")
        self.assertEqual(label_sentiment("I absolutely hate this!"), "negative")
        self.assertEqual(label_sentiment(""), "neutral")

    def test_cleaner_preserves_negation_and_removes_url(self) -> None:
        cleaned = clean_and_preprocess("This is NOT good: https://example.com")
        self.assertIn("not", cleaned.split())
        self.assertIn("good", cleaned.split())
        self.assertNotIn("https", cleaned)

    def test_cleaner_handles_non_string_input(self) -> None:
        self.assertEqual(clean_and_preprocess(None), "")


class DataTransformationTests(unittest.TestCase):
    """Check that repeated posts cannot leak across the train/test split."""

    def test_transform_drops_duplicate_texts(self) -> None:
        import tempfile
        from pathlib import Path

        import pandas as pd

        from src.data_transformation import transform_data

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "raw.csv"
            pd.DataFrame({"text": ["I love it", "I love it", "I hate it"]}).to_csv(
                input_path, index=False
            )
            result = transform_data(input_path, Path(tmp) / "out.csv")
        self.assertEqual(result["text"].tolist(), ["I love it", "I hate it"])


if __name__ == "__main__":
    unittest.main()
