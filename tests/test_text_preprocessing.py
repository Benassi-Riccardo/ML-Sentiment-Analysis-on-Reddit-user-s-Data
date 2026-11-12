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


class GoldEvaluationTests(unittest.TestCase):
    """Check the human-label comparison used for the README results."""

    def test_scores_each_method_and_rejects_bad_labels(self) -> None:
        import pandas as pd

        from src.gold_evaluation import score_against_gold

        class AlwaysNegative:
            def predict(self, texts):
                return ["negative"] * len(texts)

        gold = pd.DataFrame(
            {"text": ["I love it", "I hate it"], "human_label": ["Positive ", "negative"]}
        )
        results = score_against_gold(gold, AlwaysNegative()).set_index("method")
        self.assertEqual(results.loc["Training majority class (always positive)", "accuracy"], 0.5)
        self.assertEqual(results.loc["VADER (rule-based)", "accuracy"], 1.0)
        self.assertEqual(results.loc["TF-IDF + logistic regression", "accuracy"], 0.5)

        gold.loc[0, "human_label"] = "happy"
        with self.assertRaises(ValueError):
            score_against_gold(gold, AlwaysNegative())


class ApiTests(unittest.TestCase):
    """Check the prediction endpoint returns a label with a probability."""

    def test_predict_returns_label_and_confidence(self) -> None:
        import app

        response = app.predict(app.PredictRequest(text="I love this, best day ever!"))
        self.assertIn(response.predicted_sentiment, {"negative", "neutral", "positive"})
        self.assertGreater(response.confidence, 0.0)
        self.assertLessEqual(response.confidence, 1.0)

    def test_predict_rejects_text_with_no_known_words(self) -> None:
        import app
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as raised:
            app.predict(app.PredictRequest(text="è tutto fantastico, perfetto"))
        self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
