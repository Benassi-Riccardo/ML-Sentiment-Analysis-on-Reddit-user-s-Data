"""Shared sentiment labeling and text-cleaning functions."""

import re
from functools import lru_cache
from typing import Any

import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def ensure_nltk_resources() -> None:
    """Ensure the NLTK corpora used for stopwords and lemmatization are installed."""
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for resource_path, download_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(download_name, quiet=True):
                raise RuntimeError(
                    f"Could not download required NLTK resource: {download_name}"
                )


@lru_cache(maxsize=1)
def _preprocessing_resources() -> tuple[WordNetLemmatizer, frozenset[str]]:
    """Create and cache reusable text-cleaning resources."""
    ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    english_stop_words = frozenset(stopwords.words("english")) - {
        "no",
        "nor",
        "not",
    }
    return lemmatizer, english_stop_words


@lru_cache(maxsize=1)
def _sentiment_analyzer() -> SentimentIntensityAnalyzer:
    """Create and cache the VADER analyzer used to generate baseline labels."""
    return SentimentIntensityAnalyzer()


def label_sentiment(text: Any) -> str:
    """Map VADER's compound score to positive, neutral, or negative."""
    safe_text = text if isinstance(text, str) else ""
    compound_score = _sentiment_analyzer().polarity_scores(safe_text)["compound"]
    if compound_score >= POSITIVE_THRESHOLD:
        return "positive"
    if compound_score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def clean_and_preprocess(text: Any) -> str:
    """Normalize text identically for offline training and API inference."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = emoji.demojize(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    lemmatizer, english_stop_words = _preprocessing_resources()
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in english_stop_words
    ]
    return " ".join(cleaned_words)
