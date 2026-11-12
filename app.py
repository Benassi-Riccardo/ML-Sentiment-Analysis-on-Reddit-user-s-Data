"""FastAPI service for the saved Reddit sentiment classifier."""

from pathlib import Path  # Resolve model and input-resource paths from this file.
from typing import Any  # Type the model returned by joblib without weakening API types.

import joblib  # Load the fitted TF-IDF and logistic-regression pipeline.
from fastapi import FastAPI, HTTPException  # Define the HTTP service and API errors.
from pydantic import BaseModel  # Validate incoming JSON and format the response.
from src.text_preprocessing import clean_and_preprocess, ensure_nltk_resources

# Keep the model path independent of the shell's current working directory.
MODEL_PATH = Path(__file__).resolve().parent / "sentiment_model.pkl"

# Ensure the corpora used by the shared cleaner are available to the API.
ensure_nltk_resources()

# Load the model at startup so a missing or invalid artifact fails visibly.
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model artifact was not found: {MODEL_PATH}")
model: Any = joblib.load(MODEL_PATH)

# Describe this API in the generated OpenAPI documentation.
app = FastAPI(
    title="Reddit Sentiment Analysis API",
    description="Inference API for a TF-IDF and logistic-regression sentiment model.",
)
class PredictRequest(BaseModel):
    """JSON payload accepted by the prediction endpoint."""

    text: str


class PredictResponse(BaseModel):
    """Prediction plus the text normalization used to produce it."""

    raw_text: str
    cleaned_text: str
    predicted_sentiment: str
    confidence: float


@app.get("/")
def health_check() -> dict[str, str | bool]:
    """Return service readiness and confirm the model loaded at startup."""
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Predict positive, neutral, or negative sentiment and its probability."""
    # Normalize request text to match the model's training features.
    cleaned_text = clean_and_preprocess(payload.text)
    # Reject inputs that contain no words the model can use.
    if not cleaned_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text input contains no usable words after cleaning.",
        )
    # With no known words the features are all zero and the model would just guess its prior.
    if model.named_steps["tfidf"].transform([cleaned_text]).nnz == 0:
        raise HTTPException(
            status_code=422,
            detail="None of these words are in the model's vocabulary. "
            "The model was trained on English Reddit posts only.",
        )
    # The saved scikit-learn pipeline applies TF-IDF before classification.
    probabilities = model.predict_proba([cleaned_text])[0]
    best_index = int(probabilities.argmax())
    # Return both the original text and the model's predicted sentiment.
    return PredictResponse(
        raw_text=payload.text,
        cleaned_text=cleaned_text,
        predicted_sentiment=str(model.classes_[best_index]),
        confidence=round(float(probabilities[best_index]), 4),
    )
