FROM python:3.12-slim

WORKDIR /app

# Prevent Python from writing bytecode files and enable unbuffered logs.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download language data used by the API's text preprocessing.
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

COPY . .

# Fail the image build early if the pre-trained API artifact is not in the repo.
RUN test -f /app/sentiment_model.pkl

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
