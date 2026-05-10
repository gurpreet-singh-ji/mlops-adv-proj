# =========================
# Stage 1: Builder
# =========================
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# Download nltk data
RUN python -m nltk.downloader -d /nltk_data stopwords wordnet


# =========================
# Stage 2: Final
# =========================
FROM python:3.10-slim

WORKDIR /app

# Copy python packages
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy nltk data
COPY --from=builder /nltk_data /usr/local/nltk_data

# Copy app files
COPY app/ /app
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

EXPOSE 5000

CMD ["python", "app.py"]