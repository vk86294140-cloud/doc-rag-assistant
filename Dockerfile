FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend
COPY sample_docs ./sample_docs
RUN pip install --no-cache-dir -e .

EXPOSE 8000
# ANTHROPIC_API_KEY must be provided at runtime for live answers.
CMD ["uvicorn", "ragassistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
