FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-hin \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data
RUN python -c "\
import nltk; \
nltk.download('brown', quiet=True); \
nltk.download('punkt', quiet=True); \
nltk.download('averaged_perceptron_tagger', quiet=True)"

# Pre-download transformer models so first request isn't slow.
# Models are baked into the image (~600 MB total).
RUN python -c "\
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline; \
GPT2TokenizerFast.from_pretrained('distilgpt2'); \
GPT2LMHeadModel.from_pretrained('distilgpt2'); \
pipeline('text-classification', model='Hello-SimpleAI/chatgpt-detector-roberta', device=-1); \
print('Models cached successfully')"

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
