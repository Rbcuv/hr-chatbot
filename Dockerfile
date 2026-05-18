FROM python:3.10-slim

# Install system dependencies for ChromaDB/LangChain
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy project files
COPY . .

# Set permissions for the app directory (Hugging Face requirement)
RUN chmod -R 777 /app

EXPOSE 7860

# Start with gunicorn for better performance
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]