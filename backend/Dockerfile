FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     curl     && rm -rf /var/lib/apt/lists/*

# Copy requirements from backend/ (relativo a raiz del proyecto porque el contexto es .)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY backend/app/ ./app/

# Copy model code and artifacts (relativo a raiz del proyecto)
COPY model/ ./model/
COPY scraping/ ./scraping/

# Create directories for data and artifacts
RUN mkdir -p data/raw data/processed data/external data/splits data/uploads model/artifacts

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3     CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
