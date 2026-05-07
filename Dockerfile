FROM python:3.12-slim

# System deps needed by torch / jax / weasyprint and other heavy ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl \
    libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
    libffi-dev libglib2.0-0 libxml2 libxslt1.1 shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached separately from source code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project (data_src, metadata, src, etc.)
COPY . .

# main.py is run from its own directory so relative paths (../../.env, data_src/) resolve correctly
WORKDIR /app/src/pneuma_seeker

EXPOSE 8000

# Use 'fastapi run' (production mode, no auto-reload) instead of 'fastapi dev'
CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
