FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# DuckDB extensions used by uploads (excel) and connectors (postgres, sqlite),
# installed at build time so they don't need network access at run time.
RUN python -c "import duckdb; con = duckdb.connect(); [con.execute(f'INSTALL {e}') for e in ('excel', 'postgres', 'sqlite')]"

COPY src /app/src

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "pneuma_seeker.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
