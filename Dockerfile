FROM apache/airflow:2.7.1-python3.10

USER root
COPY requirements-heavy.txt /requirements-heavy.txt
COPY requirements.txt /requirements.txt
USER airflow

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        platformdirs==3.8.1 \
        protobuf==4.21.12 \
        "botocore>=1.31.85,<1.32.0" \
        "boto3>=1.28.0,<1.29.0"

# ── Heavy layer (torch ~2 GB) — cached unless requirements-heavy.txt changes ──
RUN pip install --no-cache-dir -r /requirements-heavy.txt

# ── Light layer — re-runs in ~1 min when requirements.txt changes ─────────────
RUN pip install --no-cache-dir -r /requirements.txt
