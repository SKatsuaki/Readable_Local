FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/nmt-venv \
    && /opt/nmt-venv/bin/pip install --no-cache-dir \
        ctranslate2 \
        torch \
        "transformers>=4.43,<5" \
        sentencepiece

WORKDIR /srv/nmt
COPY nmt_server.py ./

ENV NMT_MODEL_ID=facebook/nllb-200-distilled-600M
ENV NMT_MODEL_DIR=/models/nllb-200-distilled-600m
ENV HF_HOME=/models/huggingface
ENV NMT_PORT=8767
ENV NMT_MAX_BATCH_TOKENS=8192
ENV NMT_MAX_INPUT_TOKENS=384
ENV PATH=/opt/nmt-venv/bin:${PATH}

EXPOSE 8767

CMD ["python3", "nmt_server.py"]
