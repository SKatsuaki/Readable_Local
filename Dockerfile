FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        fontconfig \
        fonts-ipafont-gothic \
        fonts-ipafont-mincho \
        fonts-noto-cjk \
        latexmk \
        poppler-utils \
        texlive-fonts-recommended \
        texlive-lang-japanese \
        texlive-latex-extra \
        texlive-latex-recommended \
        texlive-luatex \
        texlive-science \
        texlive-xetex \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py readable_pdf.py readable_tex.py README.md ./
COPY scripts ./scripts

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /tmp/readable/tmp /tmp/readable/output \
    && chown -R appuser:appuser /app /tmp/readable

USER appuser

ENV HOST=0.0.0.0
ENV PORT=8766
ENV READABLE_TMP_DIR=/tmp/readable/tmp
ENV READABLE_OUTPUT_DIR=/tmp/readable/output
ENV READABLE_RECOVER_OUTPUTS=0
ENV READABLE_MAX_CONCURRENT_JOBS=4
ENV READABLE_MAX_QUEUED_JOBS=2
ENV READABLE_DYNAMIC_JOB_LIMITS=1
ENV READABLE_MEMORY_BUDGET_GB=100
ENV READABLE_MIN_FREE_MEMORY_GB=8
ENV READABLE_JOB_MEMORY_RESERVE_GB=20
ENV READABLE_JOB_TTL_SECONDS=1800

EXPOSE 8766

CMD ["python", "app.py"]
