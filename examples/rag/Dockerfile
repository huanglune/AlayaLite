FROM python:3.12-slim-bookworm

ARG ALAYALITE_VERSION=1.1.0

WORKDIR /workspace
COPY examples/rag/requirements.txt /workspace/examples/rag/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /workspace/examples/rag/requirements.txt && \
    python -m pip install --no-cache-dir "alayalite==${ALAYALITE_VERSION}"

COPY examples/rag /workspace/examples/rag

ENV ALAYALITE_RAG_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "examples/rag/ui.py", "--server.address=0.0.0.0"]
