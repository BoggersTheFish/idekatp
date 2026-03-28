FROM python:3.12-slim

WORKDIR /app

# System deps for numpy / torch CPU wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch (smaller image; swap whl URL for CUDA wheel if needed)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt fastapi uvicorn

COPY . .

# Initialise submodules if present (build context must include vendor/)
# Submodules are expected to be checked out: docker build after git submodule update --init

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "inference_server.py", "--host", "0.0.0.0", "--port", "8000"]
