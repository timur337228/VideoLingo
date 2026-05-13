FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-noto \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Keep the install order aligned with install.py:
# 1) install PyTorch first so its CUDA wheel is pinned correctly
# 2) install demucs without deps to avoid torchaudio downgrade
# 3) install the rest of the project requirements
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 \
    torchaudio==2.8.0

RUN python -m pip install --no-cache-dir --no-deps \
    "demucs[dev] @ git+https://github.com/adefossez/demucs"

RUN python -m pip install --no-cache-dir dora-search openunmix lameenc

RUN grep -vE '^(torch|torchaudio)==|^demucs @ git\+' requirements.txt > filtered-requirements.txt \
    && python -m pip install --no-cache-dir -r filtered-requirements.txt \
    && rm filtered-requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "st.py", "--server.address=0.0.0.0", "--server.port=8501"]
