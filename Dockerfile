FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 1. 시스템 및 Python 빌드 의존성 설치
RUN apt-get update && \
    apt-get install -y wget git curl build-essential \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
    libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev uuid-dev \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# 2. Python 소스 다운로드 및 압축 해제
RUN wget https://www.python.org/ftp/python/3.10.17/Python-3.10.17.tgz && \
    tar -xf Python-3.10.17.tgz

# 3. Python 빌드 configure
WORKDIR /app/Python-3.10.17
RUN ./configure

# 4. Python make (병렬 제한)
RUN make -j2

# 5. Python 설치
RUN make altinstall

# 6. 환경 설정 및 pip 설치
WORKDIR /app
RUN rm -rf Python-3.10.17 Python-3.10.17.tgz && \
    ln -s /usr/local/bin/python3.10 /usr/bin/python3.10 && \
    python3.10 -m ensurepip && \
    python3.10 -m pip install --upgrade pip

# 7. requirements.txt 설치
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# 8. CLIP 설치
RUN python3.10 -m pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# 9. 모델 다운로드
RUN mkdir -p app/model && \
    wget -O app/model/insight-face-v3.pt \
    https://github.com/foamliu/InsightFace-v3/releases/download/v1.0/insight-face-v3.pt

# 10. 앱 코드 복사
COPY app/ app/

# 11. PYTHONPATH 설정 및 포트 노출
ENV PYTHONPATH=/app
EXPOSE 8000

# 12. FastAPI 실행
CMD ["python3.10", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
