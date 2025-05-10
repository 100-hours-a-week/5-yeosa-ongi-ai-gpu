FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app/ app/

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# 포트 설정
EXPOSE 8000

# 실행 명령
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]




FROM python:3.10.17-slim

WORKDIR /app

# 필요한 패키지 설치 (wget 포함)
RUN apt-get update && apt-get install -y \
    git wget \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# PYTHONPATH 설정
ENV PYTHONPATH=/app

# 앱 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
