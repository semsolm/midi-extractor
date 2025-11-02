#
# STAGE 1: "빌더" (모델 변환용)
# ---------------------------------
# (STAGE 1은 기존과 동일)
FROM python:3.10-slim AS builder

WORKDIR /app
COPY . .
RUN pip install tensorflow
RUN python backend/modeling/scripts/convert_model_to_lite.py


#
# STAGE 2: "시연용" (NVIDIA 공식 CUDA 11.8 기반)
# ---------------------------------
# (베이스 이미지는 NVIDIA 공식 이미지로 동일하게 유지)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 1. Python 3.10, pip, FFmpeg 설치 (기존과 동일)
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip ffmpeg && \
    musescore3 && \
    fonts-freefont-ttf && \
    rm -rf /var/lib/apt/lists/*

# 2. [수정] 'pip3' 대신 'python3 -m pip'를 사용하여 라이브러리 설치
COPY backend/requirements.txt .
RUN python3 -m pip install --upgrade pip  # pip 자체를 먼저 업그레이드
RUN python3 -m pip install -r requirements.txt

# 3. 서버 실행에 필요한 파일만 복사 (기존과 동일)
COPY backend/app ./app
COPY backend/config.py .
COPY backend/run.py .

# 4. 1단계에서 생성한 .tflite 파일 복사 (기존과 동일)
COPY --from=builder /app/backend/app/models/drum_cnn_final.tflite ./app/models/drum_cnn_final.tflite

# 5. 서버 실행에 필요한 폴더 생성 (기존과 동일)
RUN mkdir -p /app/uploads /app/results

# 6. Flask 서버 포트 개방 (기존과 동일)
EXPOSE 5000

# 7. [수정] 'python' 대신 'python3'로 서버 실행
CMD ["python3", "-u", "run.py"]