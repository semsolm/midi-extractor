# STAGE 1: "빌더" (모델 변환용)
# ---------------------------------
# .keras 모델을 .tflite로 변환하기 위해 전체 TensorFlow를 설치
FROM python:3.10-slim AS builder

WORKDIR /app
COPY . .

# [builder] TensorFlow 설치 (tflite-runtime이 아닌)
RUN pip install tensorflow

# [builder] 변환 스크립트 실행
RUN python backend/modeling/scripts/convert_model_to_lite.py


#
# STAGE 2: "프로덕션" (NVIDIA CUDA 11.8 기반)
# ---------------------------------
# cu118은 requirements.txt의 torch/torchaudio 버전과 일치
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 1. Python 3.10, pip, FFmpeg 및 [필수] MuseScore 3와 폰트 설치
# [수정] musescore3 뒤에 \ 를 추가하고, fonts-freefont-ttf의 주석을 해제했습니다.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        ffmpeg \
        musescore3 \
        fonts-freefont-ttf && \
    rm -rf /var/lib/apt/lists/*

# 2. Python 라이브러리 설치
# (backend/requirements.txt에는 music21이 포함되어 있어야 합니다)
COPY backend/requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# 3. 서버 실행에 필요한 파일만 복사
# (run.py가 /app에서 app 폴더를 찾을 수 있도록 함)
COPY backend/app ./app
COPY backend/config.py .
COPY backend/run.py .

# 4. 1단계(builder)에서 생성한 .tflite 파일 복사
# [수정] config.py 경로에 맞게 ./app/models/로 복사
COPY --from=builder /app/backend/app/models/drum_cnn_final.tflite ./app/models/drum_cnn_final.tflite

# 5. 서버 실행에 필요한 폴더 생성
RUN mkdir -p /app/uploads /app/results

# 6. Flask 서버 포트 개방
EXPOSE 5000

# 7. Flask 서버 실행
# (참고: GCS/Cloud Run Worker로 변경 시 이 CMD를 수정하게 됩니다)
CMD ["python3", "-u", "run.py"]