# 1. Base Image: Python 3.10 Slim (가볍고 안정적인 버전 사용)
FROM python:3.10-slim

# 2. 시스템 패키지 설치 (프로젝트의 핵심 의존성)
# - ffmpeg, libsndfile1: 오디오 파일 로드 및 변환 (Librosa, Demucs)
# - musescore3: MIDI를 PDF 악보로 변환 (MuseScore3)
# - build-essential: 일부 Python 패키지 컴파일용
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    musescore3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 생성
WORKDIR /app

# 4. 의존성 파일 복사 및 설치
# (캐시 효율성을 위해 requirements.txt를 먼저 복사)
COPY backend/requirements.txt .

# 5. Python 패키지 설치
# - PyTorch는 이미지 크기를 줄이기 위해 CPU 전용 버전을 설치합니다.
# - GPU를 사용하려면 '--extra-index-url' 부분을 제거하거나 CUDA 버전으로 변경하세요.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 6. 소스 코드 및 모델 파일 복사
# backend 폴더의 모든 내용을 컨테이너의 /app으로 복사
COPY backend/ .

# [중요] 모델 가중치 파일 (.pt) 확인
# 로컬의 backend/app/outputs/ 폴더에 best_model.pt가 있어야 복사됩니다.
# 만약 경로가 다르다면 아래 경로를 실제 모델 위치로 수정해야 합니다.
# COPY path/to/local/best_model.pt /app/app/outputs/best_model.pt

# 7. 환경 변수 설정
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
# [핵심] MuseScore를 GUI 없는 서버 환경(Docker)에서 실행하기 위한 필수 설정
ENV QT_QPA_PLATFORM=offscreen

# 8. 포트 노출 (Flask 기본 포트)
EXPOSE 5000

# 9. 실행 명령어 (Gunicorn WSGI 서버 사용)
# - workers: 동시 처리 프로세스 수 (CPU 코어 수에 따라 조정)
# - timeout: 오디오 처리 및 AI 분석 시간이 길어질 수 있으므로 120초로 넉넉하게 설정
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "run:app"]