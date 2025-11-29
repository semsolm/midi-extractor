# backend/config.py
import os
class Config:
    """Flask 애플리케이션의 기본 설정을 정의"""
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # GCS 버킷 이름 정의
    GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'midi-extrator-files')


    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RESULT_FOLDER = os.path.join(BASE_DIR, 'results')

    # [수정] TFLite(.tflite) -> PyTorch(.pt) 모델 파일명 변경
    # os 검사 작성 (기본값은 Docker 환경 (Linux))
    MUSESCORE_PATH = os.environ.get('MUSESCORE_PATH', '/usr/bin/musescore3')
    MODEL_PATH = os.path.join(BASE_DIR, 'app', 'models', 'best_model.pt')

    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.RESULT_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(Config.BASE_DIR, 'app', 'models'), exist_ok=True)