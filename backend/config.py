# backend/config.py
import os

class Config:
    """Flask 애플리케이션의 기본 설정을 정의"""
    # 프로젝트의 기본 경로
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # 파일 업로드 및 결과 저장을 위한 폴더 경로
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RESULT_FOLDER = os.path.join(BASE_DIR, 'results')

    # 모델 파일 경로:
    MODEL_PATH = os.path.join(BASE_DIR, 'app', 'models', 'drum_cnn_final.tflite')

    # 폴더가 없으면 자동으로 생성
    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.RESULT_FOLDER, exist_ok=True)
        # [수정] 모델 폴더 생성 경로도 'app'을 포함하도록 변경
        os.makedirs(os.path.join(Config.BASE_DIR, 'app', 'models'), exist_ok=True)