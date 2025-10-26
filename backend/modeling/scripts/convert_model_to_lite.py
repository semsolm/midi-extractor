import tensorflow as tf
import os
import sys

# 프로젝트 루트 경로 설정 (train.py 로직 참고)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

# 원본 모델 경로 (train.py에서 저장한 경로)
ORIGINAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "modeling", "outputs", "models")
ORIGINAL_MODEL_NAME = "drum_cnn_final.keras"
ORIGINAL_MODEL_PATH = os.path.join(ORIGINAL_MODEL_DIR, ORIGINAL_MODEL_NAME)

# 경량화된 모델을 저장할 최종 경로 (app/models/ 폴더)
EXPORT_MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "app", "models")
EXPORT_MODEL_NAME = "drum_cnn_final.tflite"
EXPORT_MODEL_PATH = os.path.join(EXPORT_MODEL_DIR, EXPORT_MODEL_NAME)


def convert_to_tflite():
    """원본 Keras 모델을 TFLite로 변환하여 app/models/ 에 저장합니다."""

    if not os.path.exists(ORIGINAL_MODEL_PATH):
        print(f"오류: 원본 모델 파일을 찾을 수 없습니다. 경로를 확인하세요:")
        print(f"{ORIGINAL_MODEL_PATH}")
        sys.exit(1)

    print(f"원본 모델 로딩 중: {ORIGINAL_MODEL_PATH}")
    model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)

    print("모델 변환 시작 (TensorFlow Lite)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # (선택) 최적화 옵션 (기본 양자화 적용)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    print("모델 변환 완료.")

    # 최종 저장 경로 생성
    os.makedirs(EXPORT_MODEL_DIR, exist_ok=True)

    with open(EXPORT_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

    print(f"\n성공! 경량화된 모델이 아래 경로에 저장되었습니다:")
    print(f"{EXPORT_MODEL_PATH}")
    print(f"파일 크기: {os.path.getsize(EXPORT_MODEL_PATH) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    convert_to_tflite()