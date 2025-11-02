# modeling/scripts/train.py
import os
import sys
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

# src 폴더를 파이썬 경로에 추가하여 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_processed_data
from src.model import build_cnn_model


# 모델 저장 디렉토리
OUTPUT_MODEL_DIR = "../outputs/models"
OUTPUT_MODEL_NAME = "drum_cnn_final.keras" # .h5 대신 최신 .keras 포맷 사용

# --- MLflow 중앙 저장소 설정 ---
# 이 스크립트(train.py)의 위치는 backend/modeling/scripts/ 입니다.
# 우리는 3단계 상위 폴더인 프로젝트 루트(semsolm/midi-extractor-dev)에 있는
# mlruns 폴더를 중앙 저장소로 사용할 것입니다.
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
    MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")

    # MLflow가 파일 경로를 인식할 수 있도록 URI 형식(file://...)으로 변환
    MLFLOW_TRACKING_URI = Path(MLRUNS_DIR).as_uri()

    # MLflow에게 "모든 데이터를 이 URI에 저장하라"고 명시적으로 지시
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow 추적 URI가 설정되었습니다: {MLFLOW_TRACKING_URI}")

except Exception as e:
    print(f"MLflow 추적 URI 설정 중 오류 발생: {e}")
    print("스크립트를 계속 진행하지만, MLflow 저장이 기본 위치에 될 수 있습니다.")
# ---

# --- 설정 ---
DATA_PATH = os.path.join(PROJECT_ROOT, "backend", "modeling", "data", "raw")
NUM_CLASSES = 3 # 드럼 소리 클래스 수
INPUT_SHAPE = (128, 128, 1)

# 상대 경로('../outputs/models') 대신 PROJECT_ROOT를 기준으로 하는 절대 경로 사용
OUTPUT_MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "modeling", "outputs", "models")
OUTPUT_MODEL_NAME = "drum_cnn_final.keras" # .h5 대신 최신 .keras 포맷 사용

# --- MLflow 설정 ---
# 실험 이름 설정 (MLflow UI에 표시될 이름)
mlflow.set_experiment("Drum Sound Classification")

# --- 1. 데이터 로드 및 분할 ---
print(f"({DATA_PATH}) 데이터 로딩 중...")
X, y = load_processed_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

# --- 2. MLflow 실행 시작 ---
# 이 블록 안의 모든 학습 과정이 MLflow에 기록됩니다.
with mlflow.start_run():
    # 데이터셋의 출처와 버전을 파라미터로 기록
    mlflow.log_param("dataset_source_url", "https://www.kaggle.com/datasets/anubhavchhabra/drum-kit-sound-samples")
    mlflow.log_param("dataset_version", "1.0_downloaded_2025-10-24")

    # --- 하이퍼파라미터 설정 및 기록 ---
    epochs = 50
    batch_size = 32
    learning_rate = 0.0005

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

    # --- 학습에 사용된 원본 데이터 폴더를 'dataset'이란 이름으로 기록 ---
    print("원본 데이터셋을 MLflow 아티팩트로 기록합니다...")
    mlflow.log_artifacts(DATA_PATH, artifact_path="dataset")
    print("데이터셋 기록 완료.")
    # ---

    # --- 3. 모델 생성 ---
    print("모델 생성 중...")
    model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    # --- 4. 모델 학습 ---
    print("모델 학습 시작...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    )

    # --- 5. 최종 성능 지표 기록 ---
    val_loss, val_accuracy = model.evaluate(X_test, y_test)
    mlflow.log_metric("final_val_loss", val_loss)
    mlflow.log_metric("final_val_accuracy", val_accuracy)

    # --- 6. 최종 모델을 outputs/models 디렉토리에 저장 ---
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    final_model_path = os.path.join(OUTPUT_MODEL_DIR, OUTPUT_MODEL_NAME)

    try:
        model.save(final_model_path)
        print(f"\n최종 모델이 계획된 경로에 저장되었습니다: {os.path.abspath(final_model_path)}")
        # MLflow에도 저장된 경로를 파라미터로 기록
        mlflow.log_param("saved_model_path", final_model_path)
    except Exception as e:
        print(f"모델 파일 저장 중 오류 발생: {e}")

    # --- 7. 학습된 모델을 MLflow에 아티팩트(결과물)로 저장 ---
    mlflow.keras.log_model(
        model,
        name="model"
    )

    print("\nMLflow 실행 완료!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")