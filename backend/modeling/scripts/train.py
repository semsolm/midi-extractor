# modeling/scripts/train.py
import os
import sys
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# src 폴더를 파이썬 경로에 추가하여 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_processed_data
from src.model import build_cnn_model

# --- 설정 ---
DATA_PATH = "../data/processed"
NUM_CLASSES = 3
INPUT_SHAPE = (128, 128, 1)

# --- MLflow 설정 ---
# 실험 이름 설정 (MLflow UI에 표시될 이름)
mlflow.set_experiment("Drum Sound Classification")

# --- 1. 데이터 로드 및 분할 ---
print("데이터 로딩 중...")
X, y = load_processed_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

# --- 2. MLflow 실행 시작 ---
# 이 블록 안의 모든 학습 과정이 MLflow에 기록됩니다.
with mlflow.start_run():
    # --- 하이퍼파라미터 설정 및 기록 ---
    epochs = 50
    batch_size = 32
    learning_rate = 0.001

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

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

    # --- 6. 학습된 모델을 MLflow에 아티팩트(결과물)로 저장 ---
    # "model"이라는 이름으로 Keras 모델을 저장합니다.
    mlflow.keras.log_model(model, "model")

    print("\nMLflow 실행 완료!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")