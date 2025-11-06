import os
import sys
import mlflow
import mlflow.keras
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
import numpy as np
import tensorflow as tf


# src 폴더를 파이썬 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_processed_data
from src.model import build_cnn_model

# --- 경로 설정(현재 파일 경로) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

# ⚠️ 중요: 멀티라벨 데이터 경로로 변경!
DATA_PATH = os.path.join(PROJECT_ROOT, "backend", "modeling", "data", "raw")

# MLflow 설정
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
MLFLOW_TRACKING_URI = Path(MLRUNS_DIR).as_uri()

# 모델 저장 경로
OUTPUT_MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "modeling", "outputs", "sig_model")
OUTPUT_MODEL_NAME = "drum_multilabel_cnn.keras"

# --- 모델 설정 ---
NUM_CLASSES = 3  # kick, snare, hihat
INPUT_SHAPE = (128, 128, 1)

# --- MLflow 설정 ---
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow 추적 URI: {MLFLOW_TRACKING_URI}")
except Exception as e:
    print(f"MLflow 설정 오류: {e}")

mlflow.set_experiment("Drum Multilabel Classification")


# ---------- 멀티라벨 메트릭 계산 헬퍼 ----------

def compute_multilabel_metrics(y_true, y_prob, threshold=0.5, *, return_predictions=False):
    """Compute multilabel classification metrics using a probability threshold."""

    y_pred = (y_prob > threshold).astype(int)

    metrics = {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_samples": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "jaccard": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
    }

    if return_predictions:
        return metrics, y_pred

    return metrics


# ---------- 커스텀 콜백: 멀티라벨 메트릭 로깅 ----------
class MultilabelMetricsCallback(Callback):
    def __init__(self, X_val, y_val, threshold=0.5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        prob = self.model.predict(self.X_val, verbose=0)
        metrics = compute_multilabel_metrics(self.y_val, prob, threshold=self.threshold)

        for name, value in metrics.items():
            mlflow.log_metric(f"val_{name}", float(value), step=epoch)


# -----------------------------------------------------


# --- 1. 데이터 로드 및 분할 ---
print(f"\n{'=' * 60}")
print(f"멀티라벨 드럼 분류 모델 학습")
print(f"{'=' * 60}\n")

X, y = load_processed_data(DATA_PATH)

# 데이터 검증
assert len(y.shape) == 2, f"y는 2D 배열이어야 합니다. 현재: {y.shape}"
assert y.shape[1] == NUM_CLASSES, f"라벨 수 불일치. 예상: {NUM_CLASSES}, 실제: {y.shape[1]}"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None  # 멀티라벨은 stratify 사용 어려움
)

print(f"\n데이터 분할:")
print(f"  훈련: {X_train.shape}, {y_train.shape}")
print(f"  테스트: {X_test.shape}, {y_test.shape}")

# 라벨 분포 확인
print(f"\n라벨 분포:")
print(f"  전체 - Kick: {y.sum(axis=0)[0]}, Snare: {y.sum(axis=0)[1]}, Hihat: {y.sum(axis=0)[2]}")
print(f"  훈련 - Kick: {y_train.sum(axis=0)[0]}, Snare: {y_train.sum(axis=0)[1]}, Hihat: {y_train.sum(axis=0)[2]}")

# --- 2. MLflow 실행 시작 ---
with mlflow.start_run(run_name="multilabel_sigmoid"):
    mlflow.keras.autolog(log_models=False)  # 모델은 수동으로 저장

    # 파라미터 로깅
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    threshold = 0.5

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("num_classes", NUM_CLASSES)
    mlflow.log_param("input_shape", INPUT_SHAPE)
    mlflow.log_param("dataset_type", "multilabel_synthetic")

    # 데이터셋 기록
    print("\n원본 데이터셋을 MLflow에 기록 중...")
    mlflow.log_artifacts(DATA_PATH, artifact_path="dataset")

    # --- 3. 모델 생성 ---
    print("\n모델 생성 중...")
    model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    print("\n모델 구조:")
    model.summary()

    # --- 4. 모델 학습 ---
    print(f"\n{'=' * 60}")
    print("모델 학습 시작")
    print(f"{'=' * 60}\n")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            MultilabelMetricsCallback(X_test, y_test, threshold=threshold)
        ],
        verbose=1
    )

    # --- 5. 최종 평가 ---
    print(f"\n{'=' * 60}")
    print("최종 모델 평가")
    print(f"{'=' * 60}\n")

    # 기본 메트릭
    eval_dict = model.evaluate(X_test, y_test, return_dict=True, verbose=0)
    for k, v in eval_dict.items():
        mlflow.log_metric(f"final_{k}", float(v))
        print(f"  {k}: {v:.4f}")

    # 추가 멀티라벨 메트릭
    y_pred_prob = model.predict(X_test, verbose=0)
    metrics, y_pred = compute_multilabel_metrics(
        y_test, y_pred_prob, threshold=threshold, return_predictions=True
    )

    for name, value in metrics.items():
        mlflow.log_metric(f"final_{name}", float(value))

    print(f"\n멀티라벨 메트릭:")
    print(f"  F1 Score (micro): {metrics['f1_micro']:.4f}")
    print(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 Score (samples): {metrics['f1_samples']:.4f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"  Jaccard Score: {metrics['jaccard']:.4f}")

    # 예측 예시 출력
    print(f"\n예측 예시 (처음 5개):")
    for i in range(min(5, len(y_test))):
        true_labels = []
        pred_labels = []

        if y_test[i][0] == 1: true_labels.append("Kick")
        if y_test[i][1] == 1: true_labels.append("Snare")
        if y_test[i][2] == 1: true_labels.append("Hihat")

        if y_pred[i][0] == 1: pred_labels.append("Kick")
        if y_pred[i][1] == 1: pred_labels.append("Snare")
        if y_pred[i][2] == 1: pred_labels.append("Hihat")

        true_str = ', '.join(true_labels) if true_labels else 'None'
        pred_str = ', '.join(pred_labels) if pred_labels else 'None'
        match = "✓" if true_labels == pred_labels else "✗"

        print(f"  {i + 1}. 실제: [{true_str}] / 예측: [{pred_str}] {match}")

    # --- 6. 모델 저장 ---
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    final_model_path = os.path.join(OUTPUT_MODEL_DIR, OUTPUT_MODEL_NAME)

    try:
        model.save(final_model_path)
        print(f"\n최종 모델 저장: {os.path.abspath(final_model_path)}")
        mlflow.log_param("saved_model_path", final_model_path)
    except Exception as e:
        print(f"모델 저장 오류: {e}")

    # MLflow에 모델 등록
    mlflow.keras.log_model(model, artifact_path="model")

    print(f"\n{'=' * 60}")
    print("학습 완료!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"{'=' * 60}\n")