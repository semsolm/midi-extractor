# backend/modeling/scripts/train.py
import os
import sys
from pathlib import Path

import numpy as np
import mlflow
import mlflow.keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
from tensorflow.keras.callbacks import EarlyStopping, Callback

# --- 프로젝트 루트 및 경로 설정 ---
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

# --- 경로 모듈 추가 (src 임포트를 위해) ---
SRC_DIR = os.path.join(PROJECT_ROOT, "backend", "modeling", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_utils import load_processed_data  # noqa: E402
from model import build_cnn_model  # noqa: E402


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
        metrics_dict = compute_multilabel_metrics(self.y_val, prob, threshold=self.threshold)
        for name, value in metrics_dict.items():
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
class_counts = np.sum(y, axis=0)
print("\n라벨 분포 (train+test 합계):")
print(f"  Kick : {int(class_counts[0])}")
print(f"  Snare: {int(class_counts[1])}")
print(f"  Hihat: {int(class_counts[2])}")

# --- 2. 모델 준비 ---
model = build_cnn_model(INPUT_SHAPE, NUM_CLASSES)
model.summary(print_fn=lambda x: print(x))

# 임계치
threshold = 0.5

# --- MLflow 런 시작 ---
with mlflow.start_run(run_name="multilabel_sigmoid"):
    # 파라미터 로깅
    mlflow.log_param("input_shape", INPUT_SHAPE)
    mlflow.log_param("num_classes", NUM_CLASSES)
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("data_path", DATA_PATH)

    # --- 3. 학습 ---
    print(f"\n{'-' * 60}")
    print("모델 학습 시작")
    print(f"{'-' * 60}\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
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
    metrics_dict, y_pred = compute_multilabel_metrics(
        y_test, y_pred_prob, threshold=threshold, return_predictions=True
    )

    for name, value in metrics_dict.items():
        mlflow.log_metric(f"final_{name}", float(value))

    print(f"\n멀티라벨 메트릭:")
    print(f"  F1 Score (micro): {metrics_dict['f1_micro']:.4f}")
    print(f"  F1 Score (macro): {metrics_dict['f1_macro']:.4f}")
    print(f"  F1 Score (samples): {metrics_dict['f1_samples']:.4f}")
    print(f"  Hamming Loss: {metrics_dict['hamming_loss']:.4f}")
    print(f"  Jaccard Score: {metrics_dict['jaccard']:.4f}")

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
    model.save(final_model_path)
    print(f"\n모델 저장 완료: {final_model_path}")

    # MLflow에 모델 아티팩트 로깅
    mlflow.log_artifact(final_model_path)
