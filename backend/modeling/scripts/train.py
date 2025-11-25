# backend/modeling/scripts/train.py
import os
import sys
from pathlib import Path

import numpy as np
import mlflow
import mlflow.keras

from sklearn.model_selection import train_test_split, GroupShuffleSplit
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
        mdict = compute_multilabel_metrics(self.y_val, prob, threshold=self.threshold)
        for name, value in mdict.items():
            mlflow.log_metric(f"val_{name}", float(value), step=epoch)


# ---------- 헬퍼: 그룹 보존 3-way 분할 ----------
def group_three_way_split(X, y, groups, *, random_state=42):
    """
    groups가 주어지면 원본 파일/곡 단위로 겹치지 않게
    70% train / 15% val / 15% test로 분할한다.
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(X, y, groups=groups))

    X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
    X_temp,  y_temp,  g_temp  = X[temp_idx],  y[temp_idx],  groups[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state + 1)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=g_temp))

    X_val, y_val = X_temp[val_idx],  y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]
    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------------------------------

# --- 1. 데이터 로드 ---
print(f"\n{'=' * 60}")
print(f"멀티라벨 드럼 분류 모델 학습")
print(f"{'=' * 60}\n")

# (권장) load_processed_data가 (X, y, groups, paths) 형태를 지원하도록 수정
# 호환을 위해 try/except로 groups 유무에 대응
_loaded = load_processed_data(DATA_PATH)
if isinstance(_loaded, (list, tuple)) and len(_loaded) >= 3:
    X, y, groups = _loaded[:3]
else:
    X, y = _loaded
    groups = None

# 데이터 검증
assert len(y.shape) == 2, f"y는 2D 배열이어야 합니다. 현재: {y.shape}"
assert y.shape[1] == NUM_CLASSES, f"라벨 수 불일치. 예상: {NUM_CLASSES}, 실제: {y.shape[1]}"

# === 2. 분할: groups가 있으면 그룹 보존, 없으면 일반 3-way ===
if groups is not None:
    X_train, y_train, X_val, y_val, X_test, y_test = group_three_way_split(X, y, groups, random_state=42)
else:
    # 70% train / 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=None
    )
    # 15% val / 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=43, stratify=None
    )

# (선택) 누수 없는 정규화 예시: 반드시 train으로만 평균/표준편차 계산 후 나머지에 적용
# 이미 data_utils에서 정규화를 끝냈다면 아래 블록은 비활성화하세요.
DO_NORMALIZE = False
if DO_NORMALIZE:
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

label_totals       = y.sum(axis=0).astype(int)
train_label_totals = y_train.sum(axis=0).astype(int)
val_label_totals   = y_val.sum(axis=0).astype(int)
test_label_totals  = y_test.sum(axis=0).astype(int)

print(f"\n데이터 분할:")
print(f"  훈련:  {X_train.shape}, {y_train.shape}")
print(f"  검증:  {X_val.shape}, {y_val.shape}")
print(f"  테스트:{X_test.shape}, {y_test.shape}")

# 라벨 분포 확인
print(f"\n라벨 분포:")
print(f"  전체 - Kick: {label_totals[0]}, Snare: {label_totals[1]}, Hihat: {label_totals[2]}")
print(f"  훈련 - Kick: {train_label_totals[0]}, Snare: {train_label_totals[1]}, Hihat: {train_label_totals[2]}")
print(f"  검증 - Kick: {val_label_totals[0]}, Snare: {val_label_totals[1]}, Hihat: {val_label_totals[2]}")
print(f"  테스트 - Kick: {test_label_totals[0]}, Snare: {test_label_totals[1]}, Hihat: {test_label_totals[2]}")

# --- 3. MLflow 실행 시작 ---
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
    mlflow.log_param("data_path", DATA_PATH)

    # 데이터셋 요약 기록
    print("\n데이터셋 요약 정보를 MLflow에 기록 중...")
    dataset_summary = {
        "num_samples": int(X.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "label_distribution": {
            "all":    {"kick": int(label_totals[0]), "snare": int(label_totals[1]), "hihat": int(label_totals[2])},
            "train":  {"kick": int(train_label_totals[0]), "snare": int(train_label_totals[1]), "hihat": int(train_label_totals[2])},
            "val":    {"kick": int(val_label_totals[0]), "snare": int(val_label_totals[1]), "hihat": int(val_label_totals[2])},
            "test":   {"kick": int(test_label_totals[0]), "snare": int(test_label_totals[1]), "hihat": int(test_label_totals[2])},
        },
    }
    mlflow.log_dict(dataset_summary, artifact_file="dataset_summary.json")

    # --- 4. 모델 생성 ---
    print("\n모델 생성 중...")
    model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    print("\n모델 구조:")
    model.summary(print_fn=lambda x: print(x))

    # --- 5. 모델 학습 ---
    print(f"\n{'=' * 60}")
    print("모델 학습 시작")
    print(f"{'-' * 60}\n")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),  # ✅ 검증셋만 사용
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            MultilabelMetricsCallback(X_val, y_val, threshold=threshold)  # ✅ 검증셋 메트릭만 로깅
        ],
        verbose=1
    )



    # --- 6. 최종 평가 (테스트셋 단 한 번) ---
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
    print(f"  F1 Score (micro):   {metrics_dict['f1_micro']:.4f}")
    print(f"  F1 Score (macro):   {metrics_dict['f1_macro']:.4f}")
    print(f"  F1 Score (samples): {metrics_dict['f1_samples']:.4f}")
    print(f"  Hamming Loss:       {metrics_dict['hamming_loss']:.4f}")
    print(f"  Jaccard Score:      {metrics_dict['jaccard']:.4f}")

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

    # --- 7. 모델 저장 ---
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    final_model_path = os.path.join(OUTPUT_MODEL_DIR, OUTPUT_MODEL_NAME)
    model.save(final_model_path)
    print(f"\n모델 저장 완료: {final_model_path}")

    # MLflow에 모델 아티팩트 로깅
    mlflow.log_artifact(final_model_path)
