# modeling/scripts/evaluate.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# src 폴더를 파이썬 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_processed_data

# --- 설정 ---
MODEL_PATH = "../outputs/models/drum_cnn_v1.h5"
DATA_PATH = "../data/raw"  # 평가에는 raw 데이터 경로를 사용해 재현성 확보
REPORT_OUTPUT_PATH = "../outputs/reports/"
LABELS = ["kick", "snare", "hi-hat"]

# --- 1. 모델과 테스트 데이터 로드 ---
print("모델과 데이터를 로드합니다...")
model = load_model(MODEL_PATH)
X_test, y_test = load_processed_data(DATA_PATH) # 전체 데이터를 테스트용으로 사용

# --- 2. 모델 예측 수행 ---
print("모델 예측을 수행합니다...")
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# --- 3. 분류 리포트 생성 ---
print("--- 분류 리포트 ---")
report = classification_report(y_test, y_pred, target_names=LABELS)
print(report)
with open(os.path.join(REPORT_OUTPUT_PATH, "classification_report.txt"), "w") as f:
    f.write(report)

# --- 4. 혼동 행렬(Confusion Matrix) 시각화 ---
print("혼동 행렬을 생성합니다...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(REPORT_OUTPUT_PATH, "confusion_matrix.png"))
print(f"평가 결과가 '{REPORT_OUTPUT_PATH}' 폴더에 저장되었습니다.")