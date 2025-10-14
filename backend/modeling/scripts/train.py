# modeling/scripts/train.py
import os
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# src 폴더를 파이썬 경로에 추가하여 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_processed_data
from src.model import build_cnn_model

# --- 설정 ---
DATA_PATH = "../data/processed" # 전처리된 오디오 데이터가 있는 폴더
MODEL_OUTPUT_PATH = "../outputs/models/drum_cnn_v1.h5"
NUM_CLASSES = 3 # kick, snare, hi-hat
INPUT_SHAPE = (128, 128, 1) # (높이, 너비, 채널)

# --- 1. 데이터 로드 및 분할 ---
print("데이터 로딩 중...")
X, y = load_processed_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

# --- 2. 모델 생성 ---
print("모델 생성 중...")
model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
model.summary()

# --- 3. 모델 학습 ---
# 학습 중 가장 성능이 좋은 모델을 저장하고, 성능 개선이 없으면 조기 종료
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint(filepath=MODEL_OUTPUT_PATH, monitor='val_loss', save_best_only=True, verbose=1)
]

print("모델 학습 시작...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

print(f"모델 학습 완료! 최적 모델이 '{MODEL_OUTPUT_PATH}'에 저장되었습니다.")