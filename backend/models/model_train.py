import os, glob, numpy as np, librosa, joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

SR = 44100
N_MELS = 64
N_FFT = 1024
HOP = 256

def feature_vector_from_wav(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    # (1) log-mel
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    Mdb = librosa.power_to_db(M, ref=np.max)
    m_mean = Mdb.mean(axis=1)              # (64,)
    m_std  = Mdb.std(axis=1)               # (64,)

    # (2) 몇 가지 간단한 스펙트럴 특성
    sc   = librosa.feature.spectral_centroid(y=y, sr=sr).mean()      # 1
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()  # 1
    zcr  = librosa.feature.zero_crossing_rate(y).mean()              # 1
    rms  = librosa.feature.rms(y=y).mean()                           # 1

    # (3) 델타-멜(시간변화)도 살짝
    dM   = librosa.feature.delta(Mdb)
    d_mean = dM.mean(axis=1)                # (64,)

    feat = np.concatenate([m_mean, m_std, d_mean, [sc, roll, zcr, rms]]).astype(np.float32)
    return feat  # 총 64+64+64+4 = 196차원

def load_data(data_root="data"):
    X, y = [], []
    for lab, lbl_id in [("kick",0), ("snare",1)]:
        for p in glob.glob(os.path.join(data_root, lab, "*")):
            try:
                X.append(feature_vector_from_wav(p))
                y.append(lbl_id)
            except Exception as e:
                print(f"[skip] {p} ({e})")
    X = np.stack(X); y = np.array(y)
    return X, y

X, y = load_data("data")
print("샘플 수:", len(y), " / 킥:", (y==0).sum(), " 스네어:", (y==1).sum())
assert len(y) >= 20, "샘플이 너무 적어요. 최소 각 클래스 20개 권장."

# 파이프라인: 표준화 → 로지스틱 회귀(클래스 불균형 대비)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
])

# 5-fold Stratified CV로 대략 성능 가늠
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
print(f"CV Acc: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

# 최종 학습(전체 데이터로 피팅 → 실전 예측용)
pipe.fit(X, y)
joblib.dump(pipe, "kick_snare_lr.pkl")
print("모델 저장: kick_snare_lr.pkl")

# 훈련셋 내 분류 리포트(샘플이 적어서 참고용)
pred = pipe.predict(X)
print(classification_report(y, pred, target_names=["kick","snare"]))
print("Confusion Matrix:\n", confusion_matrix(y, pred))
