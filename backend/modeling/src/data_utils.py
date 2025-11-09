# modeling/src/data_utils.py
import librosa
import numpy as np
import os

# ====== 상단 설정값 교체 ======
SR = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 96   # 0.6초 → 정확히 128 프레임


def audio_to_melspectrogram(filepath, target_shape=(N_MELS, N_MELS)):
    """
    오디오 → (128,128) Log-Mel(dB) → [0,1] 스케일
    - 0.6초 샘플 기준으로 128프레임이 정확히 나오도록 파라미터 고정
    - 프레임 부족 시 dB의 최솟값으로 패딩(가짜 에너지 방지)
    """
    try:
        # librosa.load: mono=True 기본, float32 반환
        y, sr = librosa.load(filepath, sr=SR, mono=True)

        # 멜스펙 계산
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
            fmin=20, fmax=sr//2
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 시간 축 길이 맞추기 (128 프레임)
        T = mel_db.shape[1]
        if T < target_shape[1]:
            pad_T = target_shape[1] - T
            pad_val = float(mel_db.min())  # dB 스케일의 최솟값으로 패딩
            mel_db = np.pad(mel_db, ((0,0),(0,pad_T)), mode='constant', constant_values=pad_val)
        elif T > target_shape[1]:
            mel_db = mel_db[:, :target_shape[1]]

        # [0,1] 정규화 (샘플 단위)
        mn, mx = float(mel_db.min()), float(mel_db.max())
        mel_01 = (mel_db - mn) / (mx - mn + 1e-6)

        return mel_01.astype(np.float32)
    except Exception as e:
        print(f"파일 처리 오류 {filepath}: {e}")
        return None


def parse_multilabel_from_folder(folder_name):
    """
    폴더명에서 멀티라벨 벡터를 생성합니다.

    Args:
        folder_name: 폴더 이름 (예: 'kick', 'kick_hihat', 'kick_snare_hihat')

    Returns:
        list: [kick, snare, hihat] 형태의 멀티라벨 벡터

    Examples:
        'kick' → [1, 0, 0]
        'snare' → [0, 1, 0]
        'hihat' → [0, 0, 1]
        'kick_hihat' → [1, 0, 1]
        'snare_hihat' → [0, 1, 1]
        'kick_snare_hihat' → [1, 1, 1]
    """
    labels = [0, 0, 0]  # [kick, snare, hihat]
    folder_lower = folder_name.lower()

    if 'kick' in folder_lower or 'bass' in  folder_lower:
        labels[0] = 1
    if 'snare' in folder_lower:
        labels[1] = 1
    if 'hihat' in folder_lower or 'hat' in folder_lower:
        labels[2] = 1

    return labels


def load_processed_data(data_dir):
    """
    멀티라벨 스펙트로그램 데이터를 불러오는 함수.

    폴더 구조:
        data_dir/
        ├── kick/           → [1, 0, 0]
        ├── snare/          → [0, 1, 0]
        ├── hihat/          → [0, 0, 1]
        ├── kick_hihat/     → [1, 0, 1]
        ├── snare_hihat/    → [0, 1, 1]
        └── kick_snare_hihat/ → [1, 1, 1]

    Args:
        data_dir: 데이터 디렉토리 경로

    Returns:
        X: 스펙트로그램 배열, shape=(샘플수, 128, 128, 1)
        y: 멀티라벨 배열, shape=(샘플수, 3)
    """
    X, y = [], []

    print(f"\n데이터 로딩 시작: {data_dir}")
    print("-" * 50)

    # 모든 하위 폴더 순회
    for folder_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(class_path):
            continue

        # 폴더명으로부터 멀티라벨 생성
        label_vector = parse_multilabel_from_folder(folder_name)

        # 파일 개수 카운트
        audio_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.wav', '.mp3', '.flac'))]

        print(f"📁 {folder_name:20s} → {label_vector} ({len(audio_files)}개 파일)")

        for filename in audio_files:
            filepath = os.path.join(class_path, filename)
            spec = audio_to_melspectrogram(filepath)
            if spec is not None:
                X.append(spec)
                y.append(label_vector)

    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (N,128,128,1)
    y = np.array(y, dtype=np.float32)

    print("-" * 50)
    print(f"✅ 데이터 로딩 완료")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   y dtype: {y.dtype}")
    print(f"\n샘플 라벨 예시 (처음 5개):")
    for i in range(min(5, len(y))):
        labels = []
        if y[i][0] == 1: labels.append("Kick")
        if y[i][1] == 1: labels.append("Snare")
        if y[i][2] == 1: labels.append("Hihat")
        print(f"   {i + 1}. {y[i]} → {', '.join(labels) if labels else 'None'}")

    return X, y