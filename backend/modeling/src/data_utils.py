# modeling/src/data_utils.py
import librosa
import numpy as np
import os

# 멜 스펙트로그램 생성을 위한 설정값
SR = 44100
N_MELS = 128  # 스펙트로그램의 세로 해상도 (주파수 축)
N_FFT = 2048
HOP_LENGTH = 512


def audio_to_melspectrogram(filepath, target_shape=(N_MELS, N_MELS)):
    """오디오 파일을 불러와 고정된 크기의 멜 스펙트로그램으로 변환합니다."""
    try:
        y, sr = librosa.load(filepath, sr=SR)

        # 1초 미만의 짧은 오디오는 패딩 처리
        if len(y) < SR:
            y = np.pad(y, (0, SR - len(y)))
        else:
            y = y[:SR]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 이미지 크기를 (128, 128) 등으로 고정
        if mel_spec_db.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :target_shape[1]]

        return mel_spec_db
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
                       if f.endswith(('.wav', '.mp3', '.flac', 'wav'))]

        print(f"📁 {folder_name:20s} → {label_vector} ({len(audio_files)}개 파일)")

        for filename in audio_files:
            filepath = os.path.join(class_path, filename)
            spec = audio_to_melspectrogram(filepath)
            if spec is not None:
                X.append(spec)
                y.append(label_vector)

    X = np.array(X)[..., np.newaxis]
    y = np.array(y, dtype=np.float32)  # 멀티라벨은 float32

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