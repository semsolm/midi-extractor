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


def load_processed_data(data_dir):
    """전처리된 스펙트로그램 데이터를 불러오는 함수."""
    X, y = [], []
    labels = {"kick": 0, "snare": 1, "hi-hat": 2}  # 클래스 추가

    for label, num in labels.items():
        class_path = os.path.join(data_dir, label)
        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            spec = audio_to_melspectrogram(filepath)
            if spec is not None:
                X.append(spec)
                y.append(num)

    # CNN 입력 형식에 맞게 채널 차원 추가 (e.g., [샘플수, 높이, 너비, 1])
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)

    return X, y