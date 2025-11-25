"""
드럼 타격 검출 데이터 로더 (피드백 반영 최종 버전)

주요 개선사항:
1. ✅ ±1 프레임 확장 라벨 (onset spread)
2. ✅ SpecAugment 추가 (time_mask_param 축소: 30→10)
3. ✅ Silent sample augmentation (10%→15%)
4. ✅ Mel normalization 추가 (mean/std)
5. ✅ 클래스별 threshold 탐색 지원
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import pretty_midi
from pathlib import Path
from typing import Tuple


# ============================================
# MIDI 드럼 노트 매핑
# ============================================
DRUM_MAPPING = {
    'kick': [36],
    'snare': [38, 40],
    'hihat': [
        42, 44, 46,     # 하이햇 기본 3종
        49, 57,         # 크래쉬 1, 크래쉬 2
        51, 59,         # 라이드 1, 라이드 2
        52, 55,         # 차이나, 스플래시
        53              # 라이드 벨
    ]
}


# ============================================
# 시퀀스 길이 제한 설정
# ============================================
MAX_SEQ_LEN = 2500


class DrumDatasetConfig:
    """데이터셋 설정 (피드백 반영 최종 버전)"""
    def __init__(self):
        # 경로 설정
        self.csv_path = r"D:\model_test\e-gmd-v1.0.0\e-gmd-v1.0.0.csv"
        self.data_root = r"D:\model_test\e-gmd-v1.0.0"

        # 사전 계산 데이터 사용 여부
        self.use_precomputed = True
        self.precomputed_root = "./precomputed_bigru_data_hop256_final"

        # 필터링 설정
        self.exclude_styles = ['jazz']
        self.min_duration = 10.0

        # 오디오 전처리 설정
        self.sample_rate = 22050
        self.n_fft = 2048
        self.hop_length = 256
        self.n_mels = 128
        self.fmin = 20
        self.fmax = 8000

        # 데이터 증강 설정
        self.use_augmentation = True
        self.augment_prob = 0.5

        # SpecAugment 설정 (개선: time_mask_param 축소)
        self.use_spec_augment = True
        self.freq_mask_param = 12  # 15 → 12
        self.time_mask_param = 10  # 30 → 10 (onset 보존)
        self.n_freq_masks = 2
        self.n_time_masks = 2

        # Silent sample 증강 (개선: 10% → 15%)
        self.add_silent_prob = 0.15  # 0.1 → 0.15

        # Mel normalization 추가
        self.use_mel_normalization = True

        # 레이블 설정
        self.drum_types = ['kick', 'snare', 'hihat']
        self.n_classes = len(self.drum_types)

        # ±1 프레임 확장 설정
        self.label_spread_frames = 1

        # 시간 해상도
        self.frame_duration = self.hop_length / self.sample_rate

        # 시퀀스 길이 제한
        self.max_seq_len = MAX_SEQ_LEN


def load_metadata(config: DrumDatasetConfig) -> pd.DataFrame:
    """CSV 메타데이터 로딩 및 필터링"""
    df = pd.read_csv(config.csv_path)
    df = df[~df['style'].str.contains('jazz', case=False, na=False)]
    df = df[df['duration'] > config.min_duration]
    return df


def augment_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    간단한 오디오 증강
    ※ Gain만 적용 (onset alignment 보존)
    """
    # Gain adjustment
    if np.random.rand() < 0.5:
        gain_db = np.random.uniform(-6, 6)
        y = y * (10 ** (gain_db / 20))

    # Clipping 방지
    y = np.clip(y, -1.0, 1.0)

    return y


def spec_augment(mel_spec: np.ndarray, config: DrumDatasetConfig) -> np.ndarray:
    """
    SpecAugment 적용: Frequency/Time Masking

    개선: time_mask_param을 줄여서 onset 보존
    """
    mel_spec = mel_spec.copy()
    T, F = mel_spec.shape

    # Frequency Masking
    for _ in range(config.n_freq_masks):
        f = np.random.randint(0, min(config.freq_mask_param, F))
        if f > 0:
            f0 = np.random.randint(0, F - f)
            mel_spec[:, f0:f0+f] = 0

    # Time Masking (onset 보존을 위해 축소)
    for _ in range(config.n_time_masks):
        t = np.random.randint(0, min(config.time_mask_param, T))
        if t > 0:
            t0 = np.random.randint(0, max(1, T - t))
            mel_spec[t0:t0+t, :] = 0

    return mel_spec


def normalize_mel(mel_spec: np.ndarray) -> np.ndarray:
    """
    멜스펙트로그램 정규화 (mean/std)

    개선: 녹음 세기에 따른 dynamic range 변동 완화
    """
    mean = np.mean(mel_spec)
    std = np.std(mel_spec)

    if std > 1e-5:
        mel_spec = (mel_spec - mean) / std

    return mel_spec


def wav_to_melspectrogram(wav_path: str, config: DrumDatasetConfig, augment: bool = False) -> np.ndarray:
    """WAV 파일을 멜스펙트로그램으로 변환"""
    y, sr = librosa.load(wav_path, sr=config.sample_rate, mono=True)

    # 데이터 증강 적용 (training 시에만)
    if augment and config.use_augmentation and np.random.rand() < config.augment_prob:
        y = augment_audio(y, sr)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.n_fft, hop_length=config.hop_length,
        n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = mel_spec_db.T

    # 정규화 적용 (개선)
    if config.use_mel_normalization:
        mel_spec_db = normalize_mel(mel_spec_db)

    return mel_spec_db


def midi_to_labels(midi_path: str, n_frames: int, config: DrumDatasetConfig) -> np.ndarray:
    """
    MIDI 파일을 프레임별 멀티레이블로 변환 (±1 프레임 확장)
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    labels = np.zeros((n_frames, config.n_classes), dtype=np.float32)

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                onset_time = note.start
                frame_idx = int(onset_time / config.frame_duration)

                if 0 <= frame_idx < n_frames:
                    for drum_idx, (drum_type, note_numbers) in enumerate(DRUM_MAPPING.items()):
                        if note.pitch in note_numbers:
                            # ±1 프레임 확장
                            for offset in range(-config.label_spread_frames,
                                              config.label_spread_frames + 1):
                                target_frame = frame_idx + offset
                                if 0 <= target_frame < n_frames:
                                    labels[target_frame, drum_idx] = 1.0
                            break

    return labels


# ============================================
# 일반 Dataset (실시간 변환)
# ============================================
class DrumDataset(Dataset):
    """PyTorch Dataset for Drum Onset Detection (피드백 반영 최종 버전)"""

    def __init__(self, metadata: pd.DataFrame, config: DrumDatasetConfig, split: str = 'train'):
        self.config = config
        self.split = split
        self.data = metadata[metadata['split'] == split].reset_index(drop=True)
        self.is_training = (split == 'train')

    def __len__(self):
        # Silent sample 추가 고려 (15%로 증가)
        base_len = len(self.data)
        if self.is_training and self.config.add_silent_prob > 0:
            silent_samples = int(base_len * self.config.add_silent_prob / (1 - self.config.add_silent_prob))
            return base_len + silent_samples
        return base_len

    def __getitem__(self, idx):
        base_len = len(self.data)

        # Silent sample 생성 (training 시에만)
        if self.is_training and idx >= base_len:
            # 무음 샘플 생성
            seq_len = np.random.randint(500, 1500)
            mel_spec = torch.zeros(seq_len, self.config.n_mels, dtype=torch.float32)
            labels = torch.zeros(seq_len, self.config.n_classes, dtype=torch.float32)
            return mel_spec, labels

        # 일반 데이터 로딩
        row = self.data.iloc[idx % base_len]
        wav_filename = row['audio_filename']
        midi_filename = row['midi_filename']

        wav_path = os.path.join(self.config.data_root, wav_filename)
        midi_path = os.path.join(self.config.data_root, midi_filename)

        # WAV → Mel 변환 (training 시 augmentation 적용)
        mel_spec = wav_to_melspectrogram(wav_path, self.config, augment=self.is_training)

        # SpecAugment 적용 (training 시에만)
        if self.is_training and self.config.use_spec_augment and np.random.rand() < 0.5:
            mel_spec = spec_augment(mel_spec, self.config)

        n_frames = mel_spec.shape[0]
        labels = midi_to_labels(midi_path, n_frames, self.config)

        mel_spec = torch.FloatTensor(mel_spec)
        labels = torch.FloatTensor(labels)

        return mel_spec, labels


# ============================================
# 사전 계산 Dataset
# ============================================
class PrecomputedDrumDataset(Dataset):
    """사전 계산된 .npy 파일을 로딩하는 초고속 Dataset (피드백 반영 최종 버전)"""

    def __init__(self, metadata: pd.DataFrame, config: DrumDatasetConfig, split: str = 'train'):
        self.config = config
        self.split = split
        self.root = config.precomputed_root
        self.data = metadata[metadata['split'] == split].reset_index(drop=True)
        self.is_training = (split == 'train')

        # 파일 경로 미리 구성
        self.mel_paths = []
        self.label_paths = []

        for _, row in self.data.iterrows():
            fname = row['audio_filename']
            fname = fname.replace('/', '_').replace('\\', '_').replace('.wav', '')

            mel_path = os.path.join(self.root, split, 'mel', f"{fname}.npy")
            label_path = os.path.join(self.root, split, 'label', f"{fname}.npy")

            self.mel_paths.append(mel_path)
            self.label_paths.append(label_path)

    def __len__(self):
        base_len = len(self.data)
        if self.is_training and self.config.add_silent_prob > 0:
            silent_samples = int(base_len * self.config.add_silent_prob / (1 - self.config.add_silent_prob))
            return base_len + silent_samples
        return base_len

    def __getitem__(self, idx):
        base_len = len(self.data)

        # Silent sample 생성
        if self.is_training and idx >= base_len:
            seq_len = np.random.randint(500, 1500)
            mel_spec = torch.zeros(seq_len, self.config.n_mels, dtype=torch.float32)
            labels = torch.zeros(seq_len, self.config.n_classes, dtype=torch.float32)
            return mel_spec, labels

        # 일반 데이터 로딩
        mel_spec = np.load(self.mel_paths[idx % base_len])
        labels = np.load(self.label_paths[idx % base_len])

        # SpecAugment 적용 (training 시에만)
        if self.is_training and self.config.use_spec_augment and np.random.rand() < 0.5:
            mel_spec = spec_augment(mel_spec, self.config)

        mel_spec = torch.FloatTensor(mel_spec)
        labels = torch.FloatTensor(labels)

        return mel_spec, labels


# ============================================
# Collate Function (메모리 최적화)
# ============================================
def collate_fn_train(batch):
    """Training용 collate function - 랜덤 크롭"""
    clipped_batch = []

    for mel, label in batch:
        seq_len = mel.shape[0]

        if seq_len > MAX_SEQ_LEN:
            start = np.random.randint(0, seq_len - MAX_SEQ_LEN + 1)
            mel = mel[start:start + MAX_SEQ_LEN]
            label = label[start:start + MAX_SEQ_LEN]

        clipped_batch.append((mel, label))

    # 배치 패딩
    lengths = [mel.shape[0] for mel, _ in clipped_batch]
    max_len = max(lengths)

    batch_size = len(clipped_batch)
    n_mels = clipped_batch[0][0].shape[1]
    n_classes = clipped_batch[0][1].shape[1]

    mel_specs = torch.zeros(batch_size, max_len, n_mels)
    label_batch = torch.zeros(batch_size, max_len, n_classes)

    for i, (mel, label) in enumerate(clipped_batch):
        length = mel.shape[0]
        mel_specs[i, :length, :] = mel
        label_batch[i, :length, :] = label

    lengths = torch.LongTensor(lengths)

    return mel_specs, label_batch, lengths


def collate_fn_eval(batch):
    """Validation/Test용 collate function - deterministic"""
    clipped_batch = []

    for mel, label in batch:
        seq_len = mel.shape[0]

        if seq_len > MAX_SEQ_LEN:
            mel = mel[:MAX_SEQ_LEN]
            label = label[:MAX_SEQ_LEN]

        clipped_batch.append((mel, label))

    # 배치 패딩
    lengths = [mel.shape[0] for mel, _ in clipped_batch]
    max_len = max(lengths)

    batch_size = len(clipped_batch)
    n_mels = clipped_batch[0][0].shape[1]
    n_classes = clipped_batch[0][1].shape[1]

    mel_specs = torch.zeros(batch_size, max_len, n_mels)
    label_batch = torch.zeros(batch_size, max_len, n_classes)

    for i, (mel, label) in enumerate(clipped_batch):
        length = mel.shape[0]
        mel_specs[i, :length, :] = mel
        label_batch[i, :length, :] = label

    lengths = torch.LongTensor(lengths)

    return mel_specs, label_batch, lengths


# ============================================
# DataLoader 생성 함수들
# ============================================
def get_precomputed_dataloaders(
    config: DrumDatasetConfig,
    batch_size: int = 16,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """사전 계산된 데이터를 사용하는 초고속 DataLoader 생성"""
    print("=" * 80)
    print("🚀 초고속 DataLoader 생성 (사전 계산 데이터 사용)")
    print("=" * 80)
    print(f"사전 계산 데이터 경로: {config.precomputed_root}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Max sequence length: {MAX_SEQ_LEN} frames (~{MAX_SEQ_LEN * config.frame_duration:.1f}초)")
    print(f"Frame duration: {config.frame_duration*1000:.1f}ms (hop_length={config.hop_length})")
    print(f"Label spread: ±{config.label_spread_frames} frames")
    print(f"SpecAugment: {config.use_spec_augment} (time_mask={config.time_mask_param})")
    print(f"Silent sample prob: {config.add_silent_prob}")
    print(f"Mel normalization: {config.use_mel_normalization}")
    print("=" * 80)

    if not os.path.exists(config.precomputed_root):
        raise FileNotFoundError(
            f"❌ 사전 계산 데이터가 없습니다: {config.precomputed_root}\n"
            f"먼저 npy_maker_final.py를 실행하세요!"
        )

    metadata = load_metadata(config)

    train_dataset = PrecomputedDrumDataset(metadata, config, split='train')
    val_dataset = PrecomputedDrumDataset(metadata, config, split='validation')
    test_dataset = PrecomputedDrumDataset(metadata, config, split='test')

    print(f"\n[TRAIN] 데이터셋 초기화: {len(train_dataset)} 샘플 (silent 포함)")
    print(f"[VAL] 데이터셋 초기화: {len(val_dataset)} 파일")
    print(f"[TEST] 데이터셋 초기화: {len(test_dataset)} 파일")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_train,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_eval,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_eval,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    print(f"\n✅ 초고속 DataLoader 생성 완료")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def get_normal_dataloaders(
    config: DrumDatasetConfig,
    batch_size: int = 16,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """일반 DataLoader 생성 (실시간 WAV → Mel 변환)"""
    print("=" * 80)
    print("📂 일반 DataLoader 생성 (실시간 변환)")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Max sequence length: {MAX_SEQ_LEN} frames (~{MAX_SEQ_LEN * config.frame_duration:.1f}초)")
    print(f"Frame duration: {config.frame_duration*1000:.1f}ms (hop_length={config.hop_length})")
    print(f"Data augmentation: {config.use_augmentation}")
    print(f"Label spread: ±{config.label_spread_frames} frames")
    print(f"SpecAugment: {config.use_spec_augment} (time_mask={config.time_mask_param})")
    print(f"Silent sample prob: {config.add_silent_prob}")
    print(f"Mel normalization: {config.use_mel_normalization}")
    print("=" * 80)

    metadata = load_metadata(config)

    train_dataset = DrumDataset(metadata, config, split='train')
    val_dataset = DrumDataset(metadata, config, split='validation')
    test_dataset = DrumDataset(metadata, config, split='test')

    print(f"\n[TRAIN] 데이터셋 초기화: {len(train_dataset)} 샘플 (silent 포함)")
    print(f"[VAL] 데이터셋 초기화: {len(val_dataset)} 파일")
    print(f"[TEST] 데이터셋 초기화: {len(test_dataset)} 파일")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_train,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_eval,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_eval,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    print(f"\n✅ 일반 DataLoader 생성 완료")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ============================================
# 통합 DataLoader 생성 함수
# ============================================
def get_dataloaders(
    config: DrumDatasetConfig,
    batch_size: int = 16,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """통합 DataLoader 생성 함수"""
    if config.use_precomputed:
        return get_precomputed_dataloaders(config, batch_size, num_workers)
    else:
        return get_normal_dataloaders(config, batch_size, num_workers)


if __name__ == "__main__":
    """데이터 로딩 테스트"""
    import time

    config = DrumDatasetConfig()

    print("\n" + "=" * 80)
    print("🧪 데이터 로딩 모드 테스트 (피드백 반영 최종 버전)")
    print("=" * 80)

    config.use_precomputed = False

    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            config, batch_size=4, num_workers=2
        )

        print("\n⏱️  속도 테스트 (5 배치)...")
        start = time.time()

        for i, (mel_specs, labels, lengths) in enumerate(train_loader):
            if i >= 5:
                break
            print(f"  Batch {i+1}: mel={mel_specs.shape}, label={labels.shape}, max_len={lengths.max()}")
            print(f"    Label density: {labels.sum(dim=(0,1)) / lengths.sum()}")
            print(f"    Mel stats: mean={mel_specs.mean():.3f}, std={mel_specs.std():.3f}")

        elapsed = time.time() - start
        print(f"\n✅ 5 배치 로딩 시간: {elapsed:.2f}초")
        print(f"   배치당 평균: {elapsed/5:.3f}초")
    except Exception as e:
        print(f"❌ 오류: {e}")