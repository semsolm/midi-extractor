# backend/app/services/midi_maker.py

"""
드럼 WAV 파일 → MIDI 악보 변환 파이프라인 (통합 버전)
BiGRU_model.py의 내용을 포함하여 단일 파일로 동작하도록 수정됨.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn  # PyTorch 신경망 모듈 임포트
import librosa
import pretty_midi
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# [통합됨] DrumOnsetDetector 클래스 (구 BiGRU_model.py 내용)
# =============================================================================
class DrumOnsetDetector(nn.Module):
    """
    드럼 타격 검출 모델 (개선 버전)
    CNN + BiGRU 구조를 사용한 멀티레이블 분류 모델
    """

    def __init__(
            self,
            n_mels=128,
            n_classes=3,
            cnn_channels=[32, 64, 128],
            gru_hidden=384,
            gru_layers=2,
            dropout=0.3
    ):
        super(DrumOnsetDetector, self).__init__()

        self.n_mels = n_mels
        self.n_classes = n_classes

        # 1. CNN 레이어: Conv-Conv-Pool 패턴
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, cnn_channels[0], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[0], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(dropout * 0.5),

            # Block 2
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[1], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(dropout * 0.5),

            # Block 3
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[2], cnn_channels[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(dropout * 0.5),
        )

        self.freq_bins_after_cnn = n_mels // (2 ** 3)  # 128 // 8 = 16
        self.cnn_output_dim = cnn_channels[2] * self.freq_bins_after_cnn

        # 2. BiGRU 레이어
        self.gru = nn.GRU(
            input_size=self.cnn_output_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        self.gru_output_dim = gru_hidden * 2

        # 3. Fully Connected 레이어
        self.fc = nn.Sequential(
            nn.Linear(self.gru_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        batch_size, seq_len, n_mels = x.shape
        x = x.unsqueeze(1)
        x = self.cnn(x)
        batch_size, channels, seq_len, freq_bins = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, seq_len, -1)
        x, _ = self.gru(x)
        logits = self.fc(x)
        return logits


# =============================================================================
# 설정 및 메인 로직 (MiDi_maker.py 내용)
# =============================================================================

class InferenceConfig:
    """추론 및 MIDI 변환 설정"""

    def __init__(self):
        self.sample_rate = 22050
        self.n_fft = 2048
        self.hop_length = 256
        self.n_mels = 128
        self.fmin = 20
        self.fmax = 8000
        self.use_mel_normalization = True
        self.frame_duration = self.hop_length / self.sample_rate

        # 모델 파라미터
        self.n_classes = 3
        self.cnn_channels = [32, 64, 128]
        self.gru_hidden = 384
        self.gru_layers = 2
        self.dropout = 0.3

        # Sliding Window
        self.window_size = 2000
        self.hop_size = 1000

        self.thresholds = {
            'kick': 0.50,
            'snare': 0.50,
            'hihat': 0.15
        }
        self.bpm_start_range = 60
        self.bpm_end_range = 200

        # 양자화 및 후처리
        self.grid_division = {'kick': 16, 'snare': 16, 'hihat': 8}
        self.default_grid_division = 16
        self.merge_window_ms = 50
        self.min_gap_ms = {'kick': 80, 'snare': 60, 'hihat': 30}
        self.quantize_bias = {'kick': 0.3, 'snare': 0.3, 'hihat': 0.25}
        self.simultaneous_window_ms = 30

        # MIDI 매핑
        self.midi_mapping = {'kick': 36, 'snare': 38, 'hihat': 42}
        self.velocity = 100
        self.note_duration = 0.1

        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_grid_interval(self, bpm, drum_type=None):
        beat_duration = 60.0 / bpm
        division = self.grid_division.get(drum_type,
                                          self.default_grid_division) if drum_type else self.default_grid_division
        return beat_duration / (division / 4)

    def get_quantize_bias(self, drum_type):
        return self.quantize_bias.get(drum_type, 0.3)


def normalize_mel(mel_spec):
    mean = np.mean(mel_spec)
    std = np.std(mel_spec)
    if std > 1e-5: mel_spec = (mel_spec - mean) / std
    return mel_spec


def load_and_preprocess_audio(wav_path, config):
    y, sr = librosa.load(wav_path, sr=config.sample_rate, mono=True)
    duration = len(y) / sr
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.n_fft, hop_length=config.hop_length,
        n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T
    if config.use_mel_normalization:
        mel_spec_db = normalize_mel(mel_spec_db)
    return mel_spec_db, duration


def detect_bpm(wav_path, config):
    y, sr = librosa.load(wav_path, sr=config.sample_rate, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120.0)
    bpm = float(tempo)
    if bpm < config.bpm_start_range:
        bpm *= 2
    elif bpm > config.bpm_end_range:
        bpm /= 2
    return np.clip(bpm, config.bpm_start_range, config.bpm_end_range)


def predict_drum_onsets(mel_spec, model, config):
    model.eval()
    n_frames = mel_spec.shape[0]
    predictions = np.zeros((n_frames, config.n_classes), dtype=np.float32)
    count_map = np.zeros(n_frames, dtype=np.float32)

    with torch.no_grad():
        start = 0
        while start < n_frames:
            end = min(start + config.window_size, n_frames)
            window = mel_spec[start:end]
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(config.device)
            logits = model(window_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            predictions[start:end] += probs[:end - start]
            count_map[start:end] += 1
            start += config.hop_size

    count_map[count_map == 0] = 1
    return predictions / count_map[:, np.newaxis]


def detect_peaks(predictions, config, frame_times):
    drum_types = ['kick', 'snare', 'hihat']
    onsets = {dt: [] for dt in drum_types}
    for i, dt in enumerate(drum_types):
        probs = predictions[:, i]
        threshold = config.thresholds[dt]
        indices = np.where((probs[:-1] < threshold) & (probs[1:] >= threshold))[0] + 1
        onsets[dt] = [frame_times[idx] for idx in indices]
    return onsets


# 후처리 함수들
def merge_nearby_events(onsets, config):
    merge_window = config.merge_window_ms / 1000.0
    merged_onsets = {}
    for dt, times in onsets.items():
        if not times:
            merged_onsets[dt] = []
            continue
        times = sorted(times)
        merged = [times[0]]
        for t in times[1:]:
            if t - merged[-1] <= merge_window:
                merged[-1] = merged[-1] * 0.7 + t * 0.3
            else:
                merged.append(t)
        merged_onsets[dt] = merged
    return merged_onsets


def enforce_minimum_gap(onsets, config):
    filtered_onsets = {}
    for dt, times in onsets.items():
        if not times:
            filtered_onsets[dt] = []
            continue
        min_gap = config.min_gap_ms[dt] / 1000.0
        times = sorted(times)
        filtered = [times[0]]
        for t in times[1:]:
            if t - filtered[-1] >= min_gap:
                filtered.append(t)
        filtered_onsets[dt] = filtered
    return filtered_onsets

def estimate_bpm_from_onsets(onsets, config, fallback_bpm):
    """
    드럼 모델이 뽑은 킥/스네어 onsets(초 단위)만으로 BPM을 다시 추정하는 함수

    Args:
        onsets: {'kick': [t1, t2, ...], 'snare': [...], 'hihat': [...]} 형태 (초 단위)
        config: InferenceConfig 인스턴스
        fallback_bpm: librosa로 먼저 추정한 BPM (문제 생기면 이 값으로 복귀)

    Returns:
        refined_bpm: onsets 기반으로 다시 계산한 BPM
    """

    # 1) 킥 + 스네어 타격 시점만 사용 (박자 정보가 가장 강함)
    times = []
    for dt in ['kick', 'snare']:
        if dt in onsets and onsets[dt]:
            times.extend(onsets[dt])

    # 타격이 너무 적으면 그냥 원래 BPM 사용
    if len(times) < 2:
        return float(fallback_bpm)

    times = sorted(times)

    # 2) 연속 타격 간격(IOI: Inter-Onset Interval) 계산
    intervals = np.diff(times)  # 초 단위 간격 배열

    # 3) 너무 짧거나(노이즈) 너무 긴(몇 마디 건너뛴) 간격 제거
    #    - 최소 간격: 설정된 최대 BPM에서 나올 수 있는 최소 beat 간격
    #    - 최대 간격: 설정된 최소 BPM에서 나올 수 있는 beat 간격의 4배 정도까지 허용
    min_ioi = 60.0 / config.bpm_end_range          # 예: 200BPM이면 최소 0.3초
    max_ioi = (60.0 / config.bpm_start_range) * 4  # 예: 60BPM 기준 1beat=1초 → 4beat=4초

    intervals = intervals[(intervals >= min_ioi) & (intervals <= max_ioi)]

    if len(intervals) == 0:
        # 필터링하고 나니 남는 게 없으면 원래 BPM 사용
        return float(fallback_bpm)

    # 4) 중간값(median)으로 대표 간격 선택 (outlier에 덜 민감)
    median_ioi = float(np.median(intervals))
    if median_ioi <= 0:
        return float(fallback_bpm)

    bpm = 60.0 / median_ioi  # 간격(초) → BPM

    # 5) 설정된 BPM 범위 안으로 2배/2분의1 스케일 조정
    #    (절반 템포 / 두배 템포 문제를 완화)
    while bpm < config.bpm_start_range:
        bpm *= 2.0
    while bpm > config.bpm_end_range:
        bpm /= 2.0

    bpm = float(np.clip(bpm, config.bpm_start_range, config.bpm_end_range))

    return bpm



def quantize_to_grid(onsets, bpm, config):
    quantized_onsets = {}
    for dt, times in onsets.items():
        if not times:
            quantized_onsets[dt] = []
            continue
        grid_interval = config.get_grid_interval(bpm, dt)
        bias = config.get_quantize_bias(dt)
        quantized = []
        for t in times:
            grid_index = math.floor((t / grid_interval) + bias)
            quantized.append(grid_index * grid_interval)
        quantized_onsets[dt] = sorted(list(set(quantized)))
    return quantized_onsets


def group_simultaneous_hits(onsets, config):
    all_events = []
    for dt, times in onsets.items():
        for t in times: all_events.append((t, dt))
    all_events.sort(key=lambda x: x[0])
    if not all_events: return []

    window = config.simultaneous_window_ms / 1000.0
    grouped = []
    curr_t, curr_d = all_events[0]
    curr_drums = [curr_d]

    for t, d in all_events[1:]:
        if t - curr_t <= window:
            if d not in curr_drums: curr_drums.append(d)
        else:
            grouped.append((curr_t, sorted(curr_drums)))
            curr_t, curr_d = t, d
            curr_drums = [d]
    grouped.append((curr_t, sorted(curr_drums)))
    return grouped


def create_midi_file(grouped_events, bpm, config, output_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')
    for t, drums in grouped_events:
        for d in drums:
            note = pretty_midi.Note(
                velocity=config.velocity, pitch=config.midi_mapping[d],
                start=t, end=t + config.note_duration
            )
            drum_inst.notes.append(note)
    pm.instruments.append(drum_inst)
    pm.write(output_path)


def drum_wav_to_midi(wav_path, model_path, output_dir=None, config=None, bpm_override=None):
    """
    드럼 WAV → MIDI 변환 메인 함수

    개선 포인트:
    - 1차 BPM: librosa.beat.beat_track 으로 rough estimate
    - 모델 추론으로 킥/스네어 onsets 뽑은 뒤,
      그 onsets 간격(IOI)로 BPM을 다시 추정 (estimate_bpm_from_onsets)
    - 최종 BPM으로 그리드 양자화 + MIDI 생성
    """
    if config is None:
        config = InferenceConfig()
    if output_dir is None:
        output_dir = os.path.dirname(wav_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"모델 로딩 중: {model_path}")
    # 모델 초기화 및 가중치 로드
    model = DrumOnsetDetector(
        n_mels=config.n_mels, n_classes=config.n_classes,
        cnn_channels=config.cnn_channels, gru_hidden=config.gru_hidden,
        gru_layers=config.gru_layers, dropout=config.dropout
    ).to(config.device)

    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ----------------------------------------
    # 1) 1차 BPM: librosa 기반 rough estimate
    # ----------------------------------------
    if bpm_override:
        bpm = float(bpm_override)
        print(f"[BPM] 외부에서 BPM override 지정: {bpm:.2f}")
    else:
        bpm = detect_bpm(wav_path, config)
        print(f"[BPM] 1차(librosa) BPM 추정: {bpm:.2f}")

    # ----------------------------------------
    # 2) 모델 추론 → onsets 검출 (초 단위)
    # ----------------------------------------
    mel_spec, _ = load_and_preprocess_audio(wav_path, config)
    frame_times = np.arange(mel_spec.shape[0]) * config.frame_duration
    predictions = predict_drum_onsets(mel_spec, model, config)

    onsets = detect_peaks(predictions, config, frame_times)
    onsets = merge_nearby_events(onsets, config)
    onsets = enforce_minimum_gap(onsets, config)

    # ----------------------------------------
    # 3) 킥/스네어 onsets 기반으로 BPM 다시 추정
    #    → 드럼 전용 리듬 정보만 사용해서 refined BPM 구함
    # ----------------------------------------
    refined_bpm = estimate_bpm_from_onsets(onsets, config, fallback_bpm=bpm)
    print(f"[BPM] onsets 기반 BPM 재추정: {refined_bpm:.2f}")

    bpm = refined_bpm  # 이후 양자화/악보는 이 BPM 기준으로 처리

    # ----------------------------------------
    # 4) 최종 BPM 기준 그리드 양자화 + 그룹핑
    # ----------------------------------------
    quantized_onsets = quantize_to_grid(onsets, bpm, config)
    grouped = group_simultaneous_hits(quantized_onsets, config)

    # ----------------------------------------
    # 5) MIDI 파일 생성
    # ----------------------------------------
    base_name = Path(wav_path).stem
    midi_path = os.path.join(output_dir, f"{base_name}_drums.mid")
    create_midi_file(grouped, bpm, config, midi_path)
    print(f"변환 완료: {midi_path} (BPM={bpm:.2f}, 이벤트 수={len(grouped)})")

    return midi_path, bpm, grouped



if __name__ == "__main__":
    # 테스트용 실행 코드
    pass