# backend/app/services/midi_maker.py

"""
드럼 WAV 파일 → MIDI 악보 변환 파이프라인 (통합 버전)
BiGRU_model.py의 내용을 포함하여 단일 파일로 동작하도록 수정됨.

개선된 BPM 추정 시스템:
- Linear Regression 기반 고정밀 BPM 계산 (0.01 BPM 정밀도)
- 다중 알고리즘 앙상블 (librosa beat_track + PLP + tempogram)
- beat position 역산 방식
- IOI (Inter-Onset Interval) 분석 고도화

모델 업데이트 (과적합 방지 버전):
- GRU hidden: 384 → 256
- Dropout: 0.3 → 0.4
- LayerNorm 추가
- Spatial Dropout 적용

퀀타이즈 수정 (v2):
- quantize_bias 하향 조정 (반 마디 밀림 현상 해결)
- 첫 onset 기준 오프셋 계산 방식 개선
- Hihat grid_division 16으로 변경
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import librosa
import pretty_midi
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy import stats
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# [통합됨] SpatialDropout2d 클래스
# =============================================================================
class SpatialDropout2d(nn.Module):
    """Spatial Dropout for 2D data"""

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) > self.p).float()
        return x * mask / (1 - self.p)


# =============================================================================
# [통합됨] DrumOnsetDetector 클래스 (과적합 방지 버전)
# =============================================================================
class DrumOnsetDetector(nn.Module):
    """드럼 타격 검출 모델 (과적합 방지 버전)"""

    def __init__(
            self,
            n_mels=128,
            n_classes=3,
            cnn_channels=[32, 64, 128],
            gru_hidden=256,
            gru_layers=2,
            dropout=0.4
    ):
        super(DrumOnsetDetector, self).__init__()

        self.n_mels = n_mels
        self.n_classes = n_classes

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, cnn_channels[0], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[0], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            SpatialDropout2d(dropout * 0.5),

            # Block 2
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[1], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            SpatialDropout2d(dropout * 0.5),

            # Block 3
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[2], cnn_channels[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            SpatialDropout2d(dropout * 0.5),
        )

        self.freq_bins_after_cnn = n_mels // 8
        self.cnn_output_dim = cnn_channels[2] * self.freq_bins_after_cnn

        self.gru = nn.GRU(
            input_size=self.cnn_output_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        self.gru_output_dim = gru_hidden * 2
        self.layer_norm = nn.LayerNorm(self.gru_output_dim)

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
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
        x = self.layer_norm(x)
        logits = self.fc(x)
        return logits


# =============================================================================
# 설정 클래스 (퀀타이즈 설정 수정됨)
# =============================================================================

class InferenceConfig:
    """추론 및 MIDI 변환 설정 (퀀타이즈 수정 버전)"""

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
        self.gru_hidden = 256
        self.gru_layers = 2
        self.dropout = 0.4

        # Sliding Window
        self.window_size = 2000
        self.hop_size = 1000

        self.thresholds = {
            'kick': 0.5,
            'snare': 0.55,
            'hihat': 0.1
        }
        self.bpm_start_range = 60
        self.bpm_end_range = 200

        # ============================================
        # 양자화 설정 (수정됨 - 반 마디 밀림 해결)
        # ============================================
        # 모든 드럼 타입을 16분음표 그리드로 통일
        self.grid_division = {'kick': 16, 'snare': 16, 'hihat': 8}  # hihat: 8 → 16
        self.default_grid_division = 16

        self.merge_window_ms = 70
        self.min_gap_ms = {'kick': 90, 'snare': 70, 'hihat': 30}

        # quantize_bias 하향 조정 (핵심 수정!)
        # 기존: 0.5는 반올림이라 다음 그리드로 밀림
        # 수정: 0.3~0.4로 낮춰서 현재 그리드에 머무르게
        self.quantize_bias = {'kick': 0, 'snare': 0, 'hihat': 0}  # 기존: 0.5, 0.35, 0.28

        self.simultaneous_window_ms = 50

        # MIDI 매핑
        self.midi_mapping = {'kick': 36, 'snare': 38, 'hihat': 42}
        self.velocity = 100
        self.note_duration = 0.1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_grid_interval(self, bpm, drum_type=None):
        beat_duration = 60.0 / bpm
        division = self.grid_division.get(drum_type,
                                          self.default_grid_division) if drum_type else self.default_grid_division
        return beat_duration / (division / 4)

    def get_quantize_bias(self, drum_type):
        return self.quantize_bias.get(drum_type, 0.4)


def normalize_mel(mel_spec):
    mean = np.mean(mel_spec)
    std = np.std(mel_spec)
    if std > 1e-5:
        mel_spec = (mel_spec - mean) / std
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


# =============================================================================
# 고정밀 BPM 추정 시스템
# =============================================================================

class PrecisionBPMEstimator:
    """고정밀 BPM 추정 클래스"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.min_bpm = config.bpm_start_range
        self.max_bpm = config.bpm_end_range

    def estimate_bpm(self, wav_path: str, drum_onsets: Optional[Dict] = None) -> Tuple[float, Dict]:
        y, sr = librosa.load(wav_path, sr=self.config.sample_rate, mono=True)

        estimates = []
        methods_info = {}

        # 방법 1: Linear Regression
        bpm_lr, beats_lr, r_squared = self._estimate_with_linear_regression(y, sr)
        if bpm_lr is not None:
            methods_info['linear_regression'] = {
                'bpm': bpm_lr,
                'n_beats': len(beats_lr),
                'r_squared': r_squared
            }

            if r_squared >= 0.995:
                print(f"  [BPM] R² = {r_squared:.4f} (매우 높음) → Linear Regression 결과만 사용")
                final_bpm = self._refine_to_common_bpm(bpm_lr)
                return final_bpm, {'methods': methods_info, 'final_bpm': final_bpm, 'high_confidence': True}

            estimates.append(('linear_regression', bpm_lr, 0.45))

        # 방법 2: Tempogram
        bpm_tempogram = self._estimate_with_tempogram(y, sr)
        if bpm_tempogram is not None:
            estimates.append(('tempogram', bpm_tempogram, 0.25))
            methods_info['tempogram'] = {'bpm': bpm_tempogram}

        # 방법 3: 드럼 onset 기반
        if drum_onsets is not None:
            bpm_onset = self._estimate_from_drum_onsets(drum_onsets)
            if bpm_onset is not None:
                estimates.append(('drum_onsets', bpm_onset, 0.30))
                methods_info['drum_onsets'] = {'bpm': bpm_onset}

        final_bpm = self._ensemble_estimates(estimates)
        final_bpm = self._refine_to_common_bpm(final_bpm)

        info = {
            'methods': methods_info,
            'raw_estimates': [(m, b) for m, b, _ in estimates],
            'final_bpm': final_bpm
        }

        return final_bpm, info

    def _estimate_with_linear_regression(self, y: np.ndarray, sr: int) -> Tuple[Optional[float], np.ndarray, float]:
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time', trim=False)

            if len(beats) < 4:
                return None, np.array([]), 0.0

            indices = np.arange(len(beats))
            result = stats.linregress(indices, beats)

            beat_interval = result.slope
            r_squared = result.rvalue ** 2

            if beat_interval <= 0:
                return None, beats, 0.0

            bpm = 60.0 / beat_interval
            print(f"  [Linear Regression] R² = {r_squared:.4f}, raw BPM = {bpm:.2f}")

            if r_squared < 0.90:
                intervals = np.diff(beats)
                median_interval = np.median(intervals)
                if median_interval > 0:
                    bpm = 60.0 / median_interval
                    print(f"  [Linear Regression] R² 낮음, median 폴백: {bpm:.2f}")

            bpm = self._normalize_bpm_range(bpm)
            return bpm, beats, r_squared

        except Exception as e:
            print(f"[BPM] Linear Regression 추정 실패: {e}")
            return None, np.array([]), 0.0

    def _estimate_with_tempogram(self, y: np.ndarray, sr: int) -> Optional[float]:
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=512)
            ac_global = librosa.autocorrelate(onset_env, max_size=tempogram.shape[0])
            ac_global = librosa.util.normalize(ac_global)

            fps = sr / 512
            min_lag = int(fps * 60 / self.max_bpm)
            max_lag = int(fps * 60 / self.min_bpm)

            if max_lag > len(ac_global):
                max_lag = len(ac_global) - 1

            search_region = ac_global[min_lag:max_lag]
            peaks, properties = find_peaks(search_region, height=0.1)

            if len(peaks) == 0:
                return None

            best_peak_idx = peaks[np.argmax(properties['peak_heights'])]
            best_lag = min_lag + best_peak_idx
            bpm = fps * 60 / best_lag

            bpm = self._normalize_bpm_range(bpm)
            return bpm

        except Exception as e:
            print(f"[BPM] Tempogram 추정 실패: {e}")
            return None

    def _estimate_from_drum_onsets(self, drum_onsets: Dict) -> Optional[float]:
        try:
            main_onsets = []
            if 'kick' in drum_onsets:
                main_onsets.extend(drum_onsets['kick'])
            if 'snare' in drum_onsets:
                main_onsets.extend(drum_onsets['snare'])

            if len(main_onsets) < 4:
                return None

            main_onsets = sorted(main_onsets)
            intervals = np.diff(main_onsets)

            if len(intervals) < 3:
                return None

            q1, q3 = np.percentile(intervals, [25, 75])
            iqr = q3 - q1
            valid_mask = (intervals >= q1 - 1.5 * iqr) & (intervals <= q3 + 1.5 * iqr)
            valid_intervals = intervals[valid_mask]

            if len(valid_intervals) < 3:
                valid_intervals = intervals

            hist, bin_edges = np.histogram(valid_intervals, bins=50)
            peak_bin = np.argmax(hist)
            dominant_interval = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2

            bpm_candidates = []
            bpm_candidates.append(60.0 / dominant_interval)
            bpm_candidates.append(60.0 / (dominant_interval * 2))
            bpm_candidates.append(60.0 / (dominant_interval / 2))

            valid_bpms = [b for b in bpm_candidates if self.min_bpm <= b <= self.max_bpm]

            if not valid_bpms:
                return None

            target = 120
            best_bpm = min(valid_bpms, key=lambda x: abs(x - target))
            return best_bpm

        except Exception as e:
            print(f"[BPM] Drum onset 기반 추정 실패: {e}")
            return None

    def _normalize_bpm_range(self, bpm: float) -> float:
        while bpm < self.min_bpm:
            bpm *= 2
        while bpm > self.max_bpm:
            bpm /= 2
        return bpm

    def _ensemble_estimates(self, estimates: List[Tuple[str, float, float]]) -> float:
        if not estimates:
            return 120.0

        if len(estimates) == 1:
            return estimates[0][1]

        base_bpm = max(estimates, key=lambda x: x[2])[1]

        normalized = []
        for method, bpm, weight in estimates:
            ratio = bpm / base_bpm
            if 0.45 < ratio < 0.55:
                bpm *= 2
            elif 1.9 < ratio < 2.1:
                bpm /= 2
            normalized.append((bpm, weight))

        total_weight = sum(w for _, w in normalized)
        if total_weight == 0:
            return base_bpm

        weighted_sum = sum(b * w for b, w in normalized)
        return weighted_sum / total_weight

    def _refine_to_common_bpm(self, bpm: float) -> float:
        rounded = round(bpm)
        if abs(bpm - rounded) < 0.5:
            return float(rounded)
        return round(bpm, 2)


def detect_bpm_precision(wav_path: str, config: InferenceConfig,
                         drum_onsets: Optional[Dict] = None) -> Tuple[float, Dict]:
    estimator = PrecisionBPMEstimator(config)
    return estimator.estimate_bpm(wav_path, drum_onsets)


# =============================================================================
# 모델 추론 및 후처리
# =============================================================================

def predict_drum_onsets(mel_spec, model, config):
    """슬라이딩 윈도우 기반 모델 추론"""
    model.eval()
    total_frames = mel_spec.shape[0]
    predictions = np.zeros((total_frames, config.n_classes), dtype=np.float32)
    count_map = np.zeros(total_frames, dtype=np.float32)

    with torch.no_grad():
        start = 0
        while start < total_frames:
            end = min(start + config.window_size, total_frames)
            window = mel_spec[start:end]

            if window.shape[0] < config.window_size:
                pad_len = config.window_size - window.shape[0]
                window = np.pad(window, ((0, pad_len), (0, 0)), mode='constant')

            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(config.device)
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

        for idx in indices:
            t = frame_times[idx]
            if t >= 0:
                onsets[dt].append(t)

    return onsets


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


def quantize_to_grid(onsets, bpm, config):
    """
    onset 시간을 BPM 기반 그리드에 양자화 (수정된 버전)

    핵심 수정사항:
    1. 첫 onset을 무조건 가장 가까운 그리드로 (앞/뒤 균등 판단)
    2. quantize_bias 하향 조정으로 밀림 방지
    3. 첫 마디 시작점 보정
    """
    all_times = []
    for times in onsets.values():
        all_times.extend(times)

    if not all_times:
        return {dt: [] for dt in onsets.keys()}

    first_onset = min(all_times)
    one_beat = 60.0 / bpm
    base_grid = config.get_grid_interval(bpm, 'kick')  # 16분음표 간격

    # ============================================
    # 핵심 수정: 첫 onset 기준 오프셋 계산
    # ============================================

    # 첫 onset이 어느 마디의 몇 번째 그리드에 있는지 계산
    grid_position_in_beat = first_onset % base_grid

    # 가장 가까운 그리드로 이동 (앞/뒤 균등하게 판단)
    if grid_position_in_beat < base_grid * 0.5:
        # 현재 그리드로 당기기
        time_offset = -grid_position_in_beat
    else:
        # 다음 그리드로 밀기
        time_offset = base_grid - grid_position_in_beat

    # 단, 첫 onset이 매우 작은 경우 (0.1초 이하) → 1박 시작으로 간주
    if first_onset < 0.1:
        time_offset = -first_onset  # 0으로 맞춤
        print(f"  [양자화] 첫 onset({first_onset:.3f}s)이 매우 빠름 → 1박 시작으로 간주")
    # 첫 onset이 2박 이상 뒤에 있으면 의도적 쉼표로 판단
    elif first_onset > one_beat * 2:
        time_offset = 0
        print(f"  [양자화] 첫 onset({first_onset:.3f}s)이 2박 이상 뒤 → 오프셋 미적용")
    else:
        print(f"  [양자화] 첫 onset: {first_onset:.3f}s → 오프셋: {time_offset:+.4f}s")

    # ============================================
    # 각 드럼 타입별 양자화
    # ============================================
    quantized_onsets = {}
    for dt, times in onsets.items():
        if not times:
            quantized_onsets[dt] = []
            continue

        grid_interval = config.get_grid_interval(bpm, dt)
        bias = config.get_quantize_bias(dt)
        quantized = []

        for t in times:
            t_adjusted = t + time_offset
            if t_adjusted < 0:
                t_adjusted = 0

            # 수정된 양자화: floor 대신 round에 가깝게
            # bias가 0.4면, 그리드의 40% 지점을 기준으로 판단
            grid_index = int((t_adjusted / grid_interval) + bias)
            quantized_time = grid_index * grid_interval

            if quantized_time >= 0:
                quantized.append(quantized_time)

        quantized_onsets[dt] = sorted(list(set(quantized)))

    return quantized_onsets


def group_simultaneous_hits(onsets, config):
    all_events = []
    for dt, times in onsets.items():
        for t in times:
            all_events.append((t, dt))
    all_events.sort(key=lambda x: x[0])
    if not all_events:
        return []

    window = config.simultaneous_window_ms / 1000.0
    grouped = []
    curr_t, curr_d = all_events[0]
    curr_drums = [curr_d]

    for t, d in all_events[1:]:
        if t - curr_t <= window:
            if d not in curr_drums:
                curr_drums.append(d)
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
    """드럼 WAV → MIDI 변환 메인 함수"""
    if config is None:
        config = InferenceConfig()
    if output_dir is None:
        output_dir = os.path.dirname(wav_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"모델 로딩 중: {model_path}")
    model = DrumOnsetDetector(
        n_mels=config.n_mels, n_classes=config.n_classes,
        cnn_channels=config.cnn_channels, gru_hidden=config.gru_hidden,
        gru_layers=config.gru_layers, dropout=config.dropout
    ).to(config.device)

    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 1) 모델 추론
    mel_spec, _ = load_and_preprocess_audio(wav_path, config)
    frame_times = np.arange(mel_spec.shape[0]) * config.frame_duration
    predictions = predict_drum_onsets(mel_spec, model, config)

    onsets = detect_peaks(predictions, config, frame_times)
    onsets = merge_nearby_events(onsets, config)
    onsets = enforce_minimum_gap(onsets, config)

    # 2) BPM 추정
    if bpm_override:
        bpm = float(bpm_override)
        print(f"[BPM] 외부에서 BPM override 지정: {bpm:.2f}")
        bpm_info = {'override': True}
    else:
        bpm, bpm_info = detect_bpm_precision(wav_path, config, drum_onsets=onsets)
        print(f"[BPM] === 고정밀 BPM 추정 결과 ===")
        for method, info in bpm_info.get('methods', {}).items():
            print(f"  - {method}: {info.get('bpm', 'N/A'):.2f} BPM")
        print(f"  → 최종 BPM: {bpm:.2f}")

    # 3) 양자화 + 그룹핑
    quantized_onsets = quantize_to_grid(onsets, bpm, config)
    grouped = group_simultaneous_hits(quantized_onsets, config)

    # 4) MIDI 생성
    base_name = Path(wav_path).stem
    midi_path = os.path.join(output_dir, f"{base_name}_drums.mid")
    create_midi_file(grouped, bpm, config, midi_path)
    print(f"변환 완료: {midi_path} (BPM={bpm:.2f}, 이벤트 수={len(grouped)})")

    return midi_path, bpm, grouped


if __name__ == "__main__":
    pass