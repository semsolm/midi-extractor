"""
드럼 WAV 파일 → MIDI 악보 변환 파이프라인 (최종 개선 버전)
CNN+BiGRU 모델을 사용한 드럼 타격 검출 및 MIDI 생성

주요 개선사항:
1. ✅ Mel normalization 적용 (BiGRU_datautilr_final.py와 동일)
2. ✅ Checkpoint에서 최적 threshold 자동 로드
3. ✅ 드럼별 독립적인 양자화 그리드 설정
4. ✅ Rising edge detection으로 onset delay 최소화
5. ✅ Floor-based quantization으로 타이밍 시프트 방지

사용법:
    코드 맨 아래 실행 설정에서 파라미터 수정 후 실행
    python MiDi_maker_final.py
"""

import os
import math
import numpy as np
import torch
import librosa
import pretty_midi
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

from app.services.BiGRU_model import DrumOnsetDetector


# ============================================
# 설정 클래스
# ============================================
class InferenceConfig:
    """추론 및 MIDI 변환 설정 (최종 개선 버전)"""

    def __init__(self):
        # ============================================
        # 오디오 전처리 설정 (BiGRU_datautilr_final.py와 동일!)
        # ============================================
        self.sample_rate = 22050
        self.n_fft = 2048
        self.hop_length = 256
        self.n_mels = 128
        self.fmin = 20
        self.fmax = 8000

        # Mel normalization 적용 (개선!)
        self.use_mel_normalization = True

        # 시간 해상도
        self.frame_duration = self.hop_length / self.sample_rate  # ~11.6ms

        # ============================================
        # 모델 설정 (BiGRU_model.py와 동일!)
        # ============================================
        self.n_classes = 3  # kick, snare, hihat
        self.cnn_channels = [32, 64, 128]
        self.gru_hidden = 384
        self.gru_layers = 2
        self.dropout = 0.3

        # ============================================
        # Sliding Window 설정
        # ============================================
        self.window_size = 2000  # 프레임 (~23초)
        self.hop_size = 1000  # 50% overlap

        # ============================================
        # 타격 검출 임계값 (checkpoint에서 자동 로드됨)
        # ============================================
        self.thresholds = {
            'kick': 0.45,
            'snare': 0.45,
            'hihat': 0.35  # 기본값
        }

        # ============================================
        # BPM 자동 감지 설정
        # ============================================
        self.bpm_start_range = 60
        self.bpm_end_range = 200

        # ============================================
        # 양자화 설정 (드럼별 독립 그리드)
        # ============================================
        self.grid_division = {
            'kick': 16,  # 16분음표 그리드
            'snare': 16,
            'hihat': 8   # 8분음표 그리드
        }

        self.default_grid_division = 16

        # ============================================
        # 후처리 설정 (4단계 파이프라인)
        # ============================================
        # Stage 1: 인접 이벤트 병합
        self.merge_window_ms = 50

        # Stage 2: 최소 간격 강제
        self.min_gap_ms = {
            'kick': 80,
            'snare': 60,
            'hihat': 30
        }

        # Stage 3: 그리드 양자화 바이어스
        self.quantize_bias = {
            'kick': 0.3,
            'snare': 0.3,
            'hihat': 0.25
        }

        # Stage 4: 동시 타격 허용 범위
        self.simultaneous_window_ms = 30

        # ============================================
        # MIDI 출력 설정 (General MIDI Drum Map)
        # ============================================
        self.midi_mapping = {
            'kick': 36,  # Bass Drum 1
            'snare': 38,  # Acoustic Snare
            'hihat': 42  # Closed Hi-Hat
        }

        self.velocity = 100
        self.note_duration = 0.1

        # ============================================
        # 디바이스 설정
        # ============================================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device.type == 'cuda':
            try:
                test_tensor = torch.zeros(1).to(self.device)
                del test_tensor
                print(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
            except:
                print("⚠️  GPU 사용 불가, CPU로 전환")
                self.device = torch.device('cpu')
        else:
            print("ℹ️  CPU 모드로 실행")

    def get_grid_interval(self, bpm: float, drum_type: str = None) -> float:
        """BPM과 드럼 타입에 따른 그리드 간격 계산"""
        beat_duration = 60.0 / bpm

        if drum_type and drum_type in self.grid_division:
            division = self.grid_division[drum_type]
        else:
            division = self.default_grid_division

        grid_interval = beat_duration / (division / 4)
        return grid_interval

    def get_quantize_bias(self, drum_type: str) -> float:
        """드럼별 양자화 바이어스 가져오기"""
        return self.quantize_bias.get(drum_type, 0.3)

    def load_thresholds_from_checkpoint(self, checkpoint: dict):
        """
        Checkpoint에서 최적 threshold 로드 (개선!)

        Args:
            checkpoint: 학습된 모델 checkpoint
        """
        if 'config' in checkpoint and 'thresholds' in checkpoint['config']:
            thresholds_list = checkpoint['config']['thresholds']

            # List to dict conversion
            drum_types = ['kick', 'snare', 'hihat']
            for i, drum_type in enumerate(drum_types):
                if i < len(thresholds_list):
                    self.thresholds[drum_type] = thresholds_list[i]

            print(f"\n✅ Checkpoint에서 최적 threshold 로드:")
            print(f"   Kick:  {self.thresholds['kick']:.2f}")
            print(f"   Snare: {self.thresholds['snare']:.2f}")
            print(f"   Hihat: {self.thresholds['hihat']:.2f}")
        else:
            print(f"\n⚠️  Checkpoint에 threshold 정보 없음, 기본값 사용")


# ============================================
# Mel Normalization (개선!)
# ============================================
def normalize_mel(mel_spec: np.ndarray) -> np.ndarray:
    """
    멜스펙트로그램 정규화 (BiGRU_datautilr_final.py와 동일)

    개선: 녹음 세기에 따른 dynamic range 변동 완화
    """
    mean = np.mean(mel_spec)
    std = np.std(mel_spec)

    if std > 1e-5:
        mel_spec = (mel_spec - mean) / std

    return mel_spec


# ============================================
# WAV → Mel Spectrogram (개선!)
# ============================================
def load_and_preprocess_audio(wav_path: str, config: InferenceConfig) -> Tuple[np.ndarray, float]:
    """
    WAV 파일을 로드하고 멜스펙트로그램으로 변환

    개선: Mel normalization 추가
    """
    print(f"\n📂 오디오 로딩: {wav_path}")

    # WAV 로드
    y, sr = librosa.load(wav_path, sr=config.sample_rate, mono=True)
    duration = len(y) / sr

    print(f"   샘플레이트: {sr} Hz")
    print(f"   길이: {duration:.2f}초")
    print(f"   샘플 수: {len(y):,}")

    # 멜스펙트로그램 변환
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax
    )

    # dB 스케일 변환
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = mel_spec_db.T  # (n_frames, n_mels)

    # Mel normalization 적용 (개선!)
    if config.use_mel_normalization:
        mel_spec_db = normalize_mel(mel_spec_db)
        print(f"   ✅ Mel normalization 적용")

    print(f"   멜스펙트로그램 shape: {mel_spec_db.shape}")
    print(f"   프레임 수: {mel_spec_db.shape[0]}")
    print(f"   프레임 간격: {config.frame_duration * 1000:.1f}ms (hop_length={config.hop_length})")

    return mel_spec_db, duration


# ============================================
# BPM 자동 감지
# ============================================
def detect_bpm(wav_path: str, config: InferenceConfig) -> float:
    """BPM 자동 감지 (librosa beat tracking)"""
    print(f"\n🎵 BPM 자동 감지 중...")

    y, sr = librosa.load(wav_path, sr=config.sample_rate, mono=True)

    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr,
        start_bpm=120.0
    )

    bpm = float(tempo)
    if bpm < config.bpm_start_range:
        bpm = bpm * 2
    elif bpm > config.bpm_end_range:
        bpm = bpm / 2

    bpm = np.clip(bpm, config.bpm_start_range, config.bpm_end_range)

    print(f"   감지된 BPM: {bpm:.1f}")
    return bpm


# ============================================
# 모델 추론 (Sliding Window)
# ============================================
def predict_drum_onsets(
    mel_spec: np.ndarray,
    model: torch.nn.Module,
    config: InferenceConfig
) -> np.ndarray:
    """Sliding window 방식으로 드럼 타격 예측"""
    print(f"\n🔮 드럼 타격 예측 중...")

    model.eval()
    n_frames = mel_spec.shape[0]
    predictions = np.zeros((n_frames, config.n_classes), dtype=np.float32)
    count_map = np.zeros(n_frames, dtype=np.float32)

    print(f"   총 프레임: {n_frames}")
    print(f"   윈도우 크기: {config.window_size}")
    print(f"   홉 크기: {config.hop_size}")

    with torch.no_grad():
        start = 0
        window_idx = 0

        while start < n_frames:
            end = min(start + config.window_size, n_frames)

            window = mel_spec[start:end]
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(config.device)

            logits = model(window_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            actual_len = end - start
            predictions[start:end] += probs[:actual_len]
            count_map[start:end] += 1

            window_idx += 1
            start += config.hop_size

    count_map[count_map == 0] = 1
    predictions = predictions / count_map[:, np.newaxis]

    print(f"   ✅ 예측 완료 (윈도우 수: {window_idx})")
    return predictions


# ============================================
# Peak Detection (Rising Edge Detection)
# ============================================
def detect_peaks(
        predictions: np.ndarray,
        config: InferenceConfig,
        frame_times: np.ndarray
) -> Dict[str, List[float]]:
    """예측 확률에서 피크를 검출하여 타격 시간 추출"""
    print(f"\n🎯 드럼 타격 검출 중 (Rising Edge Detection)...")

    drum_types = ['kick', 'snare', 'hihat']
    onsets = {dt: [] for dt in drum_types}

    for drum_idx, drum_type in enumerate(drum_types):
        probs = predictions[:, drum_idx]
        threshold = config.thresholds[drum_type]

        # Rising edge detection
        onset_indices = np.where(
            (probs[:-1] < threshold) & (probs[1:] >= threshold)
        )[0] + 1

        onset_times = [frame_times[idx] for idx in onset_indices]
        onsets[drum_type] = onset_times

        print(f"   {drum_type:6s}: {len(onset_times):4d}개 검출 (threshold={threshold:.2f})")

    return onsets


# ============================================
# 후처리 Stage 1: 인접 이벤트 병합
# ============================================
def merge_nearby_events(
        onsets: Dict[str, List[float]],
        config: InferenceConfig
) -> Dict[str, List[float]]:
    """인접한 이벤트를 병합하여 중복 제거"""
    print(f"\n🔧 후처리 Stage 1: 인접 이벤트 병합 (±{config.merge_window_ms}ms)")

    merge_window = config.merge_window_ms / 1000.0
    merged_onsets = {}

    for drum_type, times in onsets.items():
        if len(times) == 0:
            merged_onsets[drum_type] = []
            continue

        times = sorted(times)
        merged = [times[0]]

        for t in times[1:]:
            if t - merged[-1] <= merge_window:
                merged[-1] = merged[-1] * 0.7 + t * 0.3
            else:
                merged.append(t)

        before_count = len(times)
        after_count = len(merged)
        merged_onsets[drum_type] = merged

        print(f"   {drum_type:6s}: {before_count:4d} → {after_count:4d} ({before_count - after_count:3d}개 병합)")

    return merged_onsets


# ============================================
# 후처리 Stage 2: 최소 간격 강제
# ============================================
def enforce_minimum_gap(
        onsets: Dict[str, List[float]],
        config: InferenceConfig
) -> Dict[str, List[float]]:
    """물리적으로 불가능한 빠른 연타 제거"""
    print(f"\n🔧 후처리 Stage 2: 최소 간격 강제")

    filtered_onsets = {}

    for drum_type, times in onsets.items():
        if len(times) == 0:
            filtered_onsets[drum_type] = []
            continue

        min_gap = config.min_gap_ms[drum_type] / 1000.0
        times = sorted(times)
        filtered = [times[0]]

        for t in times[1:]:
            if t - filtered[-1] >= min_gap:
                filtered.append(t)

        before_count = len(times)
        after_count = len(filtered)
        filtered_onsets[drum_type] = filtered

        print(f"   {drum_type:6s}: {before_count:4d} → {after_count:4d} "
              f"(최소 간격 {config.min_gap_ms[drum_type]}ms, {before_count - after_count:3d}개 제거)")

    return filtered_onsets


# ============================================
# 후처리 Stage 3: 드럼별 그리드 기반 양자화
# ============================================
def quantize_to_grid(
        onsets: Dict[str, List[float]],
        bpm: float,
        config: InferenceConfig
) -> Dict[str, List[float]]:
    """
    드럼별로 다른 그리드에 맞춰 타격 시간 양자화
    """
    print(f"\n🔧 후처리 Stage 3: 드럼별 그리드 기반 양자화 (BPM={bpm:.1f})")

    quantized_onsets = {}

    for drum_type, times in onsets.items():
        if len(times) == 0:
            quantized_onsets[drum_type] = []
            continue

        grid_interval = config.get_grid_interval(bpm, drum_type)
        grid_division = config.grid_division.get(drum_type, config.default_grid_division)
        bias = config.get_quantize_bias(drum_type)

        print(f"   {drum_type:6s}: {grid_division}분음표 그리드 (간격 {grid_interval * 1000:.1f}ms, bias={bias})")

        quantized = []
        for t in times:
            grid_index = math.floor((t / grid_interval) + bias)
            quantized_time = grid_index * grid_interval
            quantized.append(quantized_time)

        quantized = sorted(list(set(quantized)))

        before_count = len(times)
        after_count = len(quantized)
        quantized_onsets[drum_type] = quantized

        print(f"            {before_count:4d} → {after_count:4d} ({before_count - after_count:3d}개 중복 제거)")

    return quantized_onsets


# ============================================
# 후처리 Stage 4: 동시 타격 그룹핑
# ============================================
def group_simultaneous_hits(
        onsets: Dict[str, List[float]],
        config: InferenceConfig
) -> List[Tuple[float, List[str]]]:
    """동시에 발생하는 타격을 그룹핑"""
    print(f"\n🔧 후처리 Stage 4: 동시 타격 그룹핑 (±{config.simultaneous_window_ms}ms)")

    all_events = []
    for drum_type, times in onsets.items():
        for t in times:
            all_events.append((t, drum_type))

    all_events.sort(key=lambda x: x[0])

    if len(all_events) == 0:
        return []

    window = config.simultaneous_window_ms / 1000.0
    grouped_events = []
    current_time = all_events[0][0]
    current_drums = [all_events[0][1]]

    for time, drum in all_events[1:]:
        if time - current_time <= window:
            if drum not in current_drums:
                current_drums.append(drum)
        else:
            grouped_events.append((current_time, sorted(current_drums)))
            current_time = time
            current_drums = [drum]

    grouped_events.append((current_time, sorted(current_drums)))

    multi_hits = sum(1 for _, drums in grouped_events if len(drums) > 1)
    print(f"   총 이벤트: {len(grouped_events)}개")
    print(f"   동시 타격: {multi_hits}개")

    return grouped_events


# ============================================
# MIDI 파일 생성
# ============================================
def create_midi_file(
        grouped_events: List[Tuple[float, List[str]]],
        bpm: float,
        config: InferenceConfig,
        output_path: str
):
    """그룹화된 이벤트로부터 MIDI 파일 생성"""
    print(f"\n🎼 MIDI 파일 생성 중...")

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    drum_program = 0
    drum_instrument = pretty_midi.Instrument(program=drum_program, is_drum=True, name='Drums')

    for time, drum_types in grouped_events:
        for drum_type in drum_types:
            note_number = config.midi_mapping[drum_type]

            note = pretty_midi.Note(
                velocity=config.velocity,
                pitch=note_number,
                start=time,
                end=time + config.note_duration
            )
            drum_instrument.notes.append(note)

    pm.instruments.append(drum_instrument)
    pm.write(output_path)

    print(f"   ✅ MIDI 파일 저장: {output_path}")
    print(f"   BPM: {bpm:.1f}")
    print(f"   총 노트 수: {len(drum_instrument.notes)}")


# ============================================
# 디버그 텍스트 로그 생성
# ============================================
def create_debug_log(
        grouped_events: List[Tuple[float, List[str]]],
        bpm: float,
        config: InferenceConfig,
        output_path: str
):
    """디버그용 텍스트 로그 생성"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("드럼 타격 검출 결과 (최종 개선 버전)\n")
        f.write("=" * 80 + "\n")
        f.write(f"BPM: {bpm:.1f}\n")
        f.write(f"총 이벤트: {len(grouped_events)}개\n")
        f.write(f"hop_length: 256 (시간 해상도 ~11.6ms)\n")
        f.write(f"Mel normalization: {config.use_mel_normalization}\n")
        f.write(f"양자화 설정:\n")
        for drum_type, division in config.grid_division.items():
            f.write(f"  - {drum_type}: {division}분음표 그리드\n")
        f.write(f"Threshold:\n")
        for drum_type, threshold in config.thresholds.items():
            f.write(f"  - {drum_type}: {threshold:.2f}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'시간(초)':>10s} {'시간(분:초)':>12s} {'드럼':20s}\n")
        f.write("-" * 80 + "\n")

        for time, drums in grouped_events:
            minutes = int(time // 60)
            seconds = time % 60
            time_str = f"{minutes:02d}:{seconds:05.2f}"
            drums_str = " + ".join(drums)
            f.write(f"{time:10.3f} {time_str:>12s} {drums_str:20s}\n")

    print(f"   ✅ 디버그 로그 저장: {output_path}")


# ============================================
# 메인 파이프라인
# ============================================
def drum_wav_to_midi(
    wav_path: str,
    model_path: str,
    output_dir: str = None,
    config: InferenceConfig = None,
    bpm_override: float = None
):
    """드럼 WAV → MIDI 변환 메인 파이프라인 (최종 개선 버전)"""
    if config is None:
        config = InferenceConfig()

    print("\n" + "=" * 80)
    print("🥁 드럼 WAV → MIDI 악보 변환 파이프라인 (최종 개선 버전)")
    print("=" * 80)
    print("🎯 주요 개선사항:")
    print("  - ✅ Mel normalization 적용")
    print("  - ✅ Checkpoint에서 최적 threshold 자동 로드")
    print("  - ✅ 드럼별 독립 양자화 그리드")
    print("=" * 80)

    if output_dir is None:
        output_dir = os.path.dirname(wav_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = Path(wav_path).stem
    midi_path = os.path.join(output_dir, f"{base_name}_drums.mid")
    log_path = os.path.join(output_dir, f"{base_name}_drums.txt")

    # Step 1: 모델 로드 + Threshold 로드 (개선!)
    print(f"\n📦 모델 로딩: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=config.device,
        weights_only=False
    )

    # Checkpoint에서 최적 threshold 로드 (개선!)
    #config.load_thresholds_from_checkpoint(checkpoint)

    model = DrumOnsetDetector(
        n_mels=config.n_mels,
        n_classes=config.n_classes,
        cnn_channels=config.cnn_channels,
        gru_hidden=config.gru_hidden,
        gru_layers=config.gru_layers,
        dropout=config.dropout
    ).to(config.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"   ✅ 모델 로드 완료")
    print(f"   Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"   Val F1: {checkpoint['metrics'].get('f1_avg', 0):.4f}")

    # Step 2: BPM 감지
    if bpm_override is not None:
        print(f"\n🎵 BPM 수동 설정 사용: {bpm_override:.1f}")
        bpm = float(bpm_override)
    else:
        bpm = detect_bpm(wav_path, config)

    # Step 3: 오디오 전처리 (Mel normalization 포함)
    mel_spec, duration = load_and_preprocess_audio(wav_path, config)

    n_frames = mel_spec.shape[0]
    frame_times = np.arange(n_frames) * config.frame_duration

    # Step 4: 드럼 타격 예측
    predictions = predict_drum_onsets(mel_spec, model, config)

    # Step 5: Peak Detection
    onsets = detect_peaks(predictions, config, frame_times)

    # Step 6: 4단계 후처리 파이프라인
    onsets = merge_nearby_events(onsets, config)
    onsets = enforce_minimum_gap(onsets, config)
    onsets = quantize_to_grid(onsets, bpm, config)
    grouped_events = group_simultaneous_hits(onsets, config)

    # Step 7: MIDI 파일 생성
    create_midi_file(grouped_events, bpm, config, midi_path)

    # Step 8: 디버그 로그 생성
    create_debug_log(grouped_events, bpm, config, log_path)

    # 완료
    print("\n" + "=" * 80)
    print("✅ 변환 완료!")
    print("=" * 80)
    print(f"📄 입력 WAV: {wav_path}")
    print(f"🎼 출력 MIDI: {midi_path}")
    print(f"📝 출력 로그: {log_path}")
    print(f"⏱️  길이: {duration:.2f}초")
    print(f"🎵 BPM: {bpm:.1f}")
    print(f"🎯 총 이벤트: {len(grouped_events)}개")
    print("=" * 80 + "\n")

    return midi_path, bpm, grouped_events
# ============================================
# 실행 설정
# ============================================
if __name__ == "__main__":
    # ====================================
    # 🎯 여기서 파라미터 수정하세요!
    # ====================================

    # 필수 파라미터
    WAV_PATH = r"D:\model_test\drums.wav"
    MODEL_PATH = r"D:\model_test\new_BIGRU\checkpoints_final\best_model.pt"  # 최종 모델 경로
    OUTPUT_DIR = None

    # BPM 설정
    BPM = None  # None이면 자동 감지, 숫자 입력 시 수동 설정

    # 설정 객체 생성
    config = InferenceConfig()

    # ====================================
    # 🎵 드럼별 그리드 설정 (선택사항)
    # ====================================
    config.grid_division['kick'] = 16
    config.grid_division['snare'] = 8
    config.grid_division['hihat'] = 8

    config.quantize_bias['kick'] = 1
    config.quantize_bias['snare'] = 0.3
    config.quantize_bias['hihat'] = 0.25

    # ====================================
    # 🔧 임계값 수동 조정 (선택사항)
    # ====================================
    # ⚠️  주의: Checkpoint에서 자동 로드되므로 필요시에만 수정
    config.thresholds['kick'] = 0.5
    config.thresholds['snare'] = 0.5
    config.thresholds['hihat'] = 0.15

    # ====================================
    # 🔧 후처리 설정 조정 (선택사항)
    # ====================================
    config.merge_window_ms = 50

    config.min_gap_ms['kick'] = 80
    config.min_gap_ms['snare'] = 60
    config.min_gap_ms['hihat'] = 30

    config.simultaneous_window_ms = 30

    # ====================================
    # 🎼 MIDI 출력 설정 (선택사항)
    # ====================================
    config.velocity = 100
    config.note_duration = 0.1

    # ====================================
    # 🚀 실행
    # ====================================
    print("\n" + "=" * 80)
    print("🎛️  현재 설정:")
    print("=" * 80)
    print(f"📄 WAV 파일: {WAV_PATH}")
    print(f"🤖 모델 파일: {MODEL_PATH}")
    print(f"📁 출력 디렉토리: {OUTPUT_DIR if OUTPUT_DIR else 'WAV 파일과 같은 위치'}")
    print(f"🎵 BPM: {BPM if BPM else '자동 감지'}")
    print(f"✅ Mel normalization: {config.use_mel_normalization}")
    print(f"\n🎵 드럼별 양자화 그리드:")
    print(f"  - Kick:  {config.grid_division['kick']}분음표")
    print(f"  - Snare: {config.grid_division['snare']}분음표")
    print(f"  - Hihat: {config.grid_division['hihat']}분음표")
    print("=" * 80)

    # 파이프라인 실행
    drum_wav_to_midi(
        wav_path=WAV_PATH,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        config=config,
        bpm_override=BPM
    )