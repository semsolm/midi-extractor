"""
precompute_worker_improved.py (최종 개선 버전)
hop_length=256, ±1 프레임 확장 라벨 적용
"""

import os
import numpy as np
from BiGRU_datautilr import DrumDatasetConfig, wav_to_melspectrogram, midi_to_labels

def process_single_file(args):
    """
    단일 파일 처리 함수 (멀티프로세싱용)
    ±1 프레임 확장 라벨 적용
    """
    row, output_root, split = args

    # 각 프로세스에서 독립적인 설정 인스턴스 생성
    config = DrumDatasetConfig()
    # 사전계산 시에는 augmentation 끔
    config.use_augmentation = False
    config.use_spec_augment = False

    try:
        wav_filename = row['audio_filename']
        midi_filename = row['midi_filename']

        wav_path = os.path.join(config.data_root, wav_filename)
        midi_path = os.path.join(config.data_root, midi_filename)

        # 출력 경로 생성
        base_name = (
            wav_filename.replace('/', '_')
                        .replace('\\', '_')
                        .replace('.wav', '')
        )

        mel_output_dir = os.path.join(output_root, split, 'mel')
        label_output_dir = os.path.join(output_root, split, 'label')

        os.makedirs(mel_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)

        mel_output_path = os.path.join(mel_output_dir, f"{base_name}.npy")
        label_output_path = os.path.join(label_output_dir, f"{base_name}.npy")

        # 이미 처리된 파일이면 스킵
        if os.path.exists(mel_output_path) and os.path.exists(label_output_path):
            return True

        # 1. WAV → 멜스펙트로그램 (augment=False)
        mel_spec = wav_to_melspectrogram(wav_path, config, augment=False)

        # 2. MIDI → 레이블 (±1 프레임 확장 적용)
        n_frames = mel_spec.shape[0]
        labels = midi_to_labels(midi_path, n_frames, config)

        # 3. 저장
        np.save(mel_output_path, mel_spec)
        np.save(label_output_path, labels)

        return True

    except Exception as e:
        print(f"❌ 오류 발생: {row['audio_filename']}")
        print(f"   {str(e)}")
        return False