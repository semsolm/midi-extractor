"""
BiGRU용 멜 스펙트로그램 사전 계산 스크립트 (최종 개선 버전)
hop_length=256, ±1 프레임 확장 라벨로 WAV → Mel 변환

주요 개선사항:
- ±1 프레임 확장 라벨 적용
- 학습 속도를 5~20배 향상
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

from BiGRU_datautilr import DrumDatasetConfig
from precompute_worker import process_single_file

class PrecomputeConfig:
    """사전 계산 설정 (최종 개선 버전)"""
    def __init__(self):
        # 출력 디렉토리 (±1 프레임 확장 라벨용)
        self.output_root = "./precomputed_bigru_data_hop256_improved"

        # 멀티프로세싱 설정
        self.num_processes = 8  # CPU 코어 수에 맞게 조정

        # 데이터셋 설정
        self.dataset_config = DrumDatasetConfig()


def load_and_filter_metadata(config: DrumDatasetConfig) -> pd.DataFrame:
    """메타데이터 로딩 및 필터링"""
    df = pd.read_csv(config.csv_path)
    df = df[~df['style'].str.contains('jazz', case=False, na=False)]
    df = df[df['duration'] > config.min_duration]
    return df


def precompute_dataset(config: PrecomputeConfig):
    """전체 데이터셋 사전 계산"""
    print("=" * 80)
    print("🚀 BiGRU용 멜 스펙트로그램 사전 계산 시작 (최종 개선 버전)")
    print("=" * 80)
    print(f"📍 hop_length: {config.dataset_config.hop_length}")
    print(f"📍 프레임 간격: {config.dataset_config.frame_duration*1000:.1f}ms")
    print(f"📍 라벨 확장: ±{config.dataset_config.label_spread_frames} 프레임")
    print("=" * 80)

    dataset_config = config.dataset_config

    # 메타데이터 로딩
    print("\n📂 메타데이터 로딩 중...")
    df = load_and_filter_metadata(dataset_config)

    print(f"✅ 총 {len(df)} 파일 처리 예정")

    # Split별 처리
    for split in ['train', 'validation', 'test']:
        split_df = df[df['split'] == split].reset_index(drop=True)

        print(f"\n{'=' * 80}")
        print(f"📊 [{split.upper()}] 처리 중: {len(split_df)} 파일")
        print(f"{'=' * 80}")

        # 멀티프로세싱을 위한 인자 준비
        args_list = [
            (row, config.output_root, split)
            for _, row in split_df.iterrows()
        ]

        print(f"🔧 {config.num_processes}개 프로세스로 병렬 처리 중...")

        ctx = mp.get_context("spawn")  # Windows 안전하게
        with ctx.Pool(processes=config.num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, args_list),
                total=len(args_list),
                desc=f"{split}"
            ))

        success_count = sum(results)
        print(f"✅ {split}: {success_count}/{len(split_df)} 파일 성공")

    print("\n" + "=" * 80)
    print("✅ 사전 계산 완료!")
    print("=" * 80)
    print(f"📁 저장 위치: {config.output_root}")


def verify_precomputed_data(config: PrecomputeConfig):
    """사전 계산된 데이터 검증"""
    print("\n" + "=" * 80)
    print("🔍 데이터 검증 중...")
    print("=" * 80)

    for split in ['train', 'validation', 'test']:
        mel_dir = os.path.join(config.output_root, split, 'mel')
        label_dir = os.path.join(config.output_root, split, 'label')

        if not os.path.exists(mel_dir) or not os.path.exists(label_dir):
            print(f"❌ {split}: 디렉토리가 없습니다.")
            continue

        mel_files = list(Path(mel_dir).glob('*.npy'))
        label_files = list(Path(label_dir).glob('*.npy'))

        print(f"\n{split.upper()}:")
        print(f"  Mel files: {len(mel_files)}")
        print(f"  Label files: {len(label_files)}")

        if mel_files:
            sample_mel = np.load(mel_files[0])
            sample_label = np.load(str(mel_files[0]).replace('/mel/', '/label/').replace('\\mel\\', '\\label\\'))

            print(f"  샘플 shape:")
            print(f"    Mel: {sample_mel.shape}")
            print(f"    Label: {sample_label.shape}")
            print(f"    프레임 수: {sample_mel.shape[0]}")

            # 라벨 밀도 확인 (±1 프레임 확장으로 증가해야 함)
            label_density = sample_label.sum() / (sample_label.shape[0] * sample_label.shape[1])
            print(f"    Label density: {label_density:.4f} (±1 프레임 확장으로 증가)")

    print("\n✅ 검증 완료!")


if __name__ == "__main__":
    config = PrecomputeConfig()

    print("=" * 80)
    print("⚙️  사전 계산 설정 (최종 개선 버전)")
    print("=" * 80)
    print(f"출력 디렉토리: {config.output_root}")
    print(f"프로세스 수: {config.num_processes}")
    print(f"데이터 루트: {config.dataset_config.data_root}")
    print(f"hop_length: {config.dataset_config.hop_length}")
    print(f"프레임 간격: {config.dataset_config.frame_duration*1000:.1f}ms")
    print(f"라벨 확장: ±{config.dataset_config.label_spread_frames} 프레임")
    print("=" * 80)

    response = input("\n계속하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("취소되었습니다.")
        exit()

    precompute_dataset(config)
    verify_precomputed_data(config)

    print("\n" + "=" * 80)
    print("💡 다음 단계:")
    print("=" * 80)
    print("1. BiGRU_datautilr_improved.py를 사용하여 학습하세요.")
    print("2. BiGRU_train_improved.py에서 use_precomputed=True로 설정")
    print("3. 학습 속도가 5~20배 빨라집니다!")
    print("=" * 80)