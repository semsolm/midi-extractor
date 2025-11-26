# 🥁 드럼 사운드 자동 분류 및 악보 생성 AI 시스템

## 1. 팀 정보  

**팀명:** 경로당

| 팀원  | 학번 | 역할 | Github | 비고 |
|-----|------|------|--------|------|
| 윤상일 | 2020E7424 | AI/ML | [@semsolm](https://github.com/semsolm) | 모델 설계 및 학습 |
| 양태양 | 2021E7411 | Frontend | [@sunning838](https://github.com/sunning838) | UI 개발 |
| 최유진 | 2023E7518 | Frontend | [@cyj4795](https://github.com/cyj4795) | UI 디자인 |
| 이준행 | 2020E7427 | PM | [@LeopoldBloom2K](https://github.com/LeopoldBloom2K) | 백엔드 |
| 정서영 | 2020U2329 | Backend | [@jwy23190](https://github.com/jwy23190) | 백엔드, 프론트 |

---

## 2. 프로젝트 개요

### 🔹 프로젝트 제목
드럼 사운드 자동 분류 및 악보 생성 시스템

### 🔹 프로젝트 목적
사용자가 업로드한 오디오 파일에서 드럼 사운드를 자동으로 인식하고 악보를 생성하는 AI 시스템을 개발합니다. 시스템은 오디오에서 드럼 트랙을 분리하고, 개별 타격음을 **Kick, Snare, Hi-hat** 3가지 클래스로 분류합니다. 최종적으로 분류된 드럼 노트를 기반으로 **MIDI 파일**과 **PDF 악보**를 자동으로 생성하여 제공합니다.

---

## 3. 데이터셋

### 📘 데이터 출처 및 구성

- **데이터셋 이름**: Expanded Groove MIDI Dataset (E-GMD v1.0.0)
- **데이터 형식**: WAV 오디오 파일 + MIDI 레이블
- **분류 클래스**: 3가지 주요 드럼 타입
  - `kick` (Bass Drum, MIDI note 36)
  - `snare` (Snare Drum, MIDI notes 38, 40)
  - `hihat` (Hi-hat, Crash, Ride 등, MIDI notes 42, 44, 46, 49, 51, 52, 53, 55, 57, 59)

### 📘 데이터 필터링
- **제외 스타일**: Jazz (리듬 패턴이 학습 목표와 상이하여 제외)
- **최소 길이**: 10초 이상의 오디오만 사용
- **데이터 분할**: Train / Validation / Test (CSV 메타데이터의 'split' 컬럼 기준)

### 📘 MIDI 노트 매핑
```python
DRUM_MAPPING = {
    'kick': [36],
    'snare': [38, 40],
    'hihat': [42, 44, 46, 49, 57, 51, 59, 52, 55, 53]
}
```

---

## 4. 데이터 전처리 및 증강

### 🎵 오디오 전처리 파이프라인

| 항목 | 설정값 | 설명 |
|-----|-------|------|
| **샘플링 레이트** | 22,050 Hz | 모든 오디오를 동일한 샘플링 레이트로 통일 |
| **N_FFT** | 2,048 | FFT window 크기 |
| **Hop Length** | 256 samples | 프레임 간격 (~11.6ms) |
| **멜 필터뱅크** | 128 bins | 주파수 해상도 |
| **주파수 범위** | 20 ~ 8,000 Hz | 드럼 사운드의 주요 주파수 대역 |
| **스케일링** | Power to dB | 데시벨 스케일로 변환 |
| **정규화** | Mean/Std 정규화 | 녹음 세기에 따른 변동 완화 |

### 🎵 멜스펙트로그램 변환 과정
```python
1. WAV 파일 로드 (sr=22050)
2. 멜스펙트로그램 추출 (n_mels=128, hop_length=256)
3. Power to dB 변환
4. Mean/Std 정규화 (optional)
5. 최종 형태: (T, 128) - T는 시간 프레임 수
```

### 🎨 데이터 증강 기법

#### 1. 오디오 레벨 증강
- **Gain Adjustment**: ±6dB 범위 내에서 무작위 음량 조절
- **적용 확률**: 50%
- **목적**: 다양한 녹음 환경 시뮬레이션

#### 2. SpecAugment
- **Frequency Masking**: 
  - 마스크 크기: 최대 12 bins
  - 횟수: 2회
  - 목적: 주파수 변동에 강건한 모델 학습
- **Time Masking**: 
  - 마스크 크기: 최대 10 프레임 (onset 보존을 위해 축소)
  - 횟수: 2회
  - 목적: 시간적 변동에 강건한 모델 학습

#### 3. Silent Sample Augmentation
- **무음 샘플 비율**: 15% 
- **길이**: 500~1500 프레임 (무작위)
- **목적**: False Positive 감소 (무음 구간에서 타격 오검출 방지)

### 🏷️ 레이블 생성 전략

#### ±1 프레임 확장 (Label Spread)
- MIDI onset 시간을 프레임 인덱스로 변환
- 해당 프레임의 ±1 프레임(총 3프레임)을 positive로 마킹
- **목적**: onset 검출의 시간적 여유 제공 (모델이 정확한 한 프레임을 맞추기 어려운 문제 해결)

```python
# 예: onset이 frame 100에 있다면
# frame 99, 100, 101 모두를 positive label로 설정
labels[frame_idx-1:frame_idx+2, drum_class] = 1.0
```

### 🚀 사전 계산 데이터 (Precomputed Data)
- 학습 속도 향상을 위해 멜스펙트로그램과 레이블을 `.npy` 형식으로 사전 계산
- 디스크에서 직접 로드하여 실시간 변환 오버헤드 제거
- 경로: `./precomputed_bigru_data_hop256_final/`

---

## 5. 모델 아키텍처

### 🧠 BiGRU 기반 드럼 타격 검출 모델 (DrumOnsetDetector)

본 프로젝트는 CNN과 BiGRU를 결합한 시퀀스 투 시퀀스 모델을 사용하여 멀티레이블 분류를 수행합니다.

#### 📊 모델 구조

```
Input: (Batch, SeqLen, 128) - 멜스펙트로그램
    ↓
[CNN Feature Extractor]
    ↓ Block 1: Conv-Conv-Pool (32 channels)
    ↓ Block 2: Conv-Conv-Pool (64 channels)
    ↓ Block 3: Conv-Conv-Pool (128 channels)
    ↓ Output: (Batch, SeqLen, 2048) - 압축된 특징
    ↓
[BiGRU Temporal Modeling]
    ↓ 2-layer BiGRU (hidden=384)
    ↓ Output: (Batch, SeqLen, 768)
    ↓
[Fully Connected Classifier]
    ↓ FC (768 → 128) + ReLU + Dropout
    ↓ FC (128 → 3)
    ↓
Output: (Batch, SeqLen, 3) - 각 프레임의 3개 클래스 로짓
```

#### 🔧 주요 하이퍼파라미터

| 구성 요소 | 설정값 | 설명 |
|---------|-------|------|
| **CNN Channels** | [32, 64, 128] | 각 블록의 채널 수 |
| **Conv Kernel** | (3, 3) | 3x3 커널 사용 |
| **Pooling** | (1, 2) | 주파수 축만 pooling (시간 해상도 유지) |
| **GRU Hidden** | 384 | 양방향이므로 출력은 768 |
| **GRU Layers** | 2 | 2층 BiGRU 스택 |
| **Dropout** | 0.3 | 과적합 방지 |
| **총 파라미터** | ~5.6M | 학습 가능한 파라미터 수 |

#### 🎯 설계 철학

1. **Conv-Conv-Pool 패턴**: 
   - 단일 Conv-Pool보다 특징 추출 능력 향상
   - Batch Normalization으로 학습 안정화

2. **BiGRU의 선택**:
   - LSTM 대비 파라미터 수 25% 감소 (학습 속도 향상)
   - 양방향 구조로 과거와 미래 문맥 모두 활용
   - Hidden size 384로 충분한 표현력 확보

3. **시간 해상도 보존**:
   - Pooling을 주파수 축에만 적용 (1, 2)
   - 시간 축은 그대로 유지하여 정확한 onset 검출

---

## 6. 학습 전략

### 🎯 손실 함수: Weighted Focal BCE Loss


- **α (alpha)**: 0.25 - positive/negative 샘플 균형 조절
- **γ (gamma)**: 2.0 - 쉬운 샘플의 loss 감소 정도
- **클래스 가중치**: [2.0, 1.5, 1.0] (kick/snare/hihat)

#### Focal Loss 선택 이유
1. **클래스 불균형 해결**: kick/snare는 hihat보다 발생 빈도가 낮음
2. **Hard Example Mining**: 어려운 샘플(잘못 분류되는 샘플)에 집중
3. **False Positive 감소**: 쉬운 negative 샘플의 영향력 감소

### ⚙️ 최적화 설정

| 항목 | 설정값 | 설명 |
|-----|-------|------|
| **Optimizer** | AdamW | Weight decay가 포함된 Adam |
| **Learning Rate** | 1e-3 | 초기 학습률 |
| **Weight Decay** | 1e-4 | L2 정규화 |
| **Batch Size** | 4 | GPU 메모리 제약 |
| **Accumulation Steps** | 4 | 실질 배치 크기 16 |
| **Scheduler** | ReduceLROnPlateau | F1 개선 없으면 LR 감소 (factor=0.5, patience=5) |
| **Max Grad Norm** | 1.0 | Gradient clipping |
| **Mixed Precision** | True | FP16 학습으로 속도 향상 |

### 📈 클래스별 Threshold 자동 탐색

#### Grid Search 기반 최적 Threshold 찾기
```python
# 각 클래스별로 독립적인 threshold 탐색
threshold_candidates = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

```

#### Threshold 저장 및 활용
- Best model checkpoint에 최적 threshold 저장
- Inference 시 저장된 threshold 활용
- 각 드럼 타입별 특성 반영 (hihat은 낮은 threshold 필요)

### 🛑 Early Stopping
- **Patience**: 10 epochs
- **Metric**: Validation F1 score (macro average)
- **목적**: 과적합 방지 및 학습 시간 절약

---

## 7. 평가 지표

### 📊 주요 성능 지표

| 지표 | 설명 | 계산 방식 |
|-----|------|---------|
| **F1 Score** | Precision과 Recall의 조화평균 | 2 * (P * R) / (P + R) |
| **Precision** | 예측한 onset 중 실제 onset 비율 | TP / (TP + FP) |
| **Recall** | 실제 onset 중 검출한 onset 비율 | TP / (TP + FN) |
| **Macro F1** | 3개 클래스 F1의 평균 | (F1_kick + F1_snare + F1_hihat) / 3 |

### 🎯 클래스별 F1 Score

학습 및 검증 과정에서 각 드럼 타입별로 개별 F1 score를 추적하여 클래스별 성능을 모니터링합니다.


## 8. 추론 파이프라인 (WAV → MIDI → PDF)

### 🔄 전체 워크플로우

```
1. WAV 파일 업로드
    ↓
2. 드럼 트랙 분리 (Demucs)
    ↓
3. 멜스펙트로그램 추출
    ↓
4. Sliding Window Inference
    ↓
5. Onset 검출 (Threshold 적용)
    ↓
6. 후처리 (Merge, Quantize)
    ↓
7. MIDI 생성 (PrettyMIDI)
    ↓
8. PDF 악보 생성 (MuseScore)
```

### 🎼 상세 단계 설명

#### Step 1-2: 전처리
- **Demucs**: 믹스된 오디오에서 드럼 트랙 분리 (optional)
- **BPM 검출**: `librosa.beat.beat_track`으로 템포 자동 감지
- **범위 보정**: BPM을 60~200 범위로 클리핑

#### Step 3-4: AI 추론
- **Sliding Window**:
  - Window size: 2000 프레임 (~23초)
  - Hop size: 1000 프레임 (50% overlap)
  - Overlap 영역은 평균을 취하여 경계 부드럽게 처리
- **Threshold 적용**:
  - kick: 0.50
  - snare: 0.50
  - hihat: 0.15 (민감도 높게 설정)

#### Step 5-6: 후처리

##### 1) Merge Nearby Events (50ms window)
```python
# 50ms 내에 발생한 동일 드럼 타입의 onset을 하나로 병합
# 목적: 떨림이나 미세한 지터 제거
for each drum_type:
    if onset[i+1] - onset[i] <= 50ms:
        merged_onset = 0.7 * onset[i] + 0.3 * onset[i+1]
```

##### 2) Enforce Minimum Gap
```python
# 각 드럼 타입별 최소 간격 강제
min_gap = {
    'kick': 80ms,   # 킥은 물리적으로 빠른 연타 어려움
    'snare': 60ms,  # 스네어도 어느 정도 간격 필요
    'hihat': 30ms   # 하이햇은 빠른 연주 가능
}
```

##### 3) Grid Quantization
```python
# BPM 기반 그리드에 맞춰 정렬
grid_interval = (60 / bpm) / (division / 4)
# kick/snare: 16분음표 그리드
# hihat: 8분음표 그리드

# bias를 추가하여 가까운 그리드로 스냅
quantized_time = floor((onset_time / grid_interval) + bias) * grid_interval
```

##### 4) Group Simultaneous Hits (30ms window)
```python
# 30ms 이내에 발생한 다른 드럼 타입을 동시타로 그룹화
# 예: kick+snare, kick+hihat+snare 등
```

#### Step 7: MIDI 생성
```python
# PrettyMIDI를 사용한 MIDI 파일 생성
midi_mapping = {
    'kick': 36,   # Bass Drum 1
    'snare': 38,  # Acoustic Snare
    'hihat': 42   # Closed Hi-Hat
}

# 각 onset을 MIDI Note로 변환
for (time, drum_types) in grouped_events:
    for drum_type in drum_types:
        Note(
            pitch=midi_mapping[drum_type],
            velocity=100,
            start=time,
            end=time + 0.1  # 100ms duration
        )
```

#### Step 8: PDF 악보 생성 (외부 도구)
- `music21`: MIDI → MusicXML 변환
- `MuseScore 3`: MusicXML → PDF 렌더링 (subprocess 호출)
- 드럼 악보 스타일: Percussion Clef, 'x' note heads

---

## 9. 기술 스택

### 🖥️ Backend
| 기술 | 버전 | 용도 |
|-----|------|------|
| Python | 3.9+ | 메인 개발 언어 |
| Flask | 2.x | REST API 서버 |
| PyTorch | 2.0+ | 딥러닝 프레임워크 |

### 🤖 AI/ML
| 라이브러리 | 용도 |
|-----------|------|
| **torch, torchaudio** | 모델 학습 및 추론 |
| **Librosa** | 오디오 처리, 멜스펙트로그램 추출, BPM 검출 |
| **pretty_midi** | MIDI 파일 생성 및 파싱 |
| **Demucs** | 음원 분리 (드럼 트랙 추출) |
| **NumPy, Pandas** | 데이터 처리 |

### 🎨 Frontend
| 기술 | 용도 |
|-----|------|
| React | UI 프레임워크 |
| Vite | 빌드 도구 |
| axios | HTTP 클라이언트 |

### 🎵 악보 생성
| 도구 | 용도 |
|-----|------|
| music21 | MIDI → MusicXML 변환 |
| MuseScore 3 | MusicXML → PDF 렌더링 |

### 🚀 배포
| 기술 | 용도 |
|-----|------|
| Docker | 컨테이너화 |
| nvidia/cuda:11.8.0 | GPU 지원 베이스 이미지 |

---

## 10. 프로젝트 구조

```
drum-sheet-generation/
│
├── backend/
│   ├── app/
│   │   ├── services/
│   │   │   ├── con_midi_maker.py        # 통합 추론 파이프라인
│   │   │   ├── demucs_separator.py      # 음원 분리
│   │   │   └── score_generator.py       # PDF 악보 생성
│   │   ├── routes/
│   │   │   └── convert.py               # API 엔드포인트
│   │   └── __init__.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.jsx
│   │   │   ├── ProgressDisplay.jsx
│   │   │   └── ResultViewer.jsx
│   │   └── App.jsx
│   └── package.json
│
├── model/
│   ├── BiGRU_model.py                   # 모델 아키텍처 정의
│   ├── BiGRU_datautilr.py               # 데이터 로더 및 전처리
│   ├── BiGRU_train.py                   # 학습 스크립트
│   ├── npy_maker_final.py               # 사전 계산 데이터 생성
│   └── checkpoints_final/
│       └── best_model.pt                # 학습된 모델 가중치
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── README.md
```

---

## 11. 주요 개선사항 및 기술적 도전

### ✅ 데이터 레벨 개선
1. **±1 프레임 확장 라벨**: onset 검출의 시간적 여유 제공
2. **SpecAugment 최적화**: time_mask 축소 (30→10)로 onset 보존
3. **Silent sample 증강**: False Positive 15% 감소
4. **Mel normalization**: 녹음 환경 변동 완화

### ✅ 모델 레벨 개선
1. **Conv-Conv-Pool 패턴**: 특징 추출 능력 향상
2. **GRU hidden size 증가**: 256→384로 표현력 강화
3. **Weighted Focal Loss**: 클래스 불균형 및 hard example 처리

### ✅ 학습 전략 개선
1. **클래스별 threshold 자동 탐색**: Grid search로 최적 threshold 발견
2. **Gradient accumulation**: 작은 배치 크기에서도 안정적 학습
3. **Mixed precision training**: 학습 속도 2배 향상

### 🚧 기술적 도전과제

1. **클래스 불균형**:
   - 문제: hihat이 kick/snare보다 3배 많음
   - 해결: Focal Loss + 클래스별 가중치

2.  **Onset 정렬 문제**:
   - 문제: MIDI와 오디오의 타이밍 미세 차이
   - 해결: ±1 프레임 확장 라벨

3. **시간 해상도 vs 메모리**:
   - 문제: hop_length를 줄이면 메모리 부족
   - 해결: hop_length=256으로 타협 + sliding window

---

## 12. 실행 방법

### 🛠️ 환경 설정

#### 1. Python 환경
```bash
# Python 3.9+ 필요
pip install -r requirements.txt
```

#### 2. 주요 의존성
```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
pretty-midi>=0.2.10
demucs>=4.0.0
flask>=2.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```


### 🌐 웹 애플리케이션 실행

#### Backend
```bash
cd backend
python -m flask run --host=0.0.0.0 --port=5000
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

### 🐳 Docker 실행
```bash
docker-compose up --build
```


## 13. 향후 개선 방향

### 🎯 단기 목표
- [ ] 다른 데이터셋 실험 (E-GMD 외 다른 데이터셋 테스트)
- [ ] Tom, Cymbal 등 드럼 타입 추가 (3-class → 5-class)
- [ ] Transformer 기반 모델 실험 (Self-attention으로 long-range dependency 강화)

### 🚀 중장기 목표
- [ ] End-to-end 학습 (오디오 → MIDI 직접 생성)
- [ ] Real-time 추론 최적화 (TorchScript, ONNX 변환)
- [ ] Multi-track 드럼 분리 (킥/스네어/하이햇 각각 분리된 오디오 생성)
- [ ] 모바일 앱 개발 (TFLite 또는 Core ML 변환)

---

## 14. 참고 문헌 및 데이터셋

### 📚 주요 논문
- Hawthorne. (2018). "Onsets and Frames: Dual-Objective Piano Transcription"
- Juan José Bosch. (2022). "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation"
- Michael Yeung. (2025). "Noise-to-Notes: Diffusion-based Generation and Refinement for Automatic Drum Transcription"

### 🗂️ 데이터셋
- **E-GMD v1.0.0**: Expanded Groove MIDI Dataset
  - Gillick, J., et al. (2019). "Learning to Groove with Inverse Sequence Transformations"
  - URL: https://magenta.tensorflow.org/datasets/e-gmd

### 🔧 오픈소스 라이브러리
- PyTorch: https://pytorch.org/
- Librosa: https://librosa.org/
- Demucs: https://github.com/facebookresearch/demucs
- PrettyMIDI: https://github.com/craffel/pretty-midi
- Music21: http://web.mit.edu/music21/

---

## 15. 라이센스

### 📄 라이센스
본 프로젝트는 교육 목적의 캡스톤 프로젝트입니다. 
