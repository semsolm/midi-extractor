# 🥁 드럼 사운드 자동 분류 및 악보 생성 AI 시스템

## 1. 팀 정보  
- **팀명:** 경로당
  
| 팀원  | 학번 | 역할 | Github                                               | 비고        |
|-----|------|------|------------------------------------------------------|-----------|
| 윤상일 | 2020E7424 | AI/ML | [@semsolm](https://github.com/semsolm)               | 모델 설계 및 학습 |
| 양태양 | 2021E7411 | Frontend | [-]()                                                | UI 개발     |
| 최유진 | 2023E7518 | Frontend | [@cyj4795](https://github.com/cyj4795)                      | UI 디자인    |
| 이준행 | 2020E7427 | Backend | [@LeopoldBloom2K](https://github.com/LeopoldBloom2K) |           |
| 정서영 | 2020U2329 | Backend | [-]()                                                |           |

---

## 2. 프로젝트 개요  

### 🔹 프로젝트 제목  
**드럼 사운드 자동 분류 및 악보 생성 AI 시스템**

### 🔹 프로젝트 목적  
음악 녹음 파일에서 드럼 소리를 사람이 직접 듣고 악보로 옮기는 과정은 많은 시간과 노력이 필요합니다.  
본 프로젝트는 **드럼 소리를 자동으로 인식 및 분류(Kick, Snare, Hi-hat)** 하여 **MIDI 형태로 변환 및 악보를 자동 생성**하는 AI 기반 시스템을 개발하는 것을 목표로 합니다.  

---

## 3. 데이터 수집 계획 및 활동 내용  

### 📘 (1) 데이터 수집 계획

| 구분 | 내용 |
|------|------|
| **데이터 종류** | 오디오 데이터 (드럼 타격 소리: 킥, 스네어, 하이햇) |
| **데이터 출처** | 직접 수집 (스마트폰 녹음) + Kaggle “Drum Kit Sound Samples” |
| **수집 방법** | 스마트폰 녹음기로 드럼 타격음(2초) 녹음 → 라벨링 및 분류 |
| **수집 기간** | 10월 중순 (1~2주 예정) |
| **데이터 양 (예상)** | 각 클래스별 30개 (Kick, Snare, Hi-hat) + 증강 데이터 |
| **라이선스** | Kaggle: 비상업적 연구 가능 / 직접 녹음 데이터: 저작권 문제 없음 |

#### 🎧 예시 데이터 구조

| 파일명 | 라벨 | 길이(s) |
|--------|------|----------|
| Snare1.wav | snare | 1 |

---

### 📘 (2) 데이터 수집 활동 현황  

| 사운드 | Kick | Snare | Hi-hat |
|---------|-------|--------|--------|
| **개수** | 30개 | 30개 | 30개 |

**데이터 출처:**  
- [Kaggle Drum Kit Sound Samples](https://www.kaggle.com/datasets/anubhavchhabra/drum-kit-sound-samples?select=overheads)  
- 직접 녹음 데이터 (하이햇 오픈·클로즈, 세미오픈, 크레센도 등)

**수집 중 어려움:**  
- 드럼 데이터셋의 불균형 (Kick/ Snare 중심)  
- 녹음 품질 차이로 인한 음량 편차 및 노이즈  
➡ 직접 녹음과 데이터 증강(노이즈 추가, 속도/톤 변형, 리버브 효과 등)으로 보완 예정  

---

## 4. 데이터 전처리 및 EDA 계획  

| 항목 | 내용 |
|------|------|
| **예상 문제** | 녹음 환경·장비 차이 → 음량 편차, 노이즈 발생 가능 |
| **전처리 도구** | `librosa`, `soundfile`, `audiomentations`, `pydub`, `numpy` |
| **주요 전처리 단계** | <ul><li>사운드 파일 포맷 통일</li><li>앞·뒤 공백 제거 (`librosa.effects.trim`)</li><li>길이 통일 (150ms)</li><li>볼륨 정규화</li><li>데이터 증강</li></ul> |
| **EDA 목적** | 각 드럼 클래스별 음향적 특성(파형, 주파수 대역, 패턴) 시각화를 통해 이상치 식별 및 데이터 품질 개선 |
| **시각화 도구** | `librosa.display.waveshow`, `librosa.display.specshow`, `matplotlib` |

---

## 5. 기술 스택  

| 구분 | 기술 | 설명 |
|------|------|------|
| **Backend** | Python, Flask | 서버 및 API 개발 |
| **Frontend** | HTML, CSS, JavaScript | 사용자 인터페이스 |
| **Audio Processing** | FFmpeg, MoviePy, Librosa | 오디오 추출 및 변환 |
| **AI/ML** | 2D CNN (TensorFlow / PyTorch 예정) | 드럼 사운드 분류 |
| **Visualization** | MuseScore API, ReportLab | MIDI 시각화 및 PDF 생성 |

---

## 6. 기타 참고사항  

- **참고 데이터셋:**  
  [Kaggle Drum Kit Sound Samples](https://www.kaggle.com/datasets/anubhavchhabra/drum-kit-sound-samples?select=overheads)  
- **참고 논문 및 자료:**  
  - Drum Sound Recognition using CNN (2020)  
  - Automatic Drum Transcription using Spectrogram Analysis (ISMIR 2019)

---

## 7. 향후 계획  
- 데이터 증강 및 모델 학습 (10월 말)  
- 모델 정확도 검증 및 시각화 (11월 초)  
- Flask 서버 및 프론트엔드 통합 테스트 (11월 중순)  
- MuseScore API 기반 악보 자동 생성 및 PDF 출력 (11월 말 완성 목표)
