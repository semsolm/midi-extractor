## 백엔드 폴더 구조
___
```
backend/
├── .venv/                # 파이썬 가상환경
├── .flaskenv             # Flask 실행 환경 변수 (FLASK_APP, FLASK_ENV)
├── .gitignore            # Git 추적 제외 파일 목록
├── config.py             # 애플리케이션 환경설정
├── requirements.txt      # 프로젝트 의존성 라이브러리 목록
├── run.py                # 애플리케이션 실행 스크립트 (최종 진입점)
│
├── app/                  # --- 핵심 애플리케이션 소스 코드 ---
│   ├── __init__.py         # 애플리케이션 팩토리, 블루프린트 등록
│   ├── extensions.py       # Flask 확장 기능 초기화 (e.g., Celery)
│   │
│   ├── api/              # --- API 엔드포인트 및 로직 ---
│   │   ├── __init__.py     # API 블루프린트 생성
│   │   ├── routes.py       # URL 경로와 뷰 함수 매핑
│   │   └── schemas.py      # API 요청/응답 데이터 유효성 검사 (선택)
│   │
│   ├── services/         # --- 비즈니스 로직 (핵심 기능) ---
│   │   └── audio_service.py # 오디오 처리, MIDI 생성 등 실제 작업 수행
│   │
│   ├── tasks/            # --- 백그라운드 작업 (Celery) ---
│   │   └── audio_tasks.py  # 시간이 오래 걸리는 오디오 처리 작업을 정의
│   │
│   └── models/           # --- 서빙용 AI 모델 ---
│       └── kick_snare_lr.pkl
│
└── modeling/             # --- AI 모델 개발 및 실험 폴더 ---
    ├── data/             # --- 데이터 관리 ---
    │   ├── raw/          # 원본 데이터 (Kaggle, 직접 녹음한 wav 파일 등)
    │   ├── processed/    # 전처리된 데이터 (노이즈 제거, 길이 통일 등)
    │   └── features/     # 특징 벡터로 변환된 데이터 (.npy, .csv 등)
    │
    ├── notebooks/        # --- 데이터 탐색 및 실험용 노트북 ---
    │   ├── 1_data_exploration.ipynb  # EDA 및 데이터 시각화
    │   └── 2_model_prototyping.ipynb # 모델 구조 프로토타이핑
    │
    ├── src/              # --- 재사용 가능한 소스 코드 ---
    │   ├── __init__.py
    │   ├── data_utils.py # 데이터 로드, 증강, 전처리 함수
    │   └── features.py   # 특징 벡터 추출 함수 (model_train.py의 함수 분리)
    │
    ├── scripts/          # --- 실행 스크립트 ---
    │   ├── train.py      # 모델 학습 스크립트 (기존 model_train.py 역할)
    │   └── evaluate.py   # 학습된 모델 성능 평가 스크립트
    │
    └── outputs/          # --- 결과물 저장 ---
        ├── models/       # 학습된 모델 파일 (.pkl, .h5 등)
        └── reports/      # 성능 평가 결과 (혼동 행렬 이미지, 정확도 리포트 등)
```