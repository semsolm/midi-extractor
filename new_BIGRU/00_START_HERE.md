📦 백엔드 배포 패키지 - 최종 전달 사항
AI 팀에서 백엔드 팀으로 전달하는 드럼 트랜스크립션 모델 패키지입니다.

📋 전달 파일 목록
drum_transcription_package/
│
├── 📄 README_backend.md              # ⭐ 메인 문서 - 여기부터 읽으세요!
├── 📄 DEPLOYMENT_CHECKLIST.md        # 배포 전 체크리스트
├── 📄 requirements.txt               # Python 패키지 목록
├── 📄 example_usage.py               # 사용 예시 코드
│
├── model/
│   └── best_model.pt                 # 학습된 BiGRU 모델 (별도 전달)
│
└── inference/
    ├── MiDi_maker.py                 # 메인 파이프라인
    └── BiGRU_model.py                # 모델 아키텍처

🚀 빠른 시작 (Quick Start)
1단계: 환경 설정
bashpip install -r requirements.txt
2단계: 파일 배치
프로젝트 디렉토리/
├── model/best_model.pt         # AI 팀에서 별도 전달
├── inference/MiDi_maker.py     # 포함됨
└── inference/BiGRU_model.py    # 포함됨
3단계: 테스트
pythonfrom inference.MiDi_maker import drum_wav_to_midi, InferenceConfig

config = InferenceConfig()
drum_wav_to_midi(
    wav_path="test.wav",
    model_path="model/best_model.pt",
    output_dir="outputs/",
    config=config
)
4단계: API 연동
README_backend.md의 FastAPI/Flask 예시 참고

🎯 핵심 기능
✅ 지원하는 기능

WAV → MIDI 자동 변환
BPM 자동 감지
드럼 3종 (Kick, Snare, Hi-hat) 검출
General MIDI 포맷 출력
GPU/CPU 자동 선택
배치 처리 지원

📊 입력 조건

포맷: WAV (모노/스테레오)
비트 깊이: 16/24/32 bit
샘플레이트: 자동 정규화
길이: 제한 없음

📤 출력 형식

MIDI 파일 (General MIDI)
텍스트 로그 (디버깅용)


⚙️ 시스템 요구사항
최소 사양

Python: 3.10 이상
RAM: 4GB 이상
디스크: 2GB 이상 여유 공간

권장 사양

Python: 3.10
GPU: NVIDIA GPU (CUDA 11.8+)
VRAM: 4GB 이상
RAM: 8GB 이상


🔧 커스터마이징
주요 설정 변경 가능 항목
pythonconfig = InferenceConfig()

# 1. 드럼별 양자화 그리드
config.grid_division['kick'] = 16    # 16분음표
config.grid_division['snare'] = 16
config.grid_division['hihat'] = 8    # 8분음표

# 2. 검출 임계값
config.thresholds['kick'] = 0.5
config.thresholds['snare'] = 0.5
config.thresholds['hihat'] = 0.15

# 3. BPM 수동 설정
drum_wav_to_midi(..., bpm_override=120)

📞 연락처 및 지원
기술 지원

AI 팀 담당자: [이메일/연락처]
문서 관련: README_backend.md 참고
버그 리포트: [이슈 트래커 링크]

업데이트

모델 버전: v1.0 (2024-11-24)
최종 수정: 2024-11-24
다음 업데이트: TBD


⚠️ 중요 사항

모델 파일 (best_model.pt)은 별도로 전달됩니다

파일 크기: 약 50-100MB
위치: model/best_model.pt
Git에 포함하지 마세요 (용량 문제)


API 키나 인증 불필요

로컬 추론 모델
외부 API 호출 없음
네트워크 연결 불필요


라이선스 확인

내부 사용 전용
상업적 사용 시 AI 팀 문의




✅ 체크리스트
백엔드 팀이 확인해야 할 사항:

 requirements.txt 설치 완료
 model/best_model.pt 파일 확인
 간단한 테스트 실행 성공
 API 연동 코드 작성 완료
 에러 핸들링 구현
 로깅 시스템 구축
 프로덕션 배포 준비 완료


🎉 배포 완료 후
배포가 완료되면 AI 팀에 알려주세요:

배포 환경 (Dev/Staging/Prod)
테스트 결과
성능 지표 (처리 시간, 성공률 등)
발견된 이슈나 개선 사항


준비 완료되셨나요? README_backend.md를 먼저 읽어보세요! 📖