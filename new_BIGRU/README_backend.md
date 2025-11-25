# 🥁 Drum Transcription Inference API

**드럼 WAV → MIDI 변환 파이프라인 (BiGRU 기반)**

이 문서는 백엔드 팀이 AI 모델을 서비스에 연결하는 데 필요한 모든 정보를 담고 있습니다.

---

## 📁 1. 구성 파일

추론에 필요한 파일 구조:

```
project/
├── model/
│   └── best_model.pt              # 학습된 BiGRU 드럼 모델
├── inference/
│   ├── MiDi_maker.py             # 전체 변환 파이프라인
│   └── BiGRU_model.py            # 모델 아키텍처 정의
└── requirements.txt               # Python 패키지 목록
```

---

## ⚙️ 2. 설치

### Python 버전
- **Python 3.10 이상** 권장

### 패키지 설치

```bash
pip install -r requirements.txt
```

### CUDA GPU 사용 시 (선택사항)

서버 환경에 맞는 PyTorch CUDA 버전 설치:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 3. 변환 함수 사용법

### 기본 사용법

```python
from pathlib import Path
from inference.MiDi_maker import drum_wav_to_midi, InferenceConfig

def convert_drum_to_midi(input_wav_path: str, output_dir: str = "outputs/"):
    """
    드럼 WAV 파일을 MIDI로 변환
    
    Args:
        input_wav_path: 입력 WAV 파일 경로
        output_dir: 출력 디렉토리 (기본값: outputs/)
    
    Returns:
        dict: MIDI 파일 경로와 로그 파일 경로
    """
    model_path = "model/best_model.pt"
    
    # 설정 객체 생성
    config = InferenceConfig()
    
    # 변환 실행
    drum_wav_to_midi(
        wav_path=input_wav_path,
        model_path=model_path,
        output_dir=output_dir,
        config=config
    )
    
    # 출력 파일 경로
    base_name = Path(input_wav_path).stem
    
    return {
        "midi_path": f"{output_dir}/{base_name}_drums.mid",
        "log_path": f"{output_dir}/{base_name}_drums.txt",
        "status": "success"
    }
```

### 실행 예시

```python
# 단일 파일 변환
result = convert_drum_to_midi("audio/song.wav")
print(f"MIDI 파일: {result['midi_path']}")
print(f"로그 파일: {result['log_path']}")
```

---

## 📤 4. FastAPI/Flask 연동 예시

### FastAPI 예시

```python
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import os

from inference.MiDi_maker import drum_wav_to_midi, InferenceConfig

app = FastAPI()

# 설정 객체는 한 번만 생성 (성능 최적화)
config = InferenceConfig()

# 임시 파일 및 출력 디렉토리
TEMP_DIR = "temp/"
OUTPUT_DIR = "outputs/"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/api/convert")
async def convert_drum(file: UploadFile):
    """
    드럼 WAV 파일을 MIDI로 변환하는 API
    
    Args:
        file: 업로드된 WAV 파일
    
    Returns:
        dict: MIDI 파일 경로와 메타데이터
    """
    # 파일 검증
    if not file.filename.endswith(('.wav', '.WAV')):
        raise HTTPException(status_code=400, detail="WAV 파일만 지원합니다.")
    
    # 임시 파일 저장
    input_path = f"{TEMP_DIR}/{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 변환 실행
        drum_wav_to_midi(
            wav_path=input_path,
            model_path="model/best_model.pt",
            output_dir=OUTPUT_DIR,
            config=config
        )
        
        base_name = Path(input_path).stem
        midi_filename = f"{base_name}_drums.mid"
        log_filename = f"{base_name}_drums.txt"
        
        return {
            "status": "success",
            "midi_url": f"/outputs/{midi_filename}",
            "log_url": f"/outputs/{log_filename}",
            "filename": midi_filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"변환 중 오류 발생: {str(e)}")
    
    finally:
        # 임시 파일 삭제
        if os.path.exists(input_path):
            os.remove(input_path)

@app.get("/outputs/{filename}")
async def download_file(filename: str):
    """
    변환된 파일 다운로드
    """
    file_path = f"{OUTPUT_DIR}/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    return FileResponse(file_path, filename=filename)

@app.get("/health")
async def health_check():
    """
    서버 상태 체크
    """
    return {"status": "healthy", "model": "BiGRU Drum Transcription"}
```

### Flask 예시

```python
from flask import Flask, request, send_file, jsonify
from pathlib import Path
import os

from inference.MiDi_maker import drum_wav_to_midi, InferenceConfig

app = Flask(__name__)
config = InferenceConfig()

OUTPUT_DIR = "outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/api/convert', methods=['POST'])
def convert_drum():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400
    
    file = request.files['file']
    if not file.filename.endswith(('.wav', '.WAV')):
        return jsonify({"error": "WAV 파일만 지원합니다."}), 400
    
    # 임시 저장
    input_path = f"temp/{file.filename}"
    file.save(input_path)
    
    try:
        drum_wav_to_midi(
            wav_path=input_path,
            model_path="model/best_model.pt",
            output_dir=OUTPUT_DIR,
            config=config
        )
        
        base_name = Path(input_path).stem
        midi_filename = f"{base_name}_drums.mid"
        
        return jsonify({
            "status": "success",
            "midi_url": f"/outputs/{midi_filename}",
            "filename": midi_filename
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

@app.route('/outputs/<filename>')
def download_file(filename):
    return send_file(f"{OUTPUT_DIR}/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## ✔️ 5. 출력 파일 구조

변환 완료 시 총 **2개의 파일**이 생성됩니다:

```
outputs/
├── song_drums.mid   ← MIDI 악보
└── song_drums.txt   ← 디버그용 분석 로그
```

### MIDI 파일 (`_drums.mid`)
- General MIDI Drum Map 사용
- DAW 및 악보 프로그램에서 바로 재생 가능
- 드럼 매핑:
  - Kick (Bass Drum): Note 36
  - Snare: Note 38
  - Hi-hat: Note 42

### 로그 파일 (`_drums.txt`)
- 디버깅 및 품질 분석 용도
- 타임스탬프별 드럼 타격 정보
- BPM, 양자화 설정 등 메타데이터 포함

---

## 🎛 6. 주요 옵션 수정 (선택사항)

### 드럼별 그리드 조정

```python
config = InferenceConfig()

# 양자화 그리드 설정 (분음표 단위)
config.grid_division['kick'] = 16   # 16분음표
config.grid_division['snare'] = 16  # 16분음표
config.grid_division['hihat'] = 8   # 8분음표
```

### 임계값 조정

```python
# 드럼 타입별 검출 임계값 (0~1 범위)
config.thresholds['kick'] = 0.5    # 높을수록 엄격
config.thresholds['snare'] = 0.5
config.thresholds['hihat'] = 0.15  # 하이햇은 낮게 설정
```

### BPM 수동 지정

```python
# BPM 자동 감지 대신 수동 설정
drum_wav_to_midi(
    wav_path="input.wav",
    model_path="model/best_model.pt",
    output_dir="outputs/",
    config=config,
    bpm_override=120  # 수동으로 120 BPM 지정
)
```

---

## 🧪 7. 입력/출력 조건

### 입력 WAV 조건

| 항목 | 지원 범위 |
|------|-----------|
| 채널 | 모노/스테레오 모두 가능 |
| 비트 깊이 | 16bit/24bit/32bit float |
| 샘플레이트 | 자동 정규화 (22050Hz) |
| 길이 | 제한 없음 |
| 파일 크기 | 메모리 허용 범위 내 |

### 출력 MIDI 조건

| 항목 | 설정값 |
|------|--------|
| 포맷 | General MIDI |
| Drum Map | GM Drum Kit |
| Velocity | 100 (고정) |
| Note Duration | 0.1초 (고정) |

---

## ⚠️ 8. 자주 발생하는 문제

### ❌ PyTorch CUDA mismatch

**증상:**
```
RuntimeError: CUDA error: no kernel image is available
```

**해결:**
- GPU 서버 환경에 맞는 PyTorch 버전 설치 필요
- CUDA 버전 확인: `nvidia-smi`
- 해당 CUDA 버전에 맞는 torch 설치

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### ❌ pretty_midi 오류

**증상:**
```
OSError: [Errno 2] No such file or directory
```

**해결:**
- 출력 디렉토리가 없는 경우 발생
- `os.makedirs(output_dir, exist_ok=True)` 이미 처리됨
- 권한 문제 확인 필요

---

### ❌ librosa 설치 오류

**증상:**
```
ERROR: Could not build wheels for numba
```

**해결:**
- Python 3.10 이하 버전 사용 권장
- 또는 numba 수동 설치:

```bash
pip install numba==0.56.4
pip install librosa
```

---

### ❌ 메모리 부족

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결:**
- CPU 모드로 전환 (자동)
- 긴 오디오 파일은 청크 단위로 처리 (내부 구현됨)

---

## 🎉 9. 완성

이 패키지를 사용하면:
- ✅ 단일 API 호출로 WAV → MIDI 변환 가능
- ✅ 모델 로딩, Mel 변환, 슬라이딩 윈도우 추론 자동 수행
- ✅ 후처리, 그리드 양자화까지 자동 처리
- ✅ 프로덕션 레벨의 안정성 보장

---

## 📞 10. 문의

기술적 문제나 질문이 있으면 AI 팀에 문의하세요.