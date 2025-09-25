# 🎵 간편 음원 추출기 (MIDI-Extractor)

## 📌 프로젝트 개요
***
| 항목 | 내용 |
|------|------|
| 개요 | 사용자가 업로드한 음악 파일(mp3)에서 **드럼 소리를 추출**하고, **AI 모델을 활용하여 드럼을 분류**한 뒤, **MIDI 파일 및 악보(PDF)로 변환**해주는 웹 애플리케이션 |
| 대상 | 드럼을 학습하거나 연습하는 학생, 동아리 활동자, 취미 연주자 등으로 웹 환경에서 간편하게 활용 가능 |

---

## 🧑‍💻 팀 구성
***
| 이름  | 학번        | 역할(TODO) | Github(TODO) | 비고 |
|-------|-----------|------------|--------------|------|
| 윤상일 | 2020E7424 | 추가예정   | 추가예정     |      |
| 양태양 | 2021E7411 | 추가예정   | 추가예정     |      |
| 최유진 | 2023E7518 | 추가예정   | 추가예정     |      |
| 이준행 | 2020E7427 | 추가예정   | 추가예정     |      |
| 정서영 | 2020U2329 | 추가예정   | 추가예정     |      |

---

## ✨ 주요 기능
***
1. **파일 업로드**  
   - 드래그 앤 드롭 또는 파일 업로드 버튼을 통해 mp3 파일 업로드 가능  

2. **드럼 소리 추출**  
   - 소스 분리(Source Separation) 기법을 활용해 드럼 트랙 자동 추출  

3. **드럼 분류 모델**  
   - 추출된 드럼 사운드를 Kick / Snare / Hi-hat 등으로 분류  
   - 2D CNN 기반 분류 모델 적용  

4. **MIDI 변환**  
   - 분류된 드럼 이벤트를 MIDI 파일로 변환하여 다운로드 제공  

5. **악보 시각화**  
   - MuseScore API 또는 PDF 생성기를 활용해 MIDI 파일을 시각화  
   - 드럼 악보 형태로 PDF 제공  

---

## 🛠️ 기술 스택
***
- **Backend:** Python, Flask  
- **Frontend:** HTML, CSS, JavaScript (TODO: 프레임워크 사용 시 추가)  
- **Audio Processing:** FFmpeg, MoviePy  
- **AI/ML:** 2D CNN (TODO: 사용 라이브러리 기입 예정 - TensorFlow / PyTorch 등)  
- **Visualization:** MuseScore API / ReportLab (PDF 생성기)  

---

## 🚀 시작하기

### 1. 필수 설치
- **Python 3.6 이상**  
- **FFmpeg** → [공식 웹사이트](https://ffmpeg.org/download.html)에서 다운로드 후 시스템 PATH에 추가  

### 2. 프로젝트 설치
```
# 프로젝트 복제
git clone [프로젝트_레포지토리_URL]
cd [프로젝트_폴더_이름]

# 필요한 Python 라이브러리 설치
pip install -r requirements.txt
```
### ➡ 브라우저에서 http://127.0.0.1:5000 접속

## 📂 프로젝트 구조
```
/audio_extractor/
├── app.py              # 메인 애플리케이션
├── templates/          # HTML 템플릿
├── uploads/            # 업로드된 파일
└── extracted/          # 추출 결과 파일
```

## 🔮 향후 계획 (TODO)
* 회원별 작업 관리 기능
* GPU 기반 모델 가속
* UI 개선 (React.js / Vue.js 적용)
* 드럼 패턴 편집 기능 추가