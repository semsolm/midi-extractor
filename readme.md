# 🎵 간편 음원 추출기 (Simple Audio Extractor)

회원가입 절차 없이 파일을 업로드하면, 해당 영상 파일에서 음원(MP3)을 자동으로 추출해 주는 웹 애플리케이션입니다.

## ✨ 주요 기능

* **파일 업로드:** 사용자 친화적인 웹 인터페이스를 통해 MP4, AVI 등 다양한 영상 파일을 업로드할 수 있습니다.
* **자동 음원 추출:** 업로드된 영상에서 오디오를 자동으로 분리하여 MP3 파일로 변환합니다.
* **간편 다운로드:** 추출이 완료된 음원을 즉시 다운로드할 수 있는 링크를 제공합니다.

## 🚀 시작하기

### **1. 필수 설치**

이 프로그램을 실행하려면 Python과 FFmpeg이 필요합니다.

* **Python 3.6 이상**
* **FFmpeg:** [공식 웹사이트](https://ffmpeg.org/download.html)에서 운영체제에 맞는 FFmpeg을 다운로드하여 시스템 PATH에 추가해야 합니다.

### **2. 프로젝트 설치**

프로젝트를 로컬 컴퓨터에 복제(Clone)하고 필요한 라이브러리를 설치합니다.

```bash
# GitHub에서 프로젝트 복제 (만약 GitHub에 올린다면)
# git clone [프로젝트_레포지토리_URL]
# cd [프로젝트_폴더_이름]

# 필요한 Python 라이브러리 설치
pip install -r requirements.txt
requirements.txt 파일 내용:

Flask
moviepy
werkzeug
3. 애플리케이션 실행
터미널에서 다음 명령어를 실행하여 웹 서버를 시작합니다.

Bash

python app.py
서버가 실행되면, 웹 브라우저를 열고 http://127.0.0.1:5000에 접속하세요.

🛠️ 기술 스택
백엔드: Python, Flask

오디오 처리: MoviePy (FFmpeg 기반)

프로젝트 구조
/audio_extractor/
├── app.py              # 메인 애플리케이션
├── templates/          # HTML 템플릿 파일
├── uploads/            # 업로드 파일 임시 저장 폴더
└── extracted/          # 추출된 오디오 파일 저장 폴더
