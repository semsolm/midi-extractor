# 🚀  다른 PC에서 백엔드 시연을 위한 설정 가이드라인
## Step 1: 사전 준비 (PC 자체에 설치)
* Python 3.10 (또는 3.9) 설치
* NVIDIA 그래픽 드라이버 (RTX 3050용 최신 버전)
* FFmpeg: Demucs가 오디오 파일을 처리하기 위해 필수적입니다.

## Step 2: 프로젝트 파일 복사
* 원본 PC에서 프로젝트 폴더 전체 (또는 최소한 backend 폴더 전체)를 시연용 PC로 복사
* AI 모델 원본: ```backend/modeling/outputs/models/drum_cnn_final.keras```,
```backend/modeling/scripts/convert_model_to_lite.py```
## Step 3: 가상 환경(venv) 생성 및 활성화
* 1. backend 폴더로 이동
```cd C:\경로\to\midi-extractor-dev\backend```
* 2. .venv 이름의 가상 환경 생성
```python -m venv .venv```
* 3. 가상 환경 활성화 (Windows)
```.\.venv\Scripts\activate```
## (터미널 프롬프트 앞에 (.venv)가 표시되면 성공)
Step 4: GPU PyTorch 설치 (속도 문제 해결)
```venv```가 활성화된 상태에서, requirements.txt보다 먼저 GPU PyTorch를 수동으로 설치합니다.
```pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121```
## Step 5: TFLite 모델 생성 (FileNotFoundError 해결)
서버가 사용할 .tflite 모델 파일을 생성합니다. (이 작업은 무거운 tensorflow 패키지가 필요합니다.)
* (venv 활성화 상태여야 함)
* 모델 변환에 필요한 전체 tensorflow 설치
```pip install tensorflow```
* 변환 스크립트가 있는 폴더로 이동
```cd modeling\scripts```
* 변환 스크립트 실행 (```drum_cnn_final.keras``` -> ```drum_cnn_final.tflite```)
```python convert_model_to_lite.py``` (성공 메시지가 뜨면 ```backend/app/models/drum_cnn_final.tflite``` 파일이 생성됨)
* (선택) 무거운 ```tensorflow``` 패키지 삭제 (경량화를 위해 권장)
```pip uninstall tensorflow -y```
* 다시 ```backend``` 루트 폴더로 복귀
```cd ..\..```
## Step 6: 최종 라이브러리 설치
* ```venv``` 활성화 상태, ```backend``` 폴더인지 확인
* ```backend/requirements.txt``` 파일로 설치
- ```pip install -r requirements.txt```
- ```demucs```는 이미 설치된 GPU 버전의 torch를 자동으로 감지합니다.
```tflite-runtime```은 ```tensorflow```와 충돌하지 않는 **경량 런타임**입니다.
## Step 7: 시연 실행
* [터미널 1] 서버 실행:
venv가 활성화된 backend 폴더에서 서버를 시작합니다.
```python run.py```
(Running on http://127.0.0.1:5000) 메시지를 확인합니다.

* [터미널 2] 클라이언트 실행:
venv가 활성화된 backend 폴더에서 별도의 새 터미널을 엽니다.
테스트 클라이언트를 실행합니다.
```python local_test_client.py```