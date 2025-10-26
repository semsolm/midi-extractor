# backend/local_test_client.py
import requests
import time
import os

# --- 설정 ---
# 백엔드 서버의 파일 업로드 API 주소
UPLOAD_URL = "http://127.0.0.1:5000/api/process"
# 서버로 보낼 로컬 MP3 파일 경로 (예: backend 폴더에 있는 drum.mp3)
FILE_PATH = "drum.mp3"

# --- 스크립트 실행 ---
if not os.path.exists(FILE_PATH):
    print(f"오류: 파일 '{FILE_PATH}'를 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    try:
        # 1. 파일을 'multipart/form-data' 형식으로 서버에 POST 요청 보내기
        print(f"'{FILE_PATH}' 파일을 서버로 업로드합니다...")
        with open(FILE_PATH, 'rb') as f:
            files = {'audio_file': (os.path.basename(FILE_PATH), f, 'audio/mpeg')}
            response = requests.post(UPLOAD_URL, files=files)
            response.raise_for_status()  # 오류가 있으면 예외 발생

        # 2. 서버로부터 작업 ID (jobId) 받기
        result = response.json()
        job_id = result.get('jobId')

        if not job_id:
            print("오류: 서버로부터 작업 ID를 받지 못했습니다.")
            print("서버 응답:", result)
        else:
            print(f"파일 업로드 성공! 작업 ID: {job_id}")

            # 3. 작업이 완료될 때까지 주기적으로 결과 확인
            result_url = f"http://127.0.0.1:5000/api/result/{job_id}"
            while True:
                print("서버에서 결과를 확인하는 중...")
                result_response = requests.get(result_url)
                status_result = result_response.json()
                status = status_result.get('status')
                message = status_result.get('message')

                print(f"  - 상태: {status}")
                print(f"  - 메시지: {message}")

                if status == 'completed':
                    print("\n🎉 작업 완료! 최종 결과:")
                    print(status_result.get('results'))
                    break
                elif status == 'error':
                    print("\n❌ 작업 중 오류가 발생했습니다.")
                    break

                time.sleep(5)  # 5초 대기 후 다시 확인

    except requests.exceptions.RequestException as e:
        print(f"\n서버 요청 중 오류가 발생했습니다: {e}")
        print("백엔드 서버(run.py)가 실행 중인지 확인해주세요.")