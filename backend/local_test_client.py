# backend/local_test_client.py
import requests
import time
import os
import sys

# --- 설정 ---
UPLOAD_URL = "http://127.0.0.1:5000/api/process"
FILE_PATH = "drum.mp3"

# --- 스크립트 실행 ---
if not os.path.exists(FILE_PATH):
    print(f"오류: 파일 '{FILE_PATH}'를 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    try:
        # 1. 파일 업로드
        print(f"'{FILE_PATH}' 파일을 서버로 업로드합니다...")
        with open(FILE_PATH, 'rb') as f:
            files = {'audio_file': (os.path.basename(FILE_PATH), f, 'audio/mpeg')}
            response = requests.post(UPLOAD_URL, files=files)
            response.raise_for_status()

        # 2. 작업 ID 받기
        result = response.json()
        job_id = result.get('jobId')

        if not job_id:
            print("오류: 서버로부터 작업 ID를 받지 못했습니다.")
            print("서버 응답:", result)
        else:
            print(f"파일 업로드 성공! 작업 ID: {job_id}")

            # 3. [수정] 'message' 필드를 가져와 한 줄에 덮어쓰기
            result_url = f"http://127.0.0.1:5000/api/result/{job_id}"
            
            # [추가] 터미널 너비에 맞게 공백 패딩
            terminal_width = os.get_terminal_size().columns
            
            while True:
                result_response = requests.get(result_url)
                status_result = result_response.json()
                status = status_result.get('status')
                message = status_result.get('message', '') # 메시지 가져오기

                # [수정] \r로 줄의 시작으로 이동하고, 메시지를 출력한 뒤 공백으로 덮어씀
                padding = " " * (terminal_width - len(message) - 1)
                print(f"\r{message}{padding}", end="", flush=True)

                if status == 'completed':
                    # [수정] \n\n으로 줄바꿈 후 최종 결과 출력
                    print(f"\n\n🎉 작업 완료! 최종 결과:")
                    print(status_result.get('results'))
                    break
                elif status == 'error':
                    # [수정] \n\n으로 줄바꿈 후 오류 메시지 출력
                    print(f"\n\n❌ 작업 중 오류가 발생했습니다. (메시지: {message})")
                    break

                time.sleep(1)  # 1초 대기

    except requests.exceptions.RequestException as e:
        print(f"\n서버 요청 중 오류가 발생했습니다: {e}")
        print("백엔드 서버(run.py)가 실행 중인지 확인해주세요.")
    except KeyboardInterrupt:
        print("\n사용자에 의해 테스트가 중지되었습니다.")
        sys.exit(0)