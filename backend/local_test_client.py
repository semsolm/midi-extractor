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
            print("서버 상태를 1초마다 확인합니다...")

            # 3. [수정] 절차(메시지)가 변경될 때만 출력
            result_url = f"http://127.0.0.1:5000/api/result/{job_id}"
            
            # [추가] 마지막으로 출력된 메시지를 저장할 변수
            last_message = ""
            
            while True:
                result_response = requests.get(result_url)
                status_result = result_response.json()
                
                status = status_result.get('status')
                message = status_result.get('message', '')

                # [수정] 메시지가 마지막 메시지와 다를 경우에만 새로 출력
                if message != last_message:
                    # 'Separating:' 또는 'MIDI 노트 변환 중:' 같은 진행도 바 메시지는 건너뜀
                    if not message.startswith("Separating:") and not message.startswith("MIDI 노트 변환 중"):
                        print(f"  -> {message}")  # \r (덮어쓰기) 대신 \n (새 줄)로 출력
                        last_message = message    # 마지막 메시지 업데이트

                if status == 'completed':
                    print("\n🎉 작업 완료! 최종 결과:")
                    print(status_result.get('results'))
                    break
                elif status == 'error':
                    # 오류 메시지는 위에서 출력되었을 수 있으므로 확인 후 출력
                    if message != last_message:
                        print(f"  -> {message}")
                    print("\n❌ 작업 중 오류가 발생했습니다.")
                    break

                time.sleep(1)  # 1초 대기 (서버 상태 확인 주기)

    except requests.exceptions.RequestException as e:
        print(f"\n서버 요청 중 오류가 발생했습니다: {e}")
        print("백엔드 서버(run.py)가 실행 중인지 확인해주세요.")
    except KeyboardInterrupt:
        print("\n사용자에 의해 테스트가 중지되었습니다.")
        sys.exit(0)