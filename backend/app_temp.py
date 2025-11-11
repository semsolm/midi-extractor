import os
import time
import uuid
import threading
from io import BytesIO
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)

# -------------------------------------------------------------
# CORS 설정: 프론트엔드(localhost:5173 등)의 요청을 허용합니다.
# -------------------------------------------------------------
CORS(app, expose_headers=['Content-Disposition'])

# -------------------------------------------------------------
# ⚠️ 테스트용 PDF 파일 경로 (사용자 요청에 따라 절대 경로 사용)
# 이 스크립트를 실행하기 전에, 이 경로에 'test.pdf' 파일을 준비해야 합니다.
# -------------------------------------------------------------
PDF_FILE_PATH = 'C:\\testwork\\backend\\test.pdf'

# -------------------------------------------------------------
# 메모리 내 작업 상태 저장소 (tasks.py의 'jobs' 딕셔너리 역할)
# -------------------------------------------------------------
jobs = {}

# -------------------------------------------------------------
# 백그라운드 작업 시뮬레이션 함수 (audio_processor.py 역할)
# -------------------------------------------------------------
def simulate_work(job_id):
    """
    AI 처리를 시뮬레이션하며 2초마다 상태를 업데이트합니다.
    """
    print(f"[{job_id}] 백그라운드 작업 시작...")
    
    # 1. '드럼 분리 중' 상태 (실제 audio_processor.py 메시지 모방)
    time.sleep(2)
    jobs[job_id] = {
        "status": "processing",
        "message": "드럼 분리 중... 25%" 
    }
    print(f"[{job_id}] 상태 업데이트: 드럼 분리 중")

    # 2. 'MIDI 노트 변환 중' 상태 (실제 audio_processor.py 메시지 모방)
    time.sleep(2)
    jobs[job_id] = {
        "status": "processing",
        "message": "MIDI 노트 변환 중... 75%"
    }
    print(f"[{job_id}] 상태 업데이트: MIDI 변환 중")

    # 3. '완료' 상태 (프론트엔드가 기대하는 최종 결과 JSON)
    time.sleep(2)
    jobs[job_id] = {
        "status": "completed",
        "message": "작업이 완료되었습니다.",
        "results": {
            "midiUrl": f"/download/midi/{job_id}",
            "pdfUrl": f"/download/pdf/{job_id}"
        }
    }
    print(f"[{job_id}] 작업 완료.")

# -------------------------------------------------------------
# 엔드포인트 1: 파일 업로드 및 작업 시작 (routes.py의 /api/process)
# -------------------------------------------------------------
@app.route('/api/process', methods=['POST'])
def process_audio_route():
    """
    (frontend/src/services/api.js의 uploadAudioFile이 호출)
    파일을 받고, 작업 ID를 즉시 반환하며, 백그라운드 스레드를 시작합니다.
    """
    # 프론트엔드가 'audio_file' 키로 파일을 보냅니다. [cite: sunning838/testwork/testwork-f405721f350910119482f7b5a4b2e4d52743c502/frontend/src/services/api.js]
    if 'audio_file' not in request.files:
        return jsonify({"error": "오디오 파일이 없습니다."}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    # 1. 고유 작업 ID 생성
    job_id = str(uuid.uuid4())
    print(f"[{job_id}] 새 작업 요청 수신: {file.filename}")

    # 2. 작업 초기 상태 설정 (StatusTracker.jsx가 처음 가져갈 상태)
    jobs[job_id] = {
        "status": "pending",
        "message": "서버에서 작업을 준비 중입니다..."
    }

    # 3. 백그라운드 작업 시작 (AI 처리 시뮬레이션)
    thread = threading.Thread(target=simulate_work, args=(job_id,))
    thread.start()

    # 4. 프론트엔드에 작업 ID 즉시 반환
    return jsonify({
        "jobId": job_id,
        "message": "파일 업로드 성공. 처리 작업을 시작합니다."
    }), 202

# -------------------------------------------------------------
# 엔드포인트 2: 작업 상태 폴링 (routes.py의 /api/result/<job_id>)
# -------------------------------------------------------------
@app.route('/api/result/<job_id>', methods=['GET'])
def get_result_route(job_id):
    """
    (frontend/src/services/api.js의 getJobStatus가 1.5초마다 호출) [cite: sunning838/testwork/testwork-f405721f350910119482f7b5a4b2e4d52743c502/frontend/src/components/StatusTracker.jsx]
    현재 작업 상태를 JSON으로 반환합니다.
    """
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "해당 작업 ID를 찾을 수 없습니다."}), 404
    
    # 'jobs' 딕셔너리에 저장된 현재 상태(pending, processing, completed)를 반환
    return jsonify(job)

# -------------------------------------------------------------
# 엔드포인트 3: MIDI 파일 다운로드 (routes.py의 /download/midi/<job_id>)
# -------------------------------------------------------------
@app.route('/download/midi/<job_id>', methods=['GET'])
def download_midi_route(job_id):
    """
    (frontend/src/components/ResultDisplay.jsx에서 다운로드 링크로 사용) [cite: sunning838/testwork/testwork-f405721f350910119482f7b5a4b2e4d52743c502/frontend/src/components/ResultDisplay.jsx]
    가상 MIDI 파일을 생성하여 다운로드시킵니다.
    """
    print(f"[{job_id}] MIDI 다운로드 요청 수신")
    
    # 1. 가상 미디 파일 내용 생성
    modified_file_content = f"// Mock MIDI Score Data for Job {job_id}\n" + "Generated Drum Beats."
    download_data = BytesIO(modified_file_content.encode('utf-8'))
    
    # 2. 파일 다운로드 응답 전송
    return send_file(
        download_data,
        mimetype='audio/midi',
        as_attachment=True, # 다운로드되도록 설정
        download_name=f"{job_id}_score.mid"
    )

# -------------------------------------------------------------
# 엔드포인트 4: PDF 파일 전송 (routes.py의 /download/pdf/<job_id>)
# -------------------------------------------------------------
@app.route('/download/pdf/<job_id>', methods=['GET'])
def download_pdf_route(job_id):
    """
    (frontend/src/components/ResultDisplay.jsx에서 <iframe> 및 다운로드 링크로 사용) [cite: sunning838/testwork/testwork-f405721f350910119482f7b5a4b2e4d52743c502/frontend/src/components/ResultDisplay.jsx]
    로컬의 'test.pdf' 파일을 브라우저에 표시(inline)하거나 다운로드시킵니다.
    """
    print(f"[{job_id}] PDF 요청 수신")
    
    # 1. 'test.pdf' 파일 존재 여부 확인
    if not os.path.exists(PDF_FILE_PATH):
       print(f"[{job_id}] 오류: PDF 파일을 찾을 수 없습니다. 경로: {PDF_FILE_PATH}")
       return jsonify({"error": f"PDF 파일이 서버 경로에 없습니다: {PDF_FILE_PATH}"}), 404

    # 2. PDF 파일 전송
    return send_file(
        PDF_FILE_PATH,
        mimetype='application/pdf',
        as_attachment=False, # <iframe>에 표시될 수 있도록 False로 설정
        download_name=f"{job_id}_score.pdf"
    )

# -------------------------------------------------------------
# 서버 실행
# -------------------------------------------------------------
if __name__ == '__main__':
    print("프론트엔드 연동용 임시 백엔드 서버를 시작합니다...")
    print(f"⚠️ 테스트용 PDF 파일 경로: {PDF_FILE_PATH}")
    if not os.path.exists(PDF_FILE_PATH):
        print(f"⚠️ 경고: 해당 경로에 'test.pdf' 파일이 없습니다. PDF 뷰어가 404 오류를 반환합니다.")
    print("서버가 http://127.0.0.1:5000 에서 실행 중입니다.")
    app.run(debug=True, port=5000)