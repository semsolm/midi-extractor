# app/routes.py
import os
import uuid
from flask import request, jsonify, current_app, send_from_directory, Blueprint
from . import tasks

bp = Blueprint('api', __name__)


@bp.route('/api/process', methods=['POST'])
def process_audio_route():
    """오디오 파일을 업로드하고 처리 작업을 시작합니다."""
    if 'audio_file' not in request.files:
        return jsonify({"error": "오디오 파일이 없습니다."}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    if file:
        job_id = str(uuid.uuid4())
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{job_id}.mp3")
        file.save(filepath)

        # [수정] progress=0 제거
        tasks.update_job_status(job_id, 'pending', '작업을 대기 중입니다.')
        tasks.start_background_task(job_id, filepath)

        return jsonify({
            "jobId": job_id,
            "message": "파일 업로드 성공. 처리 작업을 시작합니다."
        }), 202

# --- (이하 /api/result/, /download/ 등은 기존과 동일) ---
@bp.route('/api/result/<job_id>', methods=['GET'])
def get_result_route(job_id):
    job = tasks.get_job_status(job_id)
    if not job:
        return jsonify({"error": "해당 작업 ID를 찾을 수 없습니다."}), 404
    return jsonify(job)


@bp.route('/download/midi/<job_id>', methods=['GET'])
def download_midi_route(job_id):
    result_dir = os.path.join(current_app.config['RESULT_FOLDER'], job_id)
    filename = f"{job_id}.mid"
    if os.path.exists(os.path.join(result_dir, filename)):
        return send_from_directory(result_dir, filename, as_attachment=True)
    return jsonify({"error": "MIDI 파일을 찾을 수 없습니다."}), 404


@bp.route('/download/pdf/<job_id>', methods=['GET'])
def download_pdf_route(job_id):
    result_dir = os.path.join(current_app.config['RESULT_FOLDER'], job_id)
    filename = f"{job_id}.pdf"
    if os.path.exists(os.path.join(result_dir, filename)):
        return send_from_directory(result_dir, filename, as_attachment=False)
    
    # [수정] 오류 메시지를 "MIDI"에서 "PDF"로 변경
    return jsonify({"error": "PDF 악보 파일을 찾을 수 없습니다."}), 404