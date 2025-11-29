# backend/app/routes.py
import os
import uuid
import io
from flask import request, jsonify, current_app, send_file, Blueprint
from . import tasks
# [추가] storage 모듈 임포트
from app.utils.storage import upload_stream_to_gcs, download_blob_to_memory

bp = Blueprint('api', __name__)



@bp.route('/', methods=['GET'])
def health_check():
    """로드밸런서 헬스 체크용"""
    return "OK", 200


@bp.route('/api/process', methods=['POST'])
def process_audio_route():
    if 'audio_file' not in request.files:
        return jsonify({"error": "오디오 파일이 없습니다."}), 400
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    if file:
        job_id = str(uuid.uuid4())
        
        # [변경] 로컬 저장 대신 GCS로 스트림 업로드
        gcs_filename = f"uploads/{job_id}.mp3"
        try:
            upload_stream_to_gcs(file, gcs_filename, file.content_type)
        except Exception as e:
            return jsonify({"error": f"GCS 업로드 실패: {str(e)}"}), 500

        tasks.update_job_status(job_id, 'pending', '작업을 대기 중입니다.')
        
        # [변경] 로컬 경로 대신 GCS 경로(키)를 전달
        tasks.start_background_task(job_id, gcs_filename)

        return jsonify({
            "jobId": job_id,
            "message": "파일 업로드 성공. 처리 작업을 시작합니다."
        }), 202

@bp.route('/api/result/<job_id>', methods=['GET'])
def get_result_route(job_id):
    job = tasks.get_job_status(job_id)
    if not job:
        return jsonify({"error": "해당 작업 ID를 찾을 수 없습니다."}), 404
    return jsonify(job)

@bp.route('/download/midi/<job_id>', methods=['GET'])
def download_midi_route(job_id):
    try:
        # [변경] GCS에서 다운로드하여 바로 전송
        gcs_path = f"results/{job_id}/{job_id}.mid"
        file_data = download_blob_to_memory(gcs_path)
        return send_file(
            io.BytesIO(file_data),
            as_attachment=True,
            download_name=f"{job_id}.mid",
            mimetype='audio/midi'
        )
    except Exception:
        return jsonify({"error": "MIDI 파일을 찾을 수 없습니다."}), 404

@bp.route('/download/pdf/<job_id>', methods=['GET'])
def download_pdf_route(job_id):
    try:
        # [변경] GCS에서 다운로드하여 바로 전송
        gcs_path = f"results/{job_id}/{job_id}.pdf"
        file_data = download_blob_to_memory(gcs_path)
        return send_file(
            io.BytesIO(file_data),
            as_attachment=False, # 브라우저에서 바로 보기
            download_name=f"{job_id}.pdf",
            mimetype='application/pdf'
        )
    except Exception:
        return jsonify({"error": "PDF 악보 파일을 찾을 수 없습니다."}), 404