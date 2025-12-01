# backend/app/tasks.py
import threading
from flask import current_app

jobs = {}


def update_job_status(job_id, status, message, progress=0, results=None):
    """작업의 상태를 업데이트합니다.

    Args:
        job_id: 작업 고유 ID
        status: 작업 상태 ('pending', 'processing', 'completed', 'error')
        message: 상태 메시지
        progress: 진행률 (0~100)
        results: 완료 시 결과 데이터
    """
    if job_id not in jobs:
        jobs[job_id] = {'status': 'pending', 'message': '대기 중', 'progress': 0}

    jobs[job_id]['status'] = status
    jobs[job_id]['message'] = message
    jobs[job_id]['progress'] = progress

    if results:
        jobs[job_id]['results'] = results


def start_background_task(job_id, audio_path, original_filename=None):
    """백그라운드에서 오디오 처리 파이프라인을 실행합니다.

    Args:
        job_id: 작업 고유 ID
        audio_path: 저장된 오디오 파일 경로
        original_filename: 사용자가 업로드한 원본 파일명 (악보 제목에 사용)
    """
    from app.services.audio_processor import run_processing_pipeline

    app = current_app._get_current_object()

    def task_with_context():
        with app.app_context():
            # [수정] 원본 파일명을 파이프라인에 전달
            run_processing_pipeline(job_id, audio_path, original_filename)

    thread = threading.Thread(target=task_with_context)
    thread.start()


def get_job_status(job_id):
    """특정 작업의 상태를 반환합니다."""
    return jobs.get(job_id)