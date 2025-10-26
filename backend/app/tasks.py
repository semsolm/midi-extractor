# backend/app/tasks.py
import threading
from flask import current_app

# 작업 상태를 저장할 인-메모리 딕셔너리
jobs = {}


def update_job_status(job_id, status, message, results=None):
    """작업의 상태를 업데이트합니다."""
    if job_id not in jobs:
        jobs[job_id] = {}

    jobs[job_id]['status'] = status
    jobs[job_id]['message'] = message
    if results:
        jobs[job_id]['results'] = results


def start_background_task(job_id, audio_path):
    """백그라운드에서 오디오 처리 파이프라인을 실행합니다."""
    from app.services.audio_processor import run_processing_pipeline

    # 현재 실행 중인 Flask app 객체를 안전하게 복사
    app = current_app._get_current_object()

    def task_with_context():
        # 복사된 app 객체의 컨텍스트 안에서 작업을 실행
        with app.app_context():
            run_processing_pipeline(job_id, audio_path)

    thread = threading.Thread(target=task_with_context)
    thread.start()


def get_job_status(job_id):
    """특정 작업의 상태를 반환합니다."""
    return jobs.get(job_id)