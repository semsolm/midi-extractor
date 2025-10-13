# app/tasks.py
import threading

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

    # Flask의 애플리케이션 컨텍스트를 스레드 내에서 사용 가능하게 합니다.
    from backend.run import app

    def task_with_context():
        with app.app_context():
            run_processing_pipeline(job_id, audio_path)

    thread = threading.Thread(target=task_with_context)
    thread.start()


def get_job_status(job_id):
    """특정 작업의 상태를 반환합니다."""
    return jobs.get(job_id)