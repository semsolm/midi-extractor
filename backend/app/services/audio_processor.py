# backend/app/services/audio_processor.py

import os
import sys
import subprocess
import traceback
import shutil
import re
from flask import current_app
from pathlib import Path

import music21 as m21

# 통합된 MIDI 생성 모듈 임포트
from app.services.con_midi_maker import drum_wav_to_midi, InferenceConfig

# GCS 유틸리티 임포트
from app.utils.storage import download_file_from_gcs, upload_file_to_gcs

# =============================================================================
# 1. Demucs 실행 (음원 분리)
# =============================================================================
def run_demucs_separation(input_path, output_dir, job_id):
    from app.tasks import update_job_status

    model_name = "htdemucs"
    # Demucs 출력 경로: output_dir/separated/htdemucs/{filename}/drums.wav
    demucs_out_dir = os.path.join(output_dir, "separated")

    if not os.path.exists(input_path):
        current_app.logger.error(f"[{job_id}] Demucs 입력 파일 없음: {input_path}")
        update_job_status(job_id, 'error', '업로드된 오디오 파일을 찾을 수 없습니다.', progress=0)
        return None

    command = [
        sys.executable, "-m", "demucs.separate", "-n", model_name,
        "--two-stems=drums", "-o", demucs_out_dir, input_path
    ]
    current_app.logger.info(f"[{job_id}] Demucs 명령어 실행: {' '.join(command)}")

    # 프로세스 실행
    process = subprocess.Popen(
        command, stderr=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, encoding='utf-8', bufsize=1
    )

    # 실시간 로그 모니터링
    try:
        for line in process.stderr:
            if line.strip().startswith("Separating:"):
                update_job_status(job_id, 'processing', message="배경음 제거 및 드럼 분리 중...")
    finally:
        stdout_data, stderr_data = process.communicate()

    if process.returncode != 0:
        current_app.logger.error(f"[{job_id}] Demucs 실패: {stderr_data}")
        update_job_status(job_id, 'error', f"Demucs 오류: {stderr_data[:100]}")
        return None

    # 결과 파일 경로 찾기
    input_filename = os.path.basename(input_path)
    file_stem = os.path.splitext(input_filename)[0]

    # Demucs 출력 구조: {output_dir}/htdemucs/{file_stem}/drums.wav
    separated_drum_file = os.path.join(
        demucs_out_dir, model_name, file_stem, "drums.wav"
    )

    if os.path.exists(separated_drum_file):
        current_app.logger.info(f"[{job_id}] 드럼 분리 성공: {separated_drum_file}")
        return separated_drum_file
    else:
        current_app.logger.error(f"[{job_id}] 오류: Demucs 완료 후 파일을 찾을 수 없음.")
        update_job_status(job_id, 'error', "Demucs 완료했으나 드럼 파일 없음", progress=0)
        return None


# =============================================================================
# 2. MIDI 생성 (BiGRU 모델 연결)
# =============================================================================
def generate_midi_with_new_model(drum_audio_path, result_dir, job_id):
    """
    MiDi_maker.py를 호출하여 MIDI 생성 및 이벤트 데이터 반환
    """
    from app.tasks import update_job_status

    model_path = current_app.config['MODEL_PATH']

    if not os.path.exists(model_path):
        current_app.logger.error(f"[{job_id}] 모델 파일 없음: {model_path}")
        update_job_status(job_id, 'error', '서버 설정 오류: 모델 파일 없음')
        return False, None, 0, None

    try:
        update_job_status(job_id, 'processing', 'AI 모델 추론 및 MIDI 변환 중...')

        config = InferenceConfig()

        # [연결 2] MiDi_maker.drum_wav_to_midi 호출
        # 반환값: (MIDI파일경로, BPM, 그룹화된_이벤트_리스트)
        midi_path, detected_bpm, grouped_events = drum_wav_to_midi(
            wav_path=drum_audio_path,
            model_path=model_path,
            output_dir=result_dir,
            config=config,
            bpm_override=None
        )

        current_app.logger.info(f"[{job_id}] MIDI 생성 완료. BPM: {detected_bpm}, Events: {len(grouped_events)}")
        return True, midi_path, detected_bpm, grouped_events

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"[{job_id}] MIDI 생성 실패: {e}\n{error_trace}")
        update_job_status(job_id, 'error', f'MIDI 변환 실패: {str(e)}')
        return False, None, 0, None


# =============================================================================
# 3. PDF 생성 (grouped_events 기반)
# =============================================================================

# GM 드럼 매핑 (표시 위치)
GM_DRUM_MAP = {
    'kick': ('F', 4),   # 낮은 파
    'snare': ('C', 5),  # 높은 도
    'hihat': ('G', 5),  # 높은 솔 (X 헤드)
}
MIN_DURATION = 0.25 # 16분음표

def create_drum_note(drum_type, drum_inst):
    """Music21 노트 생성 헬퍼"""
    step, octave = GM_DRUM_MAP.get(drum_type, ('C', 5))
    
    note = m21.note.Unpitched()
    note.displayStep = step
    note.displayOctave = octave
    note.duration.quarterLength = MIN_DURATION
    note.storedInstrument = drum_inst
    
    if drum_type == 'hihat':
        note.notehead = 'x'
    
    return note

def generate_pdf_from_grouped_events(grouped_events, bpm, pdf_output_path, job_id):
    """
    AI가 검출한 이벤트 리스트(grouped_events)를 기반으로 악보 생성
    """
    from app.tasks import update_job_status
    
    update_job_status(job_id, 'processing', 'MIDI 파일을 악보(PDF)로 변환 중...')
    musescore_path = current_app.config['MUSESCORE_PATH']

    if not os.path.exists(musescore_path):
        current_app.logger.error(f"[{job_id}] MuseScore 없음: {musescore_path}")
        return False

    xml_temp_path = pdf_output_path.replace(".pdf", ".xml")

    try:
        # A. Score 구조 생성
        score = m21.stream.Score()
        part = m21.stream.Part()
        inst = m21.instrument.UnpitchedPercussion()
        part.insert(0, inst)
        part.insert(0, m21.clef.PercussionClef())
        part.insert(0, m21.meter.TimeSignature('4/4'))
        part.insert(0, m21.tempo.MetronomeMark(number=round(bpm)))

        seconds_per_beat = 60.0 / bpm

        # B. 노트 배치
        for time_sec, drum_types in grouped_events:
            # 초 → 비트(quarterLength) 변환
            offset_beats = time_sec / seconds_per_beat
            
            if len(drum_types) == 1:
                # 단일 타격
                n = create_drum_note(drum_types[0], inst)
                part.insert(offset_beats, n)
            else:
                # 동시 타격 (Chord)
                notes = [create_drum_note(dt, inst) for dt in drum_types]
                chord = m21.percussion.PercussionChord(notes)
                chord.duration.quarterLength = MIN_DURATION
                part.insert(offset_beats, chord)

        score.append(part)

        # C. 포맷팅 (Voice 처리로 쉼표 자동 정리)
        for measure in part.getElementsByClass('Measure'):
            elements = list(measure.notesAndRests)
            if elements:
                voice = m21.stream.Voice()
                for elem in elements:
                    measure.remove(elem)
                    voice.insert(elem.offset, elem)
                measure.insert(0, voice)
        
        score.makeNotation(inPlace=True)
        score.write('musicxml', fp=xml_temp_path)

        # D. PDF 변환 (MuseScore)
        command = [musescore_path, '-o', pdf_output_path, xml_temp_path]
        subprocess.run(command, capture_output=True, text=True, timeout=60)

        if os.path.exists(pdf_output_path):
            current_app.logger.info(f"[{job_id}] PDF 생성 성공")
            return True
        return False

    except Exception as e:
        current_app.logger.error(f"[{job_id}] PDF 생성 실패: {e}")
        return False
    finally:
        if os.path.exists(xml_temp_path):
            os.remove(xml_temp_path)


# =============================================================================
# 4. 메인 파이프라인 (orchestrator)
# =============================================================================
def run_processing_pipeline(job_id, audio_gcs_path):
    from app.tasks import update_job_status

    # 임시 작업 디렉토리
    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)
    local_audio_path = os.path.join(work_dir, "input.mp3")

    try:
        # 2. GCS에서 오디오 다운로드
        update_job_status(job_id, 'processing', '클라우드에서 오디오 다운로드 중...')
        download_file_from_gcs(audio_gcs_path, local_audio_path)

        # 2. Demucs 실행
        separated_drum_path = run_demucs_separation(local_audio_path, work_dir, job_id)
        if not separated_drum_path: return

        # 3. MIDI 생성 (grouped_events 받아오기)
        success, generated_midi_path, detected_bpm, grouped_events = generate_midi_with_new_model(
            separated_drum_path, work_dir, job_id
        )
        if not success: return

        # 4. 파일 정리 (job_id.mid 로 이름 변경)
        final_midi_path = os.path.join(work_dir, f"{job_id}.mid")
        if generated_midi_path != final_midi_path:
            if os.path.exists(final_midi_path): os.remove(final_midi_path)
            os.rename(generated_midi_path, final_midi_path)

        # 5. PDF 생성 (grouped_events 사용)
        final_pdf_path = os.path.join(work_dir, f"{job_id}.pdf")
        generate_pdf_from_grouped_events(grouped_events, detected_bpm, final_pdf_path, job_id)

        # 6. GCS 업로드
        update_job_status(job_id, 'processing', '결과물 업로드 중...')
        upload_file_to_gcs(final_midi_path, f"results/{job_id}/{job_id}.mid")
        
        if os.path.exists(final_pdf_path):
            upload_file_to_gcs(final_pdf_path, f"results/{job_id}/{job_id}.pdf")

        # 7. 완료
        results = {
            "midiUrl": f"/download/midi/{job_id}",
            "pdfUrl": f"/download/pdf/{job_id}",
            "bpm": detected_bpm
        }
        update_job_status(job_id, 'completed', '변환 완료!', results=results)

    except Exception as e:
        current_app.logger.error(f"[{job_id}] 파이프라인 오류: {e}")
        update_job_status(job_id, 'error', f'오류 발생: {str(e)}')
    
    finally:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            current_app.logger.info(f"[{job_id}] 임시 폴더 정리 완료")