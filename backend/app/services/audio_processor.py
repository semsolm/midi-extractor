# backend/app/services/audio_processor.py

import os
import sys
import subprocess
import traceback
import shutil
from flask import current_app
from pathlib import Path

import music21 as m21

from app.services.con_midi_maker import drum_wav_to_midi, InferenceConfig


# --- Demucs 실행 헬퍼 함수 ---
def run_demucs_separation(input_path, output_dir, job_id):
    from app.tasks import update_job_status

    model_name = "htdemucs"
    demucs_out_dir = os.path.join(output_dir, "separated")

    if not os.path.exists(input_path):
        current_app.logger.error(f"[{job_id}] Demucs 입력 파일 없음: {input_path}")
        update_job_status(job_id, 'error', '업로드된 오디오 파일을 찾을 수 없습니다.')
        return None

    command = [
        sys.executable, "-m", "demucs.separate", "-n", model_name,
        "--two-stems=drums", "-o", demucs_out_dir, input_path
    ]
    current_app.logger.info(f"[{job_id}] Demucs 명령어 실행: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        current_app.logger.error(f"[{job_id}] Demucs Popen 단계에서 예외 발생: {e}")
        update_job_status(job_id, 'error', f"Demucs 실행 자체 실패: {str(e)}")
        return None

    current_app.logger.info(f"[{job_id}] Demucs returncode = {result.returncode}")
    current_app.logger.info(f"[{job_id}] Demucs STDOUT:\n{result.stdout}")
    current_app.logger.info(f"[{job_id}] Demucs STDERR:\n{result.stderr}")

    if result.returncode != 0:
        short_msg = (result.stderr or result.stdout or "알 수 없는 오류")[:200]
        current_app.logger.error(
            f"[{job_id}] Demucs 실행 실패.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        update_job_status(job_id, 'error', f"Demucs 오류: {short_msg}")
        return None

    input_filename = os.path.basename(input_path)
    file_stem = os.path.splitext(input_filename)[0]
    separated_drum_file = os.path.join(
        demucs_out_dir, model_name, file_stem, "drums.wav"
    )

    if os.path.exists(separated_drum_file):
        current_app.logger.info(f"[{job_id}] 드럼 분리 성공: {separated_drum_file}")
        return separated_drum_file
    else:
        current_app.logger.error(f"[{job_id}] 오류: Demucs 완료 후 파일을 찾을 수 없음.")
        update_job_status(job_id, 'error', "Demucs 완료했으나 드럼 파일 없음")
        return None


def generate_midi_with_new_model(drum_audio_path, result_dir, job_id):
    """
    BiGRU 모델로 MIDI 생성
    Returns: (success, midi_path, bpm, grouped_events)
    """
    from app.tasks import update_job_status

    model_path = current_app.config['MODEL_PATH']

    if not os.path.exists(model_path):
        current_app.logger.error(f"[{job_id}] 모델 파일을 찾을 수 없음: {model_path}")
        update_job_status(job_id, 'error', '서버 설정 오류: 모델 파일 없음')
        return False, None, 0, None

    try:
        update_job_status(job_id, 'processing', 'AI 채보 및 MIDI 변환 중...')

        config = InferenceConfig()

        # con_midi_maker 호출 - grouped_events도 받아옴!
        midi_path, detected_bpm, grouped_events = drum_wav_to_midi(
            wav_path=drum_audio_path,
            model_path=model_path,
            output_dir=result_dir,
            config=config,
            bpm_override=None
        )

        current_app.logger.info(
            f"[{job_id}] MIDI 생성 완료: {midi_path}, BPM: {detected_bpm}, "
            f"이벤트 수: {len(grouped_events)}"
        )
        return True, midi_path, detected_bpm, grouped_events

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"[{job_id}] MIDI 생성 실패: {e}\n{error_trace}")
        update_job_status(job_id, 'error', f'MIDI 변환 실패: {str(e)}')
        return False, None, 0, None


# =============================================================================
# PDF 생성 - grouped_events를 직접 사용 (MIDI와 동일한 데이터!)
# =============================================================================

# GM 드럼 MIDI 번호 → 드럼 보표 표시 위치 (displayStep, displayOctave)
GM_DRUM_MAP = {
    'kick': ('F', 4),  # 킥 - 첫 번째 줄 아래
    'snare': ('C', 5),  # 스네어 - 세 번째 줄
    'hihat': ('G', 5),  # 하이햇 - 위쪽
}

# 최소 duration (16분음표 = 0.25 quarterLength)
MIN_DURATION = 0.25


def create_drum_note(drum_type, drum_inst):
    """
    드럼 타입에 맞는 Unpitched 노트 생성
    """
    if drum_type in GM_DRUM_MAP:
        display_step, display_octave = GM_DRUM_MAP[drum_type]
    else:
        display_step, display_octave = ('C', 5)

    unpitched = m21.note.Unpitched()
    unpitched.displayStep = display_step
    unpitched.displayOctave = display_octave
    unpitched.duration.quarterLength = MIN_DURATION
    unpitched.storedInstrument = drum_inst

    # 하이햇은 X notehead
    if drum_type == 'hihat':
        unpitched.notehead = 'x'

    return unpitched


def generate_pdf_from_grouped_events(grouped_events, bpm, pdf_output_path, job_id, title="Drum Score"):
    """
    grouped_events를 직접 사용하여 PDF 생성
    - MIDI 파일을 다시 파싱하지 않음
    - con_midi_maker와 정확히 동일한 타이밍 보장
    - 동시 타격을 Chord로 묶어 쉼표 문제 해결
    - 레이아웃 최적화 (한 줄에 마디 4개)

    Args:
        grouped_events: [(time_sec, [drum_types]), ...] 형태
                       예: [(0.0, ['kick', 'hihat']), (0.5, ['snare']), ...]
        bpm: 감지된 BPM
        pdf_output_path: 출력 PDF 경로
        job_id: 작업 ID
        title: 악보 제목
    """
    from app.tasks import update_job_status

    update_job_status(job_id, 'processing', 'MIDI 파일을 악보(PDF)로 변환 중...')
    musescore_path = current_app.config['MUSESCORE_PATH']

    if not os.path.exists(musescore_path):
        current_app.logger.error(f"[{job_id}] MuseScore 없음: {musescore_path}")
        return False

    xml_temp_path = pdf_output_path.replace(".pdf", ".xml")

    try:
        current_app.logger.info(
            f"[{job_id}] grouped_events로 악보 생성 중... "
            f"BPM: {bpm}, 이벤트 수: {len(grouped_events)}, 제목: {title}"
        )

        # 초 → 비트 변환
        seconds_per_beat = 60.0 / bpm

        # ========================================
        # 1. music21 스코어 생성
        # ========================================
        new_score = m21.stream.Score()

        # 메타데이터
        new_score.metadata = m21.metadata.Metadata()
        new_score.metadata.title = title
        new_score.metadata.composer = ""

        # 드럼 파트 생성
        drum_part = m21.stream.Part()
        drum_part.partName = "Drums"

        drum_inst = m21.instrument.UnpitchedPercussion()
        drum_inst.partName = "Drum Set"
        drum_part.insert(0, drum_inst)
        drum_part.insert(0, m21.clef.PercussionClef())

        # 박자표, 템포
        drum_part.insert(0, m21.meter.TimeSignature('4/4'))
        drum_part.insert(0, m21.tempo.MetronomeMark(number=round(bpm)))

        # ========================================
        # 2. grouped_events → music21 노트 변환
        #    동시 타격은 Chord로 묶어서 쉼표 문제 해결
        # ========================================
        note_count = 0

        for time_sec, drum_types in grouped_events:
            # 초 → 비트(quarterLength) 변환
            offset_beats = time_sec / seconds_per_beat

            if len(drum_types) == 1:
                # 단일 노트
                note = create_drum_note(drum_types[0], drum_inst)
                drum_part.insert(offset_beats, note)
                note_count += 1

            else:
                # 동시 타격 → Chord로 묶기 (쉼표 방지)
                notes = []
                for drum_type in drum_types:
                    note = create_drum_note(drum_type, drum_inst)
                    notes.append(note)

                # PercussionChord 생성
                chord = m21.percussion.PercussionChord(notes)
                chord.duration.quarterLength = MIN_DURATION
                drum_part.insert(offset_beats, chord)
                note_count += len(drum_types)

        current_app.logger.info(f"[{job_id}] 총 {note_count}개 노트 생성됨")

        new_score.append(drum_part)

        # ========================================
        # 3. 악보 포맷팅 (쉼표 최소화)
        # ========================================
        current_app.logger.info(f"[{job_id}] 악보 포맷팅 중...")

        # Voice로 묶어서 쉼표 자동 생성 제어
        for measure in drum_part.getElementsByClass('Measure'):
            # 기존 요소들을 Voice 하나로 묶기
            elements = list(measure.notesAndRests)
            if elements:
                voice = m21.stream.Voice()
                for elem in elements:
                    measure.remove(elem)
                    voice.insert(elem.offset, elem)
                measure.insert(0, voice)

        # makeNotation 호출 (마디 구분)
        new_score.makeNotation(inPlace=True)

        # ========================================
        # 4. 레이아웃 설정 (한 줄에 마디 4개)
        # ========================================
        # StaffGroup으로 레이아웃 힌트 추가
        measures = list(drum_part.getElementsByClass('Measure'))
        for i, measure in enumerate(measures):
            if i > 0 and i % 4 == 0:
                # 4마디마다 시스템 브레이크 힌트
                sl = m21.layout.SystemLayout(isNew=True)
                measure.insert(0, sl)

        # MusicXML로 저장
        new_score.write('musicxml', fp=xml_temp_path)
        current_app.logger.info(f"[{job_id}] MusicXML 저장 완료")

    except Exception as e:
        current_app.logger.error(f"[{job_id}] MusicXML 변환 실패: {e}")
        current_app.logger.error(traceback.format_exc())
        return False

    # ========================================
    # 5. MuseScore로 PDF 변환
    # ========================================
    try:
        current_app.logger.info(f"[{job_id}] PDF 생성 중...")
        command = [musescore_path, '-o', pdf_output_path, xml_temp_path]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            current_app.logger.error(f"[{job_id}] MuseScore stderr: {result.stderr}")

        if os.path.exists(pdf_output_path):
            current_app.logger.info(f"[{job_id}] PDF 생성 성공")
            return True
        else:
            current_app.logger.error(f"[{job_id}] PDF 파일 생성 안됨")
            return False

    except subprocess.TimeoutExpired:
        current_app.logger.error(f"[{job_id}] PDF 변환 시간 초과 (60초)")
        return False
    except Exception as e:
        current_app.logger.error(f"[{job_id}] MuseScore 실행 실패: {e}")
        return False
    finally:
        if os.path.exists(xml_temp_path):
            os.remove(xml_temp_path)


def run_processing_pipeline(job_id, audio_path, original_filename=None):
    """
    전체 처리 파이프라인 실행

    Args:
        job_id: 작업 고유 ID
        audio_path: 저장된 오디오 파일 경로
        original_filename: 사용자가 업로드한 원본 파일명 (악보 제목에 사용)
    """
    from app.tasks import update_job_status

    result_dir = os.path.join(current_app.config['RESULT_FOLDER'], job_id)
    os.makedirs(result_dir, exist_ok=True)

    # 1. 드럼 분리 (Demucs)
    update_job_status(job_id, 'processing', '배경음 제거 및 드럼 분리 중...')
    separated_drum_path = run_demucs_separation(audio_path, result_dir, job_id)

    if not separated_drum_path:
        return

    # 2. MIDI 생성 (BiGRU Model + con_midi_maker)
    #    ✅ grouped_events도 함께 받아옴!
    success, generated_midi_path, detected_bpm, grouped_events = generate_midi_with_new_model(
        drum_audio_path=separated_drum_path,
        result_dir=result_dir,
        job_id=job_id
    )

    if not success or not generated_midi_path:
        return

    try:
        # 3. 파일명 변경
        final_midi_path = os.path.join(result_dir, f"{job_id}.mid")

        if os.path.exists(generated_midi_path):
            shutil.move(generated_midi_path, final_midi_path)
        else:
            raise FileNotFoundError(f"MIDI 파일 생성 실패: {generated_midi_path} 없음")

        # 4. PDF 생성 - grouped_events 직접 사용!
        pdf_path = os.path.join(result_dir, f"{job_id}.pdf")

        # [수정] 원본 파일명이 있으면 사용, 없으면 기존 방식 (fallback)
        if original_filename:
            title = os.path.splitext(original_filename)[0]
        else:
            title = os.path.splitext(os.path.basename(audio_path))[0]

        current_app.logger.info(f"[{job_id}] 악보 제목: {title}")

        generate_pdf_from_grouped_events(
            grouped_events=grouped_events,
            bpm=detected_bpm,
            pdf_output_path=pdf_path,
            job_id=job_id,
            title=title
        )

        # 5. 완료
        results = {
            "midiUrl": f"/download/midi/{job_id}",
            "pdfUrl": f"/download/pdf/{job_id}",
            "bpm": detected_bpm
        }
        update_job_status(job_id, 'completed', '변환 완료!', results=results)

    except Exception as e:
        current_app.logger.error(f"[{job_id}] 처리 중 오류: {e}")
        update_job_status(job_id, 'error', f'오류 발생: {str(e)}')