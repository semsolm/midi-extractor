# backend/app/services/audio_processor.py

import os
import sys
import subprocess
import traceback
import shutil
from flask import current_app
from pathlib import Path

import music21 as m21

# 통합된 MIDI 생성 모듈 임포트
from app.services.MiDi_maker import drum_wav_to_midi, InferenceConfig

# GCS 에서 파일 업로드 / 다운로드 기능 임포트
from app.utils.storage import download_file_from_gcs, upload_file_to_gcs

# --- Demucs 실행 헬퍼 함수 ---
def run_demucs_separation(input_path, output_dir, job_id):
    from app.tasks import update_job_status

    model_name = "htdemucs"
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

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        current_app.logger.error(f"[{job_id}] Demucs Popen 단계에서 예외 발생: {e}")
        update_job_status(job_id, 'error', f"Demucs 실행 자체 실패: {str(e)}", progress=0)
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
        update_job_status(job_id, 'error', f"Demucs 오류: {short_msg}", progress=0)
        return None

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


def generate_midi_with_new_model(drum_audio_path, result_dir, job_id):
    """
    BiGRU 모델로 MIDI 생성
    Returns: (success, midi_path, bpm, grouped_events)
    """
    from app.tasks import update_job_status

    model_path = current_app.config['MODEL_PATH']

    if not os.path.exists(model_path):
        current_app.logger.error(f"[{job_id}] 모델 파일을 찾을 수 없음: {model_path}")
        update_job_status(job_id, 'error', '서버 설정 오류: 모델 파일 없음', progress=0)
        return False, None, 0, None

    try:
        update_job_status(job_id, 'processing', 'AI 채보 및 MIDI 변환 중...', progress=40)

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
        update_job_status(job_id, 'error', f'MIDI 변환 실패: {str(e)}', progress=0)
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

    update_job_status(job_id, 'processing', 'MIDI 파일을 악보(PDF)로 변환 중...', progress=70)
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
        update_job_status(job_id, 'processing', '악보 포맷팅 중...', progress=80)

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
        update_job_status(job_id, 'processing', 'PDF 파일 생성 중...', progress=90)

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


# --- [메인] 전체 파이프라인 (GCS 버전) ---
def run_processing_pipeline(job_id, audio_gcs_path):
    from app.tasks import update_job_status

    # 1. 임시 작업 디렉토리 생성 (/tmp/{job_id})
    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)
    
    local_audio_path = os.path.join(work_dir, "input.mp3")

    try:
        # 2. GCS에서 오디오 다운로드
        update_job_status(job_id, 'processing', '클라우드에서 오디오 다운로드 중...')
        download_file_from_gcs(audio_gcs_path, local_audio_path)

        # 3. Demucs (드럼 분리)
        update_job_status(job_id, 'processing', '배경음 제거 및 드럼 분리 중...')
        separated_drum_path = run_demucs_separation(local_audio_path, work_dir, job_id)
        
        if not separated_drum_path:
            return # 실패 시 종료

        # 4. MIDI 생성 (수정된 부분: 새로운 함수 호출!)
        success, detected_bpm = generate_midi_with_new_model(separated_drum_path, work_dir, job_id)
        
        if not success:
            return # 실패 시 종료

        # 5. 파일 정리 및 GCS 업로드 준비
        # work_dir 안에 있는 .mid 파일을 찾아서 {job_id}.mid로 이름 변경
        generated_midis = [f for f in os.listdir(work_dir) if f.endswith('.mid')]
        
        if not generated_midis:
            raise FileNotFoundError("생성된 MIDI 파일을 찾을 수 없습니다.")
            
        real_midi_path = os.path.join(work_dir, generated_midis[0])
        final_midi_path = os.path.join(work_dir, f"{job_id}.mid")
        
        # 파일명 변경
        if real_midi_path != final_midi_path:
            if os.path.exists(final_midi_path):
                os.remove(final_midi_path)
            os.rename(real_midi_path, final_midi_path)

        # 6. PDF 생성
        pdf_path = os.path.join(work_dir, f"{job_id}.pdf")
        generate_pdf_from_midi(final_midi_path, pdf_path, job_id)

        # 7. 결과물 GCS 업로드
        update_job_status(job_id, 'processing', '결과물을 클라우드에 저장 중...')
        
        upload_file_to_gcs(final_midi_path, f"results/{job_id}/{job_id}.mid")
        
        if os.path.exists(pdf_path):
            upload_file_to_gcs(pdf_path, f"results/{job_id}/{job_id}.pdf")

        # 8. 완료 처리
        results = {
            "midiUrl": f"/download/midi/{job_id}",
            "midiUrl": f"/download/midi/{job_id}",
            "pdfUrl": f"/download/pdf/{job_id}",
            "bpm": detected_bpm
        }
        update_job_status(job_id, 'completed', '변환 완료!', progress=100, results=results)

    except Exception as e:
        current_app.logger.error(f"[{job_id}] 파이프라인 처리 중 오류: {e}")
        update_job_status(job_id, 'error', f'오류 발생: {str(e)}')
    
    finally:
        # [중요] 임시 폴더 삭제
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            current_app.logger.info(f"[{job_id}] 임시 폴더 삭제 완료")



# # --- 전체 파이프라인 (로컬 테스트용 메인 진입점) ---
# def run_processing_pipeline(job_id, audio_path):
#     from app.tasks import update_job_status

#     # 결과 저장 폴더: backend/results/<job_id>/
#     result_dir = os.path.join(current_app.config['RESULT_FOLDER'], job_id)
#     os.makedirs(result_dir, exist_ok=True)

#     # 1. 드럼 분리 (Demucs)
#     update_job_status(job_id, 'processing', '배경음 제거 및 드럼 분리 중...')
#     separated_drum_path = run_demucs_separation(audio_path, result_dir, job_id)
    
#     if not separated_drum_path:
#         return # Demucs 실패 시 종료

#     # 2. MIDI 생성 (New BiGRU Model)
#     # BPM 감지도 내부에서 수행됨
#     success, detected_bpm = generate_midi_with_new_model(separated_drum_path, result_dir, job_id)

#     if not success:
#         return # MIDI 실패 시 종료


#     # MIDI 변환 (midi_maker 호출)
#     update_job_status(job_id, 'processing', 'AI 채보 및 MIDI 변환 중...')
    
#     model_path = current_app.config['MODEL_PATH'] # config.py에 정의된 .pt 경로
#     config = InferenceConfig()

#     # 3. PDF 생성
#     midi_file_path = os.path.join(result_dir, f"{job_id}_drums.mid") # midi_maker 저장명 규칙 확인
#     # midi_maker.py는 "{base_name}_drums.mid"로 저장함. 
#     # Demucs 출력 파일명이 'drums.wav'이므로 결과는 'drums_drums.mid'가 될 수도 있고, 
#     # run_demucs_separation의 결과 경로에 따라 달라짐.
#     # 안전하게 midi_maker가 리턴한 경로를 사용하는 것이 좋지만, 
#     # 위 함수(generate_midi_with_new_model)에서는 경로를 리턴받지 않았으므로 확인 필요.
#     # -> midi_maker.py 수정 없이 쓰려면 파일명을 예측하거나 midi_maker가 리턴한 경로를 받아와야 함.
#     # -> 위 generate_midi_with_new_model 코드는 리턴값을 받도록 수정했음 (True, bpm).
#     # -> 하지만 정확한 파일명을 위해 generate_midi_with_new_model가 path도 리턴하게 하면 더 좋음.
#     # -> 현재 midi_maker.py의 drum_wav_to_midi는 (midi_path, bpm, grouped)를 리턴함.
    
#     # [파일명 보정]
#     # run_demucs_separation은 ".../drums.wav"를 리턴함.
#     # midi_maker는 stem 이름을 따서 ".../drums_drums.mid"로 저장할 것임.
#     # 이를 깔끔하게 rename 하거나 그대로 사용. 여기서는 그대로 사용.
    
#     # 실제 저장된 파일명 찾기 (result_dir 내의 .mid 파일)

#     try:
#         # midi_maker 실행 -> 임시 MIDI 파일 생성됨
#         generated_midi_path, detected_bpm, _ = drum_wav_to_midi(
#             wav_path=separated_drum_path,
#             model_path=model_path,
#             output_dir=result_dir,
#             config=config
#         )

#         # 3. 파일명 변경 (프론트엔드 다운로드 규칙 맞추기)
#         # 프론트엔드는 /download/midi/<job_id> 요청 시 "{job_id}.mid"를 찾음
#         final_midi_path = os.path.join(result_dir, f"{job_id}.mid")
        
#         # midi_maker가 만든 파일 이동/이름변경
#         if os.path.exists(generated_midi_path):
#             shutil.move(generated_midi_path, final_midi_path)
#         else:
#             raise FileNotFoundError("MIDI 파일 생성 실패")

#         # 4. PDF 생성
#         pdf_path = os.path.join(result_dir, f"{job_id}.pdf")
#         generate_pdf_from_midi(final_midi_path, pdf_path, job_id)

#         # 5. 완료 상태 업데이트 (프론트엔드로 URL 전달)
#         results = {
#             "midiUrl": f"/download/midi/{job_id}", # routes.py의 라우트와 일치
#             "pdfUrl": f"/download/pdf/{job_id}",
#             "bpm": detected_bpm
#         }
#         update_job_status(job_id, 'completed', '변환 완료!', results=results)

#     except Exception as e:
#         current_app.logger.error(f"[{job_id}] 처리 중 오류: {e}")
#         update_job_status(job_id, 'error', f'오류 발생: {str(e)}')