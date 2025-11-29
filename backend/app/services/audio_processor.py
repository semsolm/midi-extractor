# backend/app/services/audio_processor.py

import os
import sys
import subprocess
import traceback
import re
import shutil
from flask import current_app

# [변경] 기존 tensorflow 제거, music21은 PDF용으로 유지
import music21 as m21

# 통합된 MIDI 생성 모듈 임포트
from app.services.MiDi_maker import drum_wav_to_midi, InferenceConfig

# GCS 에서 파일 업로드 / 다운로드 기능 임포트
from app.utils.storage import download_file_from_gcs, upload_file_to_gcs

# --- Demucs 실행 헬퍼 함수 (기존 유지) ---
def run_demucs_separation(input_path, output_dir, job_id):
    from app.tasks import update_job_status

    model_name = "htdemucs"
    demucs_out_dir = os.path.join(output_dir, "separated")

    command = [
        sys.executable, "-m", "demucs.separate", "-n", model_name,
        "--two-stems=drums", "-o", demucs_out_dir, input_path
    ]
    current_app.logger.info(f"[{job_id}] Demucs 명령어 실행: {' '.join(command)}")

    process = subprocess.Popen(
        command, stderr=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, encoding='utf-8', bufsize=1
    )

    last_demucs_message = ""
    try:
        for line in process.stderr:
            line_strip = line.strip()
            # 로그가 너무 많으면 주석 처리 가능
            # current_app.logger.info(f"[Demucs/stderr - {job_id}]: {line_strip}")

            if line_strip.startswith("Separating:"):
                match = re.search(r'(\d+%)', line_strip)
                simple_message = "드럼 분리 중..."
                if match:
                    simple_message = f"드럼 분리 중... {match.group(1)}"
                
                if simple_message != last_demucs_message:
                    last_demucs_message = simple_message
                    update_job_status(job_id, 'processing', message=simple_message)
                
    finally:
        stdout_data, stderr_data = process.communicate()

    if process.returncode != 0:
        current_app.logger.error(f"[{job_id}] Demucs 실행 실패.\nSTDERR: {stderr_data}")
        update_job_status(job_id, 'error', f"Demucs 오류: {stderr_data[:100]}")
        return None

    # 분리된 파일 경로 찾기
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
        update_job_status(job_id, 'error', "Demucs 완료했으나 드럼 파일 없음")
        return None


# --- [신규] PyTorch 기반 MIDI 생성 함수 ---
def generate_midi_with_new_model(drum_audio_path, result_dir, job_id):
    from app.tasks import update_job_status

    model_path = current_app.config['MODEL_PATH']
    
    # 모델 파일 존재 확인
    if not os.path.exists(model_path):
        current_app.logger.error(f"[{job_id}] 모델 파일을 찾을 수 없음: {model_path}")
        update_job_status(job_id, 'error', '서버 설정 오류: 모델 파일 없음')
        return False, 0

    try:
        update_job_status(job_id, 'processing', 'AI 모델 추론 및 MIDI 변환 중...')
        
        # 설정 로드
        config = InferenceConfig()
        
        # [핵심] midi_maker.py의 메인 함수 호출
        # 리턴값: midi_path, bpm, grouped_events
        output_midi_path, detected_bpm, _ = drum_wav_to_midi(
            wav_path=drum_audio_path,
            model_path=model_path,
            output_dir=result_dir,
            config=config,
            bpm_override=None # 자동 감지 사용
        )
        
        current_app.logger.info(f"[{job_id}] MIDI 생성 완료: {output_midi_path}, BPM: {detected_bpm}")
        return True, detected_bpm

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"[{job_id}] MIDI 생성 실패: {e}\n{error_trace}")
        update_job_status(job_id, 'error', f'MIDI 변환 실패: {str(e)}')
        return False, 0


# --- PDF 변환 함수 (기존 유지 + 일부 경로 로직 강화) ---
def generate_pdf_from_midi(midi_path, pdf_output_path, job_id):
    from app.tasks import update_job_status
    
    update_job_status(job_id, 'processing', 'MIDI 파일을 악보(PDF)로 변환 중...')

    # OS별 MuseScore 경로 설정 (.env 파일로 구분함)
    musescore_path = current_app.config['MUSESCORE_PATH']

    if not os.path.exists(musescore_path):
        current_app.logger.error(f"[{job_id}] MuseScore 없음: {musescore_path}")
        # PDF 실패는 치명적이지 않으므로 로그만 남기고 False 리턴 (작업은 완료 처리)
        return False

    xml_temp_path = pdf_output_path.replace(".pdf", ".xml")

    try:
        # 1. Music21: MIDI -> MusicXML
        score = m21.converter.parse(midi_path)
        
        # Percussion Clef 설정
        for part in score.recurse().getElementsByClass(m21.stream.Part):
            part.insert(0, m21.clef.PercussionClef())
        
        # 하이햇(42) 노트 헤드 변경
        for note in score.recurse().getElementsByClass(m21.note.Note):
            if note.pitch.midi == 42:
                note.notehead = 'x'
                
        score.write('musicxml', fp=xml_temp_path)
        
    except Exception as e:
        current_app.logger.error(f"[{job_id}] MusicXML 변환 실패: {e}")
        return False

    try:
        # 2. MuseScore: MusicXML -> PDF
        command = [musescore_path, '-o', pdf_output_path, xml_temp_path]
        # subprocess.run의 무한 대기 루프 방지
        subprocess.run(command, capture_output=True, text=True, timeout=30)
        
        if os.path.exists(pdf_output_path):
            current_app.logger.info(f"[{job_id}] PDF 생성 성공.")
            return True
        else:
            current_app.logger.error(f"[{job_id}] PDF 파일 생성 안됨.")
            return False
    except subprocess.TimeoutExpired:
        current_app.logger.error(f"[{job_id}] PDF 변환 시간 초과 (30초)")
        return False
    except Exception as e:
        current_app.logger.error(f"[{job_id}] MuseScore 실행 실패: {e}")
        return False
    finally:
        # 완료 후 cleanup 실행 (임시 XML 파일 삭제)
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
            "pdfUrl": f"/download/pdf/{job_id}",
            "bpm": detected_bpm
        }
        update_job_status(job_id, 'completed', '변환 완료!', results=results)

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