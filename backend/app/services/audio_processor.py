# [최종 수정본] backend/app/services/audio_processor.py

import os
import numpy as np
import librosa
import pretty_midi
import csv
import time
import subprocess
import sys
import re
import io
import traceback
from tqdm import tqdm
from flask import current_app
import tensorflow as tf
import music21 as m21

# --- 상수 정의 ---
SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_SHAPE = (128, 128)

# [수정] B안 (3클래스) 적용
LABELS = ["kick", "snare", "hi-hat"] 
# [수정] B안 (3클래스) 적용 (하이햇을 42번 Cymbals 노트로 매핑)
NOTE_MAP = {"kick": 36, "snare": 38, "hi-hat": 42} 

# --- TQDM 출력을 Job Status의 'message'로 리디렉션 ---
class TqdmToJobUpdater(io.StringIO):
    def __init__(self, job_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job_id = job_id
        self._last_message = "" # 중복 전송을 막기 위한 내부 캐시

    def write(self, s):
        message = s.strip()
        if not message or message == '\r':
            return

        # 정규표현식으로 설명(desc)과 퍼센트(%) 추출
        match = re.search(r'^(.*?:)?\s*(\d+%)', message)
        simple_message = self._last_message

        if match:
            desc = match.group(1) or "처리 중..."
            percent = match.group(2)
            simple_message = f"{desc.strip()} {percent}"
        
        if simple_message != self._last_message:
            self._last_message = simple_message
            from app.tasks import update_job_status
            update_job_status(
                self.job_id,
                'processing',
                simple_message
            )

# --- TFLite 모델 로드 함수 ---
def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        current_app.logger.error(f"치명적 오류: TFLite 모델 파일 '{model_path}'를 찾을 수 없습니다.")
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    current_app.logger.info(f"TFLite 모델 로딩 중: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    current_app.logger.info("TFLite 모델 로딩 및 텐서 할당 완료.")
    return interpreter

# --- 스펙트로그램 변환 함수 ---
def audio_segment_to_melspec(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] < TARGET_SHAPE[1]:
        pad_width = TARGET_SHAPE[1] - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :TARGET_SHAPE[1]]
    return mel_spec_db

# --- Demucs 실행 헬퍼 함수 ---
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
            current_app.logger.info(f"[Demucs/stderr - {job_id}]: {line_strip}")

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
        current_app.logger.error(f"[{job_id}] Demucs 실행 실패.")
        current_app.logger.error(f"[{job_id}] STDERR: {stderr_data}")
        update_job_status(job_id, 'error', f"Demucs 오류: {stderr_data[:100]}")
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
        current_app.logger.error(f"[{job_id}] 오류: Demucs는 성공했으나 'drums.wav' 파일을 찾을 수 없습니다.")
        update_job_status(job_id, 'error', "Demucs 완료했으나 드럼 파일 없음")
        return None

# --- MIDI 생성 메인 함수 ---
def generate_midi_from_audio(audio_path, result_dir, bpm=120):
    from app.tasks import update_job_status
    
    interpreter = load_tflite_model(current_app.config['MODEL_PATH'])
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    job_id = os.path.basename(result_dir)
    midi_out = os.path.join(result_dir, f"{job_id}.mid")
    csv_out = os.path.join(result_dir, f"{job_id}.csv")
    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units='time')

        events = []
        PRE, POST = 0.04, 0.11
        L = int((PRE + POST) * sr)
        
        progress_stream = TqdmToJobUpdater(job_id)

        for t in tqdm(onsets, desc="MIDI 노트 변환 중", file=progress_stream, ncols=80, unit=" 노트"):
            s = max(0, int((t - PRE) * sr));
            e = min(len(y), int((t + POST) * sr))
            seg = y[s:e]
            if len(seg) < L: seg = np.pad(seg, (0, L - len(seg)))

            spec = audio_segment_to_melspec(seg, sr) 
            spec_input = spec[np.newaxis, ..., np.newaxis].astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], spec_input)
            interpreter.invoke()
            proba = interpreter.get_tensor(output_details[0]['index'])[0]

            lab_id = int(proba.argmax())
            lab = LABELS[lab_id]
            events.append((float(t), lab, float(proba[lab_id])))

        with open(csv_out, "w", newline="") as f:
            w = csv.writer(f);
            w.writerow(["time_sec", "label", "prob"])
            for t, lab, p in events: w.writerow([f"{t:.4f}", lab, f"{p:.3f}"])

        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)
        for t, lab, p in events:
            if lab in NOTE_MAP:
                pitch = NOTE_MAP[lab]
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=t, end=t + 0.1)
                drum_instrument.notes.append(note)
        pm.instruments.append(drum_instrument)
        pm.write(midi_out)

        return True
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"MIDI 생성 오류 (job: {job_id}): {e}\n{error_trace}")
        return False

# --- MIDI를 퍼커션 악보 PDF로 변환 함수 ---
# (중복 정의 문제 해결: 파일 내에 이 함수가 *한 번만* 있도록 함)

# [수정] backend/app/services/audio_processor.py

def generate_pdf_from_midi(midi_path, pdf_output_path, job_id):
    """
    music21로 MusicXML을 생성한 뒤, MuseScore를 subprocess로 직접 호출하여
    PDF 드럼 악보를 생성합니다.
    """
    from app.tasks import update_job_status
    update_job_status(job_id, 'processing', 'MIDI 파일을 악보(XML)로 변환 중...')

    # MuseScore 3 경로를 사용합니다.
    musescore_path = r'C:/Program Files/MuseScore 3/bin/MuseScore3.exe' # <-- 본인 경로 확인

    if not os.path.exists(musescore_path):
        current_app.logger.error(f"[{job_id}] MuseScore 경로를 찾을 수 없습니다: {musescore_path}")
        update_job_status(job_id, 'error', 'MuseScore 실행 파일을 찾을 수 없습니다.')
        return False

    xml_temp_path = pdf_output_path.replace(".pdf", ".xml")

    try:
        # 2. MIDI -> MusicXML 변환
        score = m21.converter.parse(midi_path)

        # [수정] 악기 정보를 삭제하지 않고, 기보법(Clef)만 설정합니다.
        for part in score.recurse().getElementsByClass(m21.stream.Part):
            # --- [수정] 이 3줄을 삭제하거나 주석 처리 ---
            # for el in part.getElementsByClass(m21.instrument.Instrument):
            #     part.remove(el)
            # part.insert(0, m21.instrument.Percussion())
            # --- [수정] 여기까지 ---

            # [유지] 퍼커션 기보법(Clef)은 그대로 삽입합니다.
            part.insert(0, m21.clef.PercussionClef())

        # [유지] 노트 헤드 변경 (하이햇 'x')
        for note in score.recurse().getElementsByClass(m21.note.Note):
            if note.pitch.midi == 42: # B안 (하이햇)
                note.notehead = 'x'

        score.write('musicxml', fp=xml_temp_path)
        current_app.logger.info(f"[{job_id}] MusicXML 파일 생성 성공: {xml_temp_path}")

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"[{job_id}] MusicXML 생성 실패: {e}\n{error_trace}")
        update_job_status(job_id, 'error', '악보(XML) 변환 실패')
        return False

    try:
        # 3. MusicXML -> PDF 변환 (Subprocess 호출)
        update_job_status(job_id, 'processing', '악보(PDF) 생성 중...')
        command = [musescore_path, '-o', pdf_output_path, xml_temp_path]

        result = subprocess.run(command, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            current_app.logger.error(f"[{job_id}] MuseScore PDF 변환 실패. Stderr: {result.stderr}")
            update_job_status(job_id, 'error', f'PDF 변환 실패: {result.stderr[:100]}')
            return False
        elif not os.path.exists(pdf_output_path):
            current_app.logger.error(f"[{job_id}] MuseScore는 성공했으나 PDF 파일이 생성되지 않음.")
            update_job_status(job_id, 'error', 'PDF 파일 생성 알 수 없는 오류')
            return False

        current_app.logger.info(f"[{job_id}] PDF 악보 생성 성공: {pdf_output_path}")
        return True

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"[{job_id}] PDF 변환(subprocess) 중 치명적 오류: {e}\n{error_trace}")
        update_job_status(job_id, 'error', 'PDF 변환 중 타임아웃 또는 오류 발생')
        return False

    finally:
        # 4. 임시 XML 파일 삭제
        if os.path.exists(xml_temp_path):
            os.remove(xml_temp_path)
            
# --- 전체 오디오 처리 파이프라인 ---
def run_processing_pipeline(job_id, audio_path):
    from app.tasks import update_job_status

    result_dir = os.path.join(current_app.config['RESULT_FOLDER'], job_id)
    os.makedirs(result_dir, exist_ok=True)

    # --- 1. 드럼 분리 ---
    update_job_status(job_id, 'processing', '드럼 트랙 분리 시작...')
    current_app.logger.info(f"[{job_id}] Demucs 음원 분리 시작...")
    
    separated_drum_path = run_demucs_separation(audio_path, result_dir, job_id)
    
    if not separated_drum_path:
        current_app.logger.error(f"[{job_id}] 작업 실패: Demucs 실행 오류.")
        return # [중요] 여기서 함수 종료

    # --- 2. BPM 분석 ---
    update_job_status(job_id, 'processing', '음원 템포(BPM) 분석 중...')
    current_app.logger.info(f"[{job_id}] BPM 분석 시작...")
    try:
        y, sr = librosa.load(audio_path, sr=SR)
        # [수정] librosa.beat.beat_track 사용 
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(tempo)
        current_app.logger.info(f"[{job_id}] 분석된 BPM: {bpm}")
    except Exception as e:
        bpm = 120
        current_app.logger.warning(f"[{job_id}] BPM 분석 실패: {e}. 기본값 {bpm}으로 설정.")
    
    update_job_status(job_id, 'processing', 'BPM 분석 완료. MIDI 변환 시작...')

    # --- 3. MIDI 생성 ---
    current_app.logger.info(f"[{job_id}] MIDI 생성 시작...")
    
    midi_success = generate_midi_from_audio(separated_drum_path, result_dir, bpm)

    # [수정] MIDI 생성 실패 시, 여기서 즉시 'error'로 상태 변경하고 종료
    if not midi_success:
        update_job_status(job_id, 'error', 'MIDI 생성 중 오류가 발생했습니다.')
        current_app.logger.error(f"[{job_id}] 작업 실패: MIDI 생성 실패.")
        return # [중요] 여기서 함수 종료

    # --- 4. PDF 생성 (MIDI가 성공했을 때만 실행) ---
    midi_file_path = os.path.join(result_dir, f"{job_id}.mid")
    pdf_file_path = os.path.join(result_dir, f"{job_id}.pdf")
    
    pdf_success = generate_pdf_from_midi(midi_file_path, pdf_file_path, job_id)

    if not pdf_success:
        current_app.logger.error(f"[{job_id}] PDF 변환 실패. MIDI만 제공됩니다.")
        # PDF 실패는 전체 실패가 아님, MIDI는 성공했으므로 계속 진행

    # --- 5. 최종 결과 업데이트 (MIDI 성공 시 무조건 'completed'
    results = {
        "midiUrl": f"/download/midi/{job_id}",
        # [수정] PDF가 실패했어도 URL은 보냄 (프론트/백엔드에서 404 처리)
        "pdfUrl": f"/download/pdf/{job_id}", 
    }
    update_job_status(job_id, 'completed', '작업이 완료되었습니다.', results=results)
    current_app.logger.info(f"[{job_id}] 모든 작업 완료.")