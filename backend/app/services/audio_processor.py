# app/services/audio_processor.py
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
import traceback

# --- 상수 정의 (기존과 동일) ---
SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_SHAPE = (128, 128)
LABELS = ["kick", "snare", "toms", "overheads"]
NOTE_MAP = {"kick": 36, "snare": 38, "toms": 45, "overheads": 42}

# --- [수정] TQDM 출력을 Job Status의 'message'로 리디렉션 ---
class TqdmToJobUpdater(io.StringIO):
    """
    tqdm의 진행도 바(bar) 문자열을 가로채서
    update_job_status()의 'message' 필드에 통째로 씁니다.
    """
    def __init__(self, job_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job_id = job_id

    def write(self, s):
        # tqdm이 출력하는 문자열 (e.g., "MIDI 변환: 50%|███...| 50/100 ...")
        message = s.strip()
        
        # 빈 줄이 아니거나 \r (캐리지 리턴)만 있지 않은 경우에만 업데이트
        if message and message != '\r':
            from app.tasks import update_job_status
            # [수정] 'message' 필드에 tqdm 출력 문자열 원본을 전달
            update_job_status(
                self.job_id,
                'processing',
                message
            )

# --- TFLite 모델 로드 함수 (기존과 동일, logger 사용) ---
def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        current_app.logger.error(f"치명적 오류: TFLite 모델 파일 '{model_path}'를 찾을 수 없습니다.")
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    current_app.logger.info(f"TFLite 모델 로딩 중: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    current_app.logger.info("TFLite 모델 로딩 및 텐서 할당 완료.")
    return interpreter

# --- 스펙트로그램 변환 함수 (기존과 동일) ---
def audio_segment_to_melspec(y, sr):
    # ... (내용 동일) ...
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] < TARGET_SHAPE[1]:
        pad_width = TARGET_SHAPE[1] - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :TARGET_SHAPE[1]]
    return mel_spec_db

# --- [수정] Demucs 실행 헬퍼 함수 (진행도 문자열을 'message'로) ---
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

    # [수정] Demucs의 "Separating: ..." 문자열만 찾기
    progress_re = re.compile(r"Separating:.*")

    try:
        for line in iter(process.stderr.readline, ''):
            if not line and process.poll() is not None:
                break
            
            line_strip = line.strip()
            current_app.logger.info(f"[Demucs/stderr - {job_id}]: {line_strip}")

            # [수정] "Separating: ..."으로 시작하는 라인만 'message'로 업데이트
            if line_strip.startswith("Separating:"):
                update_job_status(job_id, 'processing', message=line_strip)
                
    finally:
        stdout_data, stderr_data = process.communicate()

    if process.returncode != 0:
        current_app.logger.error(f"[{job_id}] Demucs 실행 실패.")
        current_app.logger.error(f"[{job_id}] STDERR: {stderr_data}")
        update_job_status(job_id, 'error', f"Demucs 오류: {stderr_data[:100]}")
        return None

    # --- (파일 찾기 로직) ---
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

# --- [수정] MIDI 생성 메인 함수 (tqdm 수정) ---
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
        
        # [수정] TqdmToJobUpdater 인스턴스 (task_name 제거)
        progress_stream = TqdmToJobUpdater(job_id)

        # [수정] tqdm에 'desc' (설명) 추가 -> 'message'에 찍힘
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

        # (CSV 및 MIDI 생성 로직은 동일)
        with open(csv_out, "w", newline="") as f:
            w = csv.writer(f);
            w.writerow(["time_sec", "label", "prob"])
            for t, lab, p in events: w.writerow([f"{t:.4f}", lab, f"{p:.3f}"])
        # ... pretty_midi 로직 ...
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

# --- 전체 오디오 처리 파이프라인 (progress 제거) ---
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
        return 

    current_app.logger.info(f"[{job_id}] 드럼 분리 완료.")
    
    # --- 2. BPM 분석 ---
    update_job_status(job_id, 'processing', '음원 템포(BPM) 분석 중...')
    current_app.logger.info(f"[{job_id}] BPM 분석 시작...")
    try:
        y, sr = librosa.load(audio_path, sr=SR)
        tempo, _ = librosa.beat.track(y=y, sr=sr)
        bpm = int(tempo)
        current_app.logger.info(f"[{job_id}] 분석된 BPM: {bpm}")
    except Exception as e:
        bpm = 120
        current_app.logger.warning(f"[{job_id}] BPM 분석 실패: {e}. 기본값 {bpm}으로 설정.")
    
    update_job_status(job_id, 'processing', 'BPM 분석 완료. MIDI 변환 시작...')

    # --- 3. MIDI 생성 ---
    current_app.logger.info(f"[{job_id}] MIDI 생성 시작...")
    
    # [수정] midi_success 변수에 성공 여부 저장
    midi_success = generate_midi_from_audio(separated_drum_path, result_dir, bpm)

    if not midi_success:
        update_job_status(job_id, 'error', 'MIDI 생성 중 오류가 발생했습니다.')
        current_app.logger.error(f"[{job_id}] 작업 실패.")
        return

    # --- 4. (신규) PDF 생성 ---
    if midi_success:
        midi_file_path = os.path.join(result_dir, f"{job_id}.mid")
        pdf_file_path = os.path.join(result_dir, f"{job_id}.pdf")
        
        pdf_success = generate_pdf_from_midi(midi_file_path, pdf_file_path, job_id)

        if not pdf_success:
            current_app.logger.error(f"[{job_id}] PDF 변환 실패. MIDI만 제공됩니다.")
            # PDF가 실패해도 MIDI는 성공했으므로 계속 진행합니다.

    # --- 5. [수정] 최종 결과 업데이트 ---
    if midi_success:
        results = {
            "midiUrl": f"/download/midi/{job_id}",
            "pdfUrl": f"/download/pdf/{job_id}",
        }
        update_job_status(job_id, 'completed', '작업이 완료되었습니다.', results=results)
        current_app.logger.info(f"[{job_id}] 모든 작업 완료.")
    
    else: 
        # midi_success가 False인 경우 (generate_midi_from_audio 실패)
        update_job_status(job_id, 'error', 'MIDI 생성 중 오류가 발생했습니다.')
        current_app.logger.error(f"[{job_id}] 작업 실패.")


# --- MuseScore 경로 설정 ---
# 1. PC에 설치된 MuseScore 경로를 여기에 붙여넣으세요.
MUSESCORE_PATH = r'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'

try:
    us = m21.environment.UserSettings()
    # 'musicxmlPath' (MuseScore) 또는 'lilypondPath' 경로를 직접 지정
    us['musicxmlPath'] = MUSESCORE_PATH
    us['musescoreDirectPNGPath'] = MUSESCORE_PATH # PDF 변환에도 이 경로가 사용됨
    current_app.logger.info(f"music21: MuseScore 경로가 {MUSESCORE_PATH}로 설정되었습니다.")
except Exception as e:
    current_app.logger.warning(f"music21: 환경 설정 중 경고 발생: {e}")

# --- MIDI를 퍼커션 악보 PDF로 변환 함수 ---
def generate_pdf_from_midi(midi_path, pdf_output_path, job_id):
    """
    music21을 사용해 MIDI 파일을 드럼 악보(PDF)로 변환합니다.
    MuseScore가 로컬에 설치되어 있어야 합니다.
    """
    from app.tasks import update_job_status
    update_job_status(job_id, 'processing', 'MIDI 파일을 악보로 변환 중...')

    musescore_path = r'C:/Program Files/MuseScore 4/bin/MuseScore4.exe' # <-- 본인 경로로 수정!

    try:
        # 2. music21 환경 설정 객체를 가져옵니다.
        us = m21.environment.UserSettings()
        
        # 3. PDF 변환에 사용될 MuseScore 실행 파일 경로를 명시적으로 지정합니다.
        us['musicxmlPath'] = musescore_path
        us['musescoreDirectPNGPath'] = musescore_path 
        
        current_app.logger.info(f"[{job_id}] music21: MuseScore 경로가 {musescore_path}로 설정되었습니다.")
    
    except Exception as e:
        current_app.logger.error(f"[{job_id}] music21: MuseScore 경로 설정 중 심각한 오류 발생: {e}")
        update_job_status(job_id, 'error', f'MuseScore 경로 설정 오류: {e}')
        return False
    
    try:
        # 1. MIDI 파일 로드
        score = m21.converter.parse(midi_path)

        # 2. (중요) 모든 파트를 '퍼커션 악보'로 강제 변환
        for part in score.recurse().getElementsByClass(m21.stream.Part):
            # 기존 악기 정보 삭제
            for el in part.getElementsByClass(m21.instrument.Instrument):
                part.remove(el)

            # 퍼커션 기호(Clef) 및 악기 삽입
            part.insert(0, m21.clef.PercussionClef())
            part.insert(0, m21.instrument.Percussion()) # 드럼 악보임을 명시

        # 3. (선택) 노트 헤드 변경 (예: 하이햇/오버헤드를 'x'로)
        for note in score.recurse().getElementsByClass(m21.note.Note):
            # 42번(Overheads) 또는 46번(Hi-hat) 등...
            if note.pitch.midi in [42, 46, 49, 51, 57]: 
                note.notehead = 'x'

        # 4. PDF로 변환 (로컬에 설치된 MuseScore를 호출)
        # 'fp' (filepath) 인자를 사용해야 합니다.
        score.write('pdf', fp=pdf_output_path)
        current_app.logger.info(f"[{job_id}] PDF 악보 생성 성공: {pdf_output_path}")
        return True

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"[{job_id}] PDF 생성 실패 (MuseScore 설치 확인): {e}\n{error_trace}")
        update_job_status(job_id, 'error', 'PDF 악보 변환 실패 (MuseScore 설치 확인)')
        return False