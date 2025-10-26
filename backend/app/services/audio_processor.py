# app/services/audio_processor.py
import os
import numpy as np
import librosa
import pretty_midi
import csv
import time
import subprocess  # 추가: Demucs를 실행하기 위한 모듈
import sys  # 추가: 현재 파이썬 실행 파일 경로를 찾기 위한 모듈
from flask import current_app
from packaging.tags import interpreter_name
from tensorflow.keras.models import load_model

# --- AI 모델 로드 및 상수 정의 ---
SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_SHAPE = (128, 128)
LABELS = ["kick", "snare", "toms", "overheads"]
NOTE_MAP = {"kick": 36, "snare": 38, "toms": 45, "overheads": 42}

# --- TFLite 모델 로드 함수 ---
def load_tflite_model(model_path):
    """TFLite 모델 인터프리터를 로드하고 텐서를 할당합니다."""
    if not os.path.exists(model_path):
        print(f"치명적 오류: TFLite 모델 파일 '{model_path}'를 찾을 수 없습니다.")
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    print(f"TFLite 모델 로딩 중: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("TFLite 모델 로딩 및 텐서 할당 완료.")
    return interpreter

# # --- Keras 모델 로드 함수 ---
# def load_keras_model(model_path):
#     """Keras 모델 파일을 로드합니다."""
#     if not os.path.exists(model_path):
#         print(f"치명적 오류: 모델 파일 '{model_path}'를 찾을 수 없습니다.")
#         raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
#
#     print(f"Keras 모델 로딩 중: {model_path}")
#     model = load_model(model_path)
#     print("모델 로딩 완료.")
#     return model


# --- CNN 입력용 스펙트로그램 변환 함수 ---
def audio_segment_to_melspec(y, sr):
    """오디오 세그먼트를 CNN 입력용 멜 스펙트로그램으로 변환합니다."""
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec_db.shape[1] < TARGET_SHAPE[1]:
        pad_width = TARGET_SHAPE[1] - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :TARGET_SHAPE[1]]

    return mel_spec_db


# --- Demucs 실행 헬퍼 함수 ---
def run_demucs_separation(input_path, output_dir, job_id):
    """
    Demucs를 subprocess로 실행하여 드럼 트랙을 분리합니다.
    성공 시 드럼 파일 경로를 반환하고, 실패 시 None을 반환합니다.
    """
    model_name = "htdemucs"  # Demucs 4의 빠른 모델 (기본 X)

    # Demucs가 생성할 최종 출력 폴더
    # 예: .../results/{job_id}/separated/
    demucs_out_dir = os.path.join(output_dir, "separated")

    # --- Demucs 실행 명령어 생성 ---
    # 예: python -m demucs.separate -n htdemucs_ft --two-stems=drums -o "results/job123/separated" "uploads/job123.mp3"
    command = [
        sys.executable,  # 현재 가상환경의 파이썬 실행파일
        "-m", "demucs.separate",
        "-n", model_name,
        "--two-stems=drums",  # 'drums'와 'no_drums' 두 파일만 생성
        "-o", demucs_out_dir,  # 출력 폴더 지정
        input_path  # 입력 파일 지정
    ]

    print(f"[{job_id}] Demucs 명령어 실행: {' '.join(command)}")
    try:
        # Demucs 실행
        subprocess.run(command, check=True, capture_output=True, text=True)

        # --- 성공한 파일 경로 찾기 ---
        # Demucs는 'demucs_out_dir / model_name / input_filename_without_ext / drums.wav'
        # 경로에 분리된 파일을 저장합니다.
        input_filename = os.path.basename(input_path)
        file_stem = os.path.splitext(input_filename)[0]  # 'job_id.mp3' -> 'job_id'

        separated_drum_file = os.path.join(
            demucs_out_dir, model_name, file_stem, "drums.wav"
        )

        if os.path.exists(separated_drum_file):
            print(f"[{job_id}] 드럼 분리 성공: {separated_drum_file}")
            return separated_drum_file
        else:
            print(f"[{job_id}] 오류: Demucs는 성공했으나 'drums.wav' 파일을 찾을 수 없습니다.")
            print(f"[{job_id}] 예상 경로: {separated_drum_file}")
            return None

    except subprocess.CalledProcessError as e:
        # Demucs 실행이 실패했을 때
        print(f"[{job_id}] Demucs 실행 실패.")
        print(f"[{job_id}] STDERR: {e.stderr}")
        return None
    except Exception as e:
        print(f"[{job_id}] Demucs 실행 중 알 수 없는 오류: {e}")
        return None


# --- MIDI 생성 메인 함수 (CNN 모델 사용) ---
def generate_midi_from_audio(audio_path, result_dir, bpm=120):
    """오디오 파일로부터 MIDI와 CSV를 생성하는 함수."""

    # Keras 모델
    # model = load_keras_model(current_app.config['MODEL_PATH'])
    # job_id = os.path.basename(result_dir)
    # midi_out = os.path.join(result_dir, f"{job_id}.mid")
    # csv_out = os.path.join(result_dir, f"{job_id}.csv")

    # TFLite 인터프리터 로드
    interpreter = load_tflite_model(current_app.config['MODEL_PATH'])

    # TFLite 모델의 입력 및 출력 세부 정보 가져오기
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

        for t in onsets:
            s = max(0, int((t - PRE) * sr));
            e = min(len(y), int((t + POST) * sr))
            seg = y[s:e]
            if len(seg) < L: seg = np.pad(seg, (0, L - len(seg)))

            # CNN용 2D 스펙트로그램 특징 추출
            # spec = audio_segment_to_melspec(seg, sr)
            # spec_input = spec[np.newaxis, ..., np.newaxis]
            #
            # CNN 모델로 예측
            # proba = model.predict(spec_input)[0]
            # lab_id = int(proba.argmax())
            # lab = LABELS[lab_id]

            # TFLite 입력 형식에 맞게 차원 확장 및 타입 변경
            spec_input = spec[np.newaxis, ..., np.newaxis].astype(np.float32)

            # --- TFLite 모델로 예측 ---
            interpreter.set_tensor(input_details[0]['index'], spec_input)
            interpreter.invoke()
            proba = interpreter.get_tensor(output_details[0]['index'])[0]
            # --- 예측 완료 ---

            lab_id = int(proba.argmax())
            lab = LABELS[lab_id]

            events.append((float(t), lab, float(proba[lab_id])))

        # (CSV 및 MIDI 생성 로직은 동일)
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
        print(f"MIDI 생성 오류 (job: {job_id}): {e}")
        import traceback
        traceback.print_exc()
        return False


# --- [수정] 전체 오디오 처리 파이프라인 (Demucs 연동) ---
def run_processing_pipeline(job_id, audio_path):
    """드럼 분리부터 MIDI 생성까지 전체 프로세스를 실행합니다."""
    from app.tasks import update_job_status  # 순환 참조 방지

    result_dir = os.path.join(current_app.config['RESULT_FOLDER'], job_id)
    os.makedirs(result_dir, exist_ok=True)

    # --- 1. 드럼 분리 (Demucs 실행) ---
    update_job_status(job_id, 'processing', '드럼 트랙을 분리하는 중입니다...')
    print(f"[{job_id}] Demucs 음원 분리 시작...")

    # time.sleep(5) 대신 실제 Demucs 실행 함수 호출
    separated_drum_path = run_demucs_separation(audio_path, result_dir, job_id)

    # Demucs 실행 실패 시 작업 중단
    if not separated_drum_path:
        update_job_status(job_id, 'error', '드럼 트랙 분리(Demucs) 중 오류가 발생했습니다.')
        print(f"[{job_id}] 작업 실패: Demucs 실행 오류.")
        return  # 함수 종료

    print(f"[{job_id}] 드럼 분리 완료. 분리된 파일: {separated_drum_path}")

    #
    # ★★★★★
    # 다음 단계(MIDI 생성)에서 사용할 파일은 원본이 아닌
    # 방금 분리된 'separated_drum_path' 입니다.
    # ★★★★★
    #

    # --- 2. BPM 분석 ---
    # (BPM 분석은 원본 오디오로 수행하는 것이 더 정확합니다)
    update_job_status(job_id, 'processing', '음원 템포(BPM)를 분석하는 중입니다...')
    print(f"[{job_id}] BPM 분석 시작 (원본 파일 기준)...")
    try:
        y, sr = librosa.load(audio_path, sr=SR)  # 원본(audio_path) 사용
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(tempo)
        print(f"[{job_id}] 분석된 BPM: {bpm}")
    except Exception as e:
        bpm = 120
        print(f"[{job_id}] BPM 분석 실패: {e}. 기본값 {bpm}으로 설정.")

    # --- 3. MIDI 생성 ---
    update_job_status(job_id, 'processing', '드럼 노트를 MIDI 파일로 변환하는 중입니다...')
    print(f"[{job_id}] MIDI 생성 시작 (분리된 드럼 파일 기준)...")

    # 분리된 드럼 파일(separated_drum_path)을 전달
    success = generate_midi_from_audio(separated_drum_path, result_dir, bpm)

    # --- 4. 최종 결과 업데이트 ---
    if success:
        results = {
            "midiUrl": f"/download/midi/{job_id}",
            "pdfUrl": f"/download/pdf/{job_id}",  # PDF 기능은 아직 미구현
        }
        update_job_status(job_id, 'completed', '작업이 완료되었습니다.', results)
        print(f"[{job_id}] 모든 작업 완료.")
    else:
        update_job_status(job_id, 'error', 'MIDI 생성 중 오류가 발생했습니다.')
        print(f"[{job_id}] 작업 실패.")