# app/services/audio_processor.py
import os
import numpy as np
import librosa
import joblib
import pretty_midi
import csv
import time
from flask import current_app

# --- AI 모델 로드 및 상수 정의 ---
SR = 44100
NOTE_MAP = {"kick": 36, "snare": 38}


class MockModel:
    def __init__(self):
        self.classes_ = [0, 1]

    def predict_proba(self, X):
        return np.random.rand(1, 2)


def load_model(model_path):
    """모델 파일을 로드하거나 없으면 가상 모델을 생성합니다."""
    if not os.path.exists(model_path):
        print(f"경고: 모델 파일 '{model_path}'를 찾을 수 없습니다. 가상 모델을 생성합니다.")
        joblib.dump(MockModel(), model_path)
    return joblib.load(model_path)


# --- 특징 벡터 추출 함수 ---
def feature_vector_from_wav(y, sr):
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64, power=2.0)
    Mdb = librosa.power_to_db(M, ref=np.max)
    m_mean = Mdb.mean(axis=1);
    m_std = Mdb.std(axis=1)
    dM = librosa.feature.delta(Mdb);
    d_mean = dM.mean(axis=1)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()
    return np.concatenate([m_mean, m_std, d_mean, [sc, roll, zcr, rms]]).astype(np.float32)


# --- MIDI 생성 메인 함수 ---
def generate_midi_from_audio(audio_path, result_dir, bpm=120):
    """오디오 파일로부터 MIDI와 CSV를 생성하는 함수."""
    pipe = load_model(current_app.config['MODEL_PATH'])
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

            f = feature_vector_from_wav(seg, sr)
            proba = pipe.predict_proba([f])[0]
            lab_id = int(proba.argmax())
            lab = "kick" if pipe.classes_[lab_id] == 0 else "snare"
            events.append((float(t), lab, float(proba[lab_id])))

        with open(csv_out, "w", newline="") as f:
            w = csv.writer(f);
            w.writerow(["time_sec", "label", "prob"])
            for t, lab, p in events: w.writerow([f"{t:.4f}", lab, f"{p:.3f}"])

        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)
        for t, lab, p in events:
            pitch = NOTE_MAP[lab]
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=t, end=t + 0.1)
            drum_instrument.notes.append(note)
        pm.instruments.append(drum_instrument)
        pm.write(midi_out)

        return True
    except Exception as e:
        print(f"MIDI 생성 오류 (job: {job_id}): {e}")
        return False


# --- 전체 오디오 처리 파이프라인 ---
def run_processing_pipeline(job_id, audio_path):
    """드럼 분리부터 MIDI 생성까지 전체 프로세스를 실행합니다."""
    from app.tasks import update_job_status  # 순환 참조 방지

    result_dir = os.path.join(current_app.config['RESULT_FOLDER'], job_id)
    os.makedirs(result_dir, exist_ok=True)

    # 1. 드럼 분리 (시뮬레이션)
    update_job_status(job_id, 'processing', '드럼 트랙을 분리하는 중입니다...')
    print(f"[{job_id}] 드럼 분리 시작...")
    time.sleep(5)  # demucs 실행을 5초 대기로 가정
    print(f"[{job_id}] 드럼 분리 완료.")

    # 2. BPM 분석
    update_job_status(job_id, 'processing', '음원 템포(BPM)를 분석하는 중입니다...')
    print(f"[{job_id}] BPM 분석 시작...")
    try:
        y, sr = librosa.load(audio_path, sr=SR)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(tempo)
        print(f"[{job_id}] 분석된 BPM: {bpm}")
    except Exception as e:
        bpm = 120
        print(f"[{job_id}] BPM 분석 실패: {e}. 기본값 {bpm}으로 설정.")

    # 3. MIDI 생성
    update_job_status(job_id, 'processing', '드럼 노트를 MIDI 파일로 변환하는 중입니다...')
    print(f"[{job_id}] MIDI 생성 시작...")
    success = generate_midi_from_audio(audio_path, result_dir, bpm)

    # 4. 최종 결과 업데이트
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