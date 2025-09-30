import numpy as np, librosa, csv, joblib, pretty_midi

MODEL_PATH = "kick_snare_lr.pkl"  # 방금 저장한 모델
AUDIO = "my_song_or_drums.wav"    # mp3/wav 어떤 파일이든 (가능하면 드럼만)
SR = 44100
PRE, POST = 0.04, 0.11            # 온셋 전/후 40ms/110ms
NOTE_MAP = {"kick":36, "snare":38}

pipe = joblib.load(MODEL_PATH)

def feats(y, sr):
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64, power=2.0)
    Mdb = librosa.power_to_db(M, ref=np.max)
    m_mean = Mdb.mean(axis=1); m_std = Mdb.std(axis=1)
    dM = librosa.feature.delta(Mdb); d_mean = dM.mean(axis=1)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()
    return np.concatenate([m_mean, m_std, d_mean, [sc, roll, zcr, rms]]).astype(np.float32)

def infer_to_csv_midi(audio_path, csv_out="preds_lr.csv", midi_out="preds.mid", bpm=120):
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    on = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)  # 민감도는 아래 “튜닝” 참고
    ts = librosa.frames_to_time(on, sr=sr)

    events=[]; L=int((PRE+POST)*sr)
    for t in ts:
        s=max(0,int((t-PRE)*sr)); e=min(len(y),int((t+POST)*sr))
        seg=y[s:e]
        if len(seg)<L: seg=np.pad(seg,(0,L-len(seg)))
        f=feats(seg, sr)
        proba=pipe.predict_proba([f])[0]
        lab_id=int(proba.argmax())
        lab = pipe.classes_[lab_id]   # 0=kick, 1=snare (훈련 때 그 순서)
        lab = "kick" if lab==0 else "snare"
        events.append((float(t), lab, float(proba[lab_id])))

    # CSV
    with open(csv_out,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["time_sec","label","prob"])
        for t,lab,p in events: w.writerow([f"{t:.4f}",lab,f"{p:.3f}"])
    print("CSV 저장:", csv_out, f"(예측 {len(events)}건)")

    # MIDI
    pm=pretty_midi.PrettyMIDI(initial_tempo=bpm)
    drum=pretty_midi.Instrument(program=0, is_drum=True)
    for t,lab,p in events:
        pitch=NOTE_MAP[lab]
        drum.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=t, end=t+0.05))
    pm.instruments.append(drum); pm.write(midi_out)
    print("MIDI 저장:", midi_out)

# 사용 예
# infer_to_csv_midi("separated/htdemucs/내파일/drums.wav")
# infer_to_csv_midi("my_youtube.mp3")  # 풀믹스도 가능(정확도는 ↓)