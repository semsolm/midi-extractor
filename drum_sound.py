import subprocess
import pathlib

# mp3 파일 경로
FILE = "drum.mp3"

# demucs 실행 (드럼 소리 분리)
subprocess.run(["demucs", FILE])

# 결과 폴더에서 drums.wav 찾기
song_dir = list(pathlib.Path("separated/htdemucs").glob("*"))[0]
DRUMS = str(song_dir / "drums.wav")

print("드럼 파일 경로:", DRUMS)
