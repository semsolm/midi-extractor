from pydub import AudioSegment
from pydub.generators import Sine
import os


def create_dummy_mp3(output_folder="..", output_filename="drum.mp3"):
    """
    간단한 킥과 스네어 사운드를 조합하여 더미 드럼 비트 MP3 파일을 생성합니다.
    """
    output_path = os.path.join(output_folder, output_filename)
    try:
        print(f"'{output_path}' 파일 생성을 시작합니다...")

        # (사운드 생성 및 조합 로직은 이전과 동일)
        kick_sound = Sine(60).to_audio_segment(duration=150).fade_out(100)
        snare_sound = Sine(4000).to_audio_segment(duration=150).fade_out(120)
        snare_sound = snare_sound.overlay(Sine(200).to_audio_segment(duration=150).fade_out(120))
        one_beat = AudioSegment.silent(duration=500)
        beat = AudioSegment.empty()
        beat += kick_sound
        beat = beat.append(one_beat, crossfade=0)
        beat += snare_sound
        beat = beat.append(one_beat, crossfade=0)
        beat += kick_sound
        beat = beat.append(one_beat, crossfade=0)
        beat += snare_sound
        beat = beat.append(one_beat, crossfade=0)

        beat.export(output_path, format="mp3")

        print(f"성공! 테스트용 더미 파일 '{output_path}'이 생성되었습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")
        print("pydub 라이브러리와 FFmpeg가 올바르게 설치되었는지 확인해주세요.")


if __name__ == "__main__":
    # 이 스크립트는 backend/utils/ 폴더에 있으므로,
    # drum.mp3 파일은 한 단계 위인 backend/ 폴더에 생성됩니다.
    create_dummy_mp3()