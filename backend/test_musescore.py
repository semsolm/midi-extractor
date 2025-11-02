import music21 as m21
import os
import traceback
import subprocess # [신규] 서브프로세스 모듈 임포트
import time

print("--- MuseScore 연동 진단 테스트 시작 (수동 변환 방식) ---")

# --- 1. 경로 설정 ---
# [유지] MuseScore 3 경로를 사용합니다.
musescore_path = r'C:/Program Files/MuseScore 3/bin/MuseScore3.exe' 

print(f"테스트할 MuseScore 경로: {musescore_path}")

# --- 2. 경로 존재 여부 확인 ---
if not os.path.exists(musescore_path):
    print("\n[오류] ❌ 해당 경로에 MuseScore 실행 파일이 존재하지 않습니다.")
    exit()

print("파일 시스템에서 경로를 확인했습니다.")

# --- 3. music21 환경 설정 (불필요하지만 확인용으로 유지) ---
try:
    us = m21.environment.UserSettings()
    us['musicxmlPath'] = musescore_path
    print("music21 환경 변수 설정 완료 (참고용).")
except Exception as e:
    print(f"\n[경고] ❌ music21 환경 설정 중 오류 발생: {e}")
    pass # 계속 진행

# --- 4. PDF 변환 테스트 (수동 방식) ---
xml_path = 'test_output.xml'
output_pdf = 'test_output.pdf'

try:
    print("간단한 악보(C4 노트)를 생성합니다...")
    s = m21.stream.Score()
    p = m21.stream.Part()
    p.insert(0, m21.instrument.Piano())
    p.insert(0, m21.clef.TrebleClef())
    p.append(m21.note.Note("C4", type='whole'))
    s.append(p)
    
    # [수정] 1. 'pdf'가 아닌 'musicxml'로 저장합니다.
    print(f"'{xml_path}' 파일로 MusicXML 변환을 시도합니다...")
    s.write('musicxml', fp=xml_path)
    print("MusicXML 파일 생성 성공.")
    
    # [수정] 2. subprocess로 MuseScore를 직접 호출하여 PDF로 변환합니다.
    print(f"MuseScore를 호출하여 '{output_pdf}' 파일 생성을 시도합니다...")
    
    # MuseScore 3 명령어: MuseScore3.exe -o [출력.pdf] [입력.xml]
    command = [musescore_path, '-o', output_pdf, xml_path]
    
    # 타임아웃 10초 설정
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)

    # [수정] 3. 결과 확인
    if result.returncode != 0:
        print("\n[오류] ❌ MuseScore 실행 중 오류가 발생했습니다.")
        print(f"Return Code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
    elif not os.path.exists(output_pdf):
        print("\n[오류] ❌ MuseScore는 성공했으나, PDF 파일이 생성되지 않았습니다.")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
    else:
        print(f"\n[성공] ✅ '{output_pdf}' 파일이 성공적으로 생성되었습니다!")
        print("MuseScore 연동에 성공했습니다. 이 로직을 백엔드에 적용합니다.")
        os.remove(output_pdf) # 테스트 파일 삭제
        
except Exception as e:
    print("\n[오류] ❌ 테스트 중 치명적인 오류가 발생했습니다:")
    print(traceback.format_exc())

finally:
    # 테스트 후 임시 XML 파일 삭제
    if os.path.exists(xml_path):
        os.remove(xml_path)

print("--- 진단 테스트 종료 ---")