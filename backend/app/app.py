from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from io import BytesIO
import time  # 시간 지연을 위한 time 모듈 추가

app = Flask(__name__)

# -------------------------------------------------------------
# CORS 설정: Content-Disposition 헤더 노출
CORS(app, expose_headers=['Content-Disposition'])

# 파일 수정 및 다운로드 엔드포인트
@app.route('/api/download_modified', methods=['POST'])
def download_modified_file():
    if 'file' not in request.files:
        # 파일이 없을 경우, JSON 오류 메시지와 400 상태 코드를 반환
        return jsonify({"error": "파일을 선택해주세요."}), 400
    
    file = request.files['file']

    try:
        # ===============================================
        # 1. 5초 지연 시간 추가 (음악 분석/처리 시뮬레이션)
        print("서버: 5초 동안 의도적으로 작업을 지연합니다 (음악 분석 시뮬레이션)...")
        time.sleep(5) 
        print("서버: 지연 시간 종료. 파일 처리 시작.")
        # ===============================================

        # 2. 파일 내용 읽기 (실제 로직에서는 오디오 파일을 처리합니다.)
        # 현재는 오디오 파일 처리가 아니므로, 파일을 읽지 않고 다운로드용 가상 데이터만 준비합니다.
        
        # 3. 수정된 내용을 메모리상의 파일로 생성 (악보 파일 시뮬레이션)
        # 실제 MIDI 파일 데이터 대신 가상 텍스트 데이터 생성
        original_filename = file.filename
        content_prefix = f"// This is the generated score data for {original_filename}\n"
        modified_file_content = content_prefix + "Drum score: KICK, SNARE, HIHAT patterns...\n"
        
        # 가상 MIDI 파일 내용을 바이트로 인코딩
        download_data = BytesIO(modified_file_content.encode('utf-8'))
        
        # 4. 다운로드될 파일 이름 설정 (핵심 수정)
        
        # 오디오 파일 이름에서 확장자를 제거하고 "_score.midi"를 추가합니다.
        if '.' in original_filename:
            name_part = original_filename.rsplit('.', 1)[0]
            download_filename = f"{name_part}_score.midi"
        else:
             download_filename = original_filename + "_score.midi" # 확장자가 없는 경우 처리

        # 5. 파일을 응답으로 전송
        return send_file(
            download_data,
            # MIME 타입은 악보 파일 형식(MIDI)을 시뮬레이션합니다.
            mimetype='audio/midi', 
            as_attachment=True,    
            download_name=download_filename # 수정된 악보 파일명 지정
        )

    except Exception as e:
        # 예외 발생 시 JSON 오류 메시지 반환
        print(f"Server Error: {e}") 
        return jsonify({"error": f"서버 파일 처리 중 오류 발생: {str(e)}"}), 500

if __name__ == '__main__':
    # 서버 실행: Ctrl+C로 종료 후 이 코드를 다시 실행해야 합니다.
    app.run(debug=True, port=5000)