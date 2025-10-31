# backend/run.py
from app import create_app

app = create_app()

if __name__ == '__main__':
    # [수정] host='0.0.0.0' 추가
    app.run(host='0.0.0.0', debug=True, port=5000)