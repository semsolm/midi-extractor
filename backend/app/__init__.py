# app/__init__.py
from flask import Flask
from config import Config
from flask_cors import CORS

def create_app():
    """Flask 애플리케이션 팩토리 함수."""
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app, resources={r"/api/*": {"origins": "*"}})  # [2] CORS 적용
    # (더 엄격하게 하려면 "origins": "http://localhost:3000"로 설정)

    # 설정 클래스의 init_app 메서드를 호출하여 필요한 폴더 생성
    Config.init_app(app)

    # 라우트(API 엔드포인트) 등록
    from . import routes
    app.register_blueprint(routes.bp)

    return app