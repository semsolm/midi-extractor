# app/__init__.py
from flask import Flask
from config import Config

def create_app():
    """Flask 애플리케이션 팩토리 함수."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # 설정 클래스의 init_app 메서드를 호출하여 필요한 폴더 생성
    Config.init_app(app)

    # 라우트(API 엔드포인트) 등록
    from . import routes
    app.register_blueprint(routes.bp)

    return app