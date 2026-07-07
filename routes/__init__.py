"""注册所有 Flask 蓝图。"""
from flask import Flask


def register_routes(app: Flask) -> None:
    from routes.pages import bp as pages_bp
    from routes.service_routes import bp as service_bp
    from routes.send_routes import bp as send_bp
    from routes.chat_routes import bp as chat_bp
    from routes.settings_routes import bp as settings_bp
    from routes.observability_routes import bp as observability_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(service_bp)
    app.register_blueprint(send_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(observability_bp)
