"""注册所有 FastAPI 路由。"""
from fastapi import FastAPI


def register_routes(app: FastAPI) -> None:
    from routes.pages import router as pages_router
    from routes.service_routes import router as service_router
    from routes.send_routes import router as send_router
    from routes.chat_routes import router as chat_router
    from routes.settings_routes import router as settings_router
    from routes.observability_routes import router as observability_router

    app.include_router(pages_router)
    app.include_router(service_router)
    app.include_router(send_router)
    app.include_router(chat_router)
    app.include_router(settings_router)
    app.include_router(observability_router)
