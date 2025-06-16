from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger


def setup_exception_handler(app: FastAPI):
    """
    모든 요청에 대해 전역 예외 처리를 적용하는 미들웨어를 설정합니다.

    이 미들웨어는 FastAPI 앱에 등록되어 각 요청에서 발생할 수 있는 예외를
    로깅하고, 클라이언트에게 일관된 JSON 형식의 500 에러 응답을 반환합니다.

    Args:
        app (FastAPI): 예외 미들웨어를 설정할 FastAPI 인스턴스

    """
    @app.middleware("http")
    async def catch_exceptions_middleware(request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception:
            logger.error("middleware error")
            return JSONResponse(
                status_code=500, content={"detail": "Internal Server Error"}
            )
