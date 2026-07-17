"""API エラーレスポンスを `{"detail": "<表示メッセージ>", "code": "<機械判定コード|null>"}` の1形に統一する。"""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
# フレームワークが投げる 404/405 は starlette の HTTPException なので base で登録する
# （fastapi.HTTPException はそのサブクラス）。
from starlette.exceptions import HTTPException as StarletteHTTPException

from exceptions import AppError

logger = logging.getLogger(__name__)

# 内部情報を出さないための汎用 5xx メッセージ
_INTERNAL_MESSAGE = "予期しないエラーが発生しました"


def _format_validation_message(errors: list) -> str:
    """pydantic の検証エラーを表示用の1メッセージ（改行区切り）に集約する。"""
    lines = []
    for e in errors:
        if e.get("type") == "missing":
            lines.append("必須項目が入力されていません")
            continue
        msg = e.get("msg", "")
        # pydantic は ValueError を "Value error, <本文>" として返すため接頭辞を除く
        msg = msg.replace("Value error, ", "", 1) if msg.startswith("Value error, ") else msg
        if msg:
            lines.append(msg)
    return "\n".join(lines) if lines else "入力内容が不正です"


def register_error_handlers(app: FastAPI) -> None:
    """エラーハンドラを登録する。

    @app.exception_handler で登録すると CORSMiddleware の内側で実行され、
    エラーレスポンスにも CORS ヘッダが付く（独自 middleware にすると CORS が落ちる）。
    """

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        # code を持たない 5xx はサーバー障害なので内部情報を隠す
        if exc.status_code >= 500 and not exc.code:
            logger.error("5xx AppError at %s: %r", request.url.path, str(exc))
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": _INTERNAL_MESSAGE, "code": "internal_error"},
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc), "code": exc.code},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        detail = exc.detail
        headers = getattr(exc, "headers", None)

        # dict detail は {"error": code, "message": text} 形として code/message を取り出す。
        # 5xx で通すのは code を持つ場合のみ（それ以外の 5xx は下でマスクして内部情報を守る）。
        if isinstance(detail, dict) and (exc.status_code < 500 or detail.get("error")):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": detail.get("message") or str(detail), "code": detail.get("error")},
                headers=headers,
            )

        # code を持たない 5xx はサーバー障害なので内部情報を隠す
        if exc.status_code >= 500:
            logger.error("5xx HTTPException at %s: %r", request.url.path, detail)
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": _INTERNAL_MESSAGE, "code": "internal_error"},
                headers=headers,
            )

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(detail), "code": None},
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # 生の検証エラーはデバッグ用にログへ残す（クライアントには集約文字列のみ返す）
        logger.info("Validation error at %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content={"detail": _format_validation_message(exc.errors()), "code": "validation_error"},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # 想定外の例外の最終防御。内部情報はログのみに残す
        logger.exception("Unhandled exception at %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": _INTERNAL_MESSAGE, "code": "internal_error"},
        )
