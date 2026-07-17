"""API エラーレスポンスを `{"detail": "<表示メッセージ>", "code": "<機械判定コード|null>"}` の1形に統一する。

既存の `raise HTTPException(...)` は変更せず、ここでレスポンスを包む。
"""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
# fastapi.HTTPException はこのサブクラス。フレームワークが投げる 404/405 は
# 純 starlette の HTTPException なので、必ず starlette の base で登録する。
from starlette.exceptions import HTTPException as StarletteHTTPException

from exceptions import AppError

logger = logging.getLogger(__name__)

# クライアントに内部情報を出さないための汎用 5xx メッセージ
_INTERNAL_MESSAGE = "予期しないエラーが発生しました"


def _format_validation_message(errors: list) -> str:
    """pydantic の検証エラー list を表示用の1メッセージに集約する。

    loc（フィールドパス）は使わず msg のみを日本語向けに整形して改行結合する。
    """
    lines = []
    for e in errors:
        if e.get("type") == "missing":
            lines.append("必須項目が入力されていません")
            continue
        msg = e.get("msg", "")
        # pydantic は ValueError を "Value error, <本文>" として返すため接頭辞を除去
        msg = msg.replace("Value error, ", "", 1) if msg.startswith("Value error, ") else msg
        if msg:
            lines.append(msg)
    return "\n".join(lines) if lines else "入力内容が不正です"


def register_error_handlers(app: FastAPI) -> None:
    """アプリにエラーハンドラを登録する（main.py から呼ぶ）。

    @app.exception_handler で登録するため、これらは Starlette の ExceptionMiddleware
    層（CORSMiddleware の内側）で実行され、エラーレスポンスにも CORS ヘッダが付く。
    エラー処理を独自 middleware として実装すると CORS ヘッダが落ちるため、必ずここで行う。
    """

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        # code を持つ 5xx（endpoint_not_ready 等）は意図的なエラーなので message/code を保持。
        # code 無しの 5xx（＝素の AppError 等）だけ内部情報を隠して汎用メッセージにする。
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

        # {"error": "...", "message": "..."} 形（endpoint_not_ready 等）は、機械判定コードと
        # 表示メッセージを保持する。ただし 5xx で保持するのは明示コード(error)を持つ意図的な
        # エラーのみ。error 無しの 5xx dict は下の 5xx マスクに流し、内部情報の漏洩を防ぐ。
        if isinstance(detail, dict) and (exc.status_code < 500 or detail.get("error")):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": detail.get("message") or str(detail), "code": detail.get("error")},
                headers=headers,
            )

        # 明示コードを持たない 5xx は内部情報を隠して汎用メッセージに（元 detail はログのみ）
        if exc.status_code >= 500:
            logger.error("5xx HTTPException at %s: %r", request.url.path, detail)
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": _INTERNAL_MESSAGE, "code": "internal_error"},
                headers=headers,
            )

        # 4xx の string detail（大多数）
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(detail), "code": None},
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # 生の検証エラーはデバッグ用にサーバーログへ残す（クライアントには集約文字列のみ）
        logger.info("Validation error at %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content={"detail": _format_validation_message(exc.errors()), "code": "validation_error"},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # router の except で拾えなかった例外（依存解決・middleware 等）の最終防御
        logger.exception("Unhandled exception at %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": _INTERNAL_MESSAGE, "code": "internal_error"},
        )
