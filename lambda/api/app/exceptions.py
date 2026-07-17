"""アプリのドメイン例外。

services / repositories はこれらを raise し、errors.py のハンドラが
HTTP レスポンス（status / code / message）へマップする。
HTTPException は routers / dependencies のみで使い、ここには持ち込まない。

直接 `AppError` を raise しないこと（意味が曖昧な 500 になる）。用途に応じたサブクラスを使う。
"""


class AppError(Exception):
    """ドメイン例外の基底。status_code と機械判定用 code を持つ。"""
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str = ""):
        super().__init__(message)


class NotFoundError(AppError):
    """リソースが見つからない。"""
    status_code = 404
    code = "not_found"


class BadRequestError(AppError):
    """リクエストが不正（業務ルール上の入力エラー）。"""
    status_code = 400
    code = "bad_request"


class ForbiddenError(AppError):
    """権限不足。"""
    status_code = 403
    code = "forbidden"


class ConflictError(AppError):
    """リソースの状態競合。"""
    status_code = 409
    code = "conflict"


class LastOwnerError(AppError):
    """最後のオーナーを削除しようとした。"""
    status_code = 400
    code = "last_owner"


class EndpointNotReadyError(AppError):
    """OCR エンドポイントが起動中。"""
    status_code = 503
    code = "endpoint_not_ready"
