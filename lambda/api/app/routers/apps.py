"""Apps Router - /apps prefix

App CRUD, schema, s3-sync, usecase tools, batch jobs.
"""
from fastapi import APIRouter, HTTPException, Request, Depends
import logging
from typing import Optional

from schemas import (
    SchemaGenerateRequest,
    PresignedUrlRequest, CustomPromptRequest, SchemaSaveRequest,
    OcrStartRequest, JobStartResponse,
    S3ImportBatchRequest, UsecaseToolsUpdate,
    SchemaGenerateStartResponse, SchemaGenerateStatusResponse,
)
from services.schema_service import SchemaService
from services.s3_sync_service import S3SyncService
from services.ocr_service import OcrService
from dependencies.services import get_schema_service, get_s3_sync_service, get_ocr_service
from dependencies.auth import (
    require_user, get_cognito_sub,
    RequirePermission, RequireRole,
)
from repositories import tool_repository
from repositories.usecase_repository import get_usecase_by_app_name, register_usecase_owner
from repositories.schema_repository import get_app_schema

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Apps"])


# === App CRUD ===

@router.get("/apps")
async def get_apps(user=Depends(require_user), service: SchemaService = Depends(get_schema_service)):
    """アプリ一覧を取得する（権限のあるもののみ）"""
    return await service.get_apps_list(
        user_id=str(user["id"]),
        role=user["role"],
    )


@router.get("/apps/{app_name}")
async def get_app_details(app_name: str, user=Depends(RequirePermission("viewer")), service: SchemaService = Depends(get_schema_service)):
    """アプリ詳細を取得する（viewer 以上）"""
    return await service.get_app_details(app_name)


@router.get("/apps/{app_name}/fields")
async def get_app_fields(app_name: str, user=Depends(RequirePermission("viewer")), service: SchemaService = Depends(get_schema_service)):
    """アプリのフィールド一覧を取得する（viewer 以上）"""
    return await service.get_app_fields(app_name)


@router.get("/apps/{app_name}/custom-prompt")
async def get_custom_prompt(app_name: str, user=Depends(RequirePermission("viewer")), service: SchemaService = Depends(get_schema_service)):
    """カスタムプロンプトを取得する（viewer 以上）"""
    return await service.get_custom_prompt(app_name)


@router.put("/apps/{app_name}/custom-prompt")
async def update_custom_prompt(app_name: str, request: CustomPromptRequest, user=Depends(RequirePermission("editor")), service: SchemaService = Depends(get_schema_service)):
    """カスタムプロンプトを更新する（editor 以上）"""
    await service.update_custom_prompt(app_name, request)
    return {"status": "success", "message": "Custom prompt updated successfully"}


@router.post("/apps")
async def create_app(request: SchemaSaveRequest, user=Depends(RequireRole("author")), service: SchemaService = Depends(get_schema_service)):
    """アプリを新規作成する（author 以上）"""
    return await service.save_schema(request, user_id=str(user["id"]))


@router.put("/apps/{app_name}")
async def update_app(app_name: str, request: SchemaSaveRequest, user=Depends(RequirePermission("editor")), service: SchemaService = Depends(get_schema_service)):
    """既存アプリを更新する（editor 以上）"""
    return await service.update_schema(app_name, request)


@router.delete("/apps/{app_name}")
async def delete_app(app_name: str, user=Depends(RequirePermission("owner")), service: SchemaService = Depends(get_schema_service)):
    """アプリを削除する（owner 以上）"""
    await service.delete_app(app_name)
    return {"status": "success", "message": f"App '{app_name}' deleted successfully"}


# === Schema ===

@router.post("/apps/schema/upload-url")
async def generate_app_schema_presigned_url(request: PresignedUrlRequest, user=Depends(RequireRole("author")), service: SchemaService = Depends(get_schema_service)):
    """アプリスキーマ用の署名付きURLを生成する（author 以上）"""
    return await service.generate_schema_presigned_url(request)


@router.get("/apps/{app_name}/sample-image-url")
async def get_app_sample_image_url(app_name: str, user=Depends(RequirePermission("viewer")), service: SchemaService = Depends(get_schema_service)):
    """スキーマに紐づくサンプル画像の署名付き GET URL を取得する（viewer 以上）。

    サンプル画像が未紐付けの場合は {"url": null} を返す。
    """
    return await service.get_sample_image_url(app_name)


@router.post("/apps/{app_name}/schema/generate", response_model=SchemaGenerateStartResponse)
async def generate_app_schema(app_name: str, request: SchemaGenerateRequest, user=Depends(RequireRole("author")), service: SchemaService = Depends(get_schema_service)):
    """アプリのスキーマを自動生成する（非同期・author 以上）。

    Bedrock 呼び出しに 40-50 秒かかり API Gateway 29 秒制限を超えるため、
    ジョブを作成して Worker Lambda を非同期起動、job_id を即返却する。
    フロントは GET /apps/schema/generate/{job_id} で結果をポーリングする。

    app_name は URL 上のみで、実際のスキーマ生成では使わない (生成後にユーザーが指定)。
    """
    return await service.start_schema_generation(request)


@router.get("/apps/schema/generate/{job_id}", response_model=SchemaGenerateStatusResponse)
async def get_app_schema_generation_status(job_id: str, user=Depends(RequireRole("author")), service: SchemaService = Depends(get_schema_service)):
    """スキーマ生成ジョブの状態を取得する（author 以上）。

    フロントからのポーリング用。
    - processing: 処理中
    - completed: 完了 (result に {"fields": [...]} が入る)
    - failed: 失敗 (error にメッセージ)
    """
    return await service.get_schema_generation_result(job_id)


# === Batch Jobs ===

@router.post("/apps/{app_name}/jobs", response_model=JobStartResponse)
async def start_batch_job(
    app_name: str,
    user=Depends(RequirePermission("viewer")),
    service: OcrService = Depends(get_ocr_service),
):
    """バッチ一括パイプライン起動（OCR→抽出→Agent）"""
    request = OcrStartRequest(app_name=app_name)
    result = await service.start_step_functions_job(request)
    return JobStartResponse(jobId=result["jobId"])


# === S3 Sync ===

@router.post("/apps/{app_name}/s3-sync")
async def sync_s3_files(
    app_name: str,
    prefix: Optional[str] = None,
    user=Depends(RequirePermission("viewer")),
    service: S3SyncService = Depends(get_s3_sync_service),
):
    """S3バケットからファイルを同期する"""
    return await service.sync_s3_files(app_name, prefix)


@router.post("/apps/{app_name}/s3-sync/import")
async def import_s3_files(
    app_name: str,
    body: S3ImportBatchRequest,
    req: Request,
    user=Depends(RequirePermission("editor")),
    service: S3SyncService = Depends(get_s3_sync_service),
):
    """S3バケットから複数ファイルを一括インポートする（重い処理は Worker に委譲）"""
    sub = get_cognito_sub(req)
    return await service.import_s3_files_batch(
        app_name,
        [f.model_dump() for f in body.files],
        body.page_processing_mode,
        uploaded_by=sub,
    )


@router.get("/apps/{app_name}/s3-sync/files")
async def list_s3_files(
    app_name: str,
    prefix: Optional[str] = None,
    user=Depends(RequirePermission("viewer")),
    service: S3SyncService = Depends(get_s3_sync_service),
):
    """S3ファイル一覧を重複チェック付きで取得する"""
    return await service.get_files_with_duplicate_check(app_name, prefix)


# === Usecase Tools ===

def _resolve_usecase_id(app_name: str, user_id: str | None = None) -> str:
    usecase = get_usecase_by_app_name(app_name)
    if usecase:
        return str(usecase["id"])

    schema = get_app_schema(app_name)
    if not schema:
        raise HTTPException(404, f"Usecase not found: {app_name}")

    owner_id = user_id or "00000000-0000-0000-0000-000000000000"
    logger.info(f"Auto-registering pre-existing usecase in DSQL: {app_name}")
    register_usecase_owner(app_name, owner_id)

    usecase = get_usecase_by_app_name(app_name)
    if not usecase:
        raise HTTPException(500, f"Failed to auto-register usecase: {app_name}")
    return str(usecase["id"])


@router.get("/apps/{app_name}/tools")
async def get_usecase_tools(app_name: str, user=Depends(RequirePermission("editor"))):
    """ユースケースに現在割当済みのツール一覧（is_active=true のみ）"""
    usecase_id = _resolve_usecase_id(app_name, user_id=str(user["id"]))
    return {"tools": tool_repository.get_usecase_tools(usecase_id, active_only=True)}


@router.put("/apps/{app_name}/tools")
async def set_usecase_tools(
    app_name: str,
    body: UsecaseToolsUpdate,
    user=Depends(RequirePermission("editor")),
):
    """ユースケースのツールを一括設定"""
    usecase_id = _resolve_usecase_id(app_name, user_id=str(user["id"]))

    if user["role"] != "admin":
        visible = tool_repository.get_visible_tools_for_user(str(user["id"]))
        visible_ids = {str(t["id"]) for t in visible}

        current_tools = tool_repository.get_usecase_tools(usecase_id, active_only=False)
        current_ids = {str(t["id"]) for t in current_tools}

        newly_added = [tid for tid in body.tool_ids if tid not in current_ids]
        invalid = [tid for tid in newly_added if tid not in visible_ids]
        if invalid:
            raise HTTPException(403, f"Cannot assign tools not visible to you: {invalid}")

    tool_repository.set_usecase_tools(usecase_id, body.tool_ids)
    return {"ok": True}


@router.get("/apps/{app_name}/available-tools")
async def get_available_tools(app_name: str, user=Depends(RequirePermission("editor"))):
    """ログインユーザーが選択可能なツール一覧"""
    _resolve_usecase_id(app_name, user_id=str(user["id"]))

    if user["role"] == "admin":
        return {"tools": tool_repository.list_tools()}
    return {"tools": tool_repository.get_visible_tools_for_user(str(user["id"]))}
