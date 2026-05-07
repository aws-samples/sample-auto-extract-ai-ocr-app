from fastapi import APIRouter, HTTPException, Depends
import logging

from schemas import (
    SchemaGenerateRequest,
    PresignedUrlRequest, CustomPromptRequest, SchemaSaveRequest
)
from services.schema_service import SchemaService
from dependencies.services import get_schema_service
from dependencies.auth import (
    require_auth, RequirePermission, RequireRole,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Schema & Apps"])


@router.get("/apps")
async def get_apps(user=Depends(require_auth), service: SchemaService = Depends(get_schema_service)):
    """アプリ一覧を取得する（権限のあるもののみ）"""
    try:
        return await service.get_apps_list(
            user_id=str(user["id"]),
            role=user["role"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting apps list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/apps/{app_name}")
async def get_app_details(app_name: str, user=Depends(RequirePermission("viewer")), service: SchemaService = Depends(get_schema_service)):
    """アプリ詳細を取得する（viewer 以上）"""
    try:
        return await service.get_app_details(app_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting app details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/apps/{app_name}/fields")
async def get_app_fields(app_name: str, user=Depends(RequirePermission("viewer")), service: SchemaService = Depends(get_schema_service)):
    """アプリのフィールド一覧を取得する（viewer 以上）"""
    try:
        return await service.get_app_fields(app_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting app fields: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/apps/{app_name}/custom-prompt")
async def get_custom_prompt(app_name: str, user=Depends(RequirePermission("viewer")), service: SchemaService = Depends(get_schema_service)):
    """カスタムプロンプトを取得する（viewer 以上）"""
    try:
        return await service.get_custom_prompt(app_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting custom prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.put("/apps/{app_name}/custom-prompt")
async def update_custom_prompt(app_name: str, request: CustomPromptRequest, user=Depends(RequirePermission("editor")), service: SchemaService = Depends(get_schema_service)):
    """カスタムプロンプトを更新する（editor 以上）"""
    try:
        await service.update_custom_prompt(app_name, request)
        return {"status": "success", "message": "Custom prompt updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating custom prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/apps")
async def create_app(request: SchemaSaveRequest, user=Depends(RequireRole("author")), service: SchemaService = Depends(get_schema_service)):
    """アプリを新規作成する（author 以上）"""
    try:
        return await service.save_schema(request, user_id=str(user["id"]))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating app: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/apps/{app_name}")
async def update_app(app_name: str, request: SchemaSaveRequest, user=Depends(RequirePermission("editor")), service: SchemaService = Depends(get_schema_service)):
    """既存アプリを更新する（editor 以上）"""
    try:
        return await service.update_schema(app_name, request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating app: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/apps/{app_name}")
async def delete_app(app_name: str, user=Depends(RequirePermission("owner")), service: SchemaService = Depends(get_schema_service)):
    """アプリを削除する（owner 以上）"""
    try:
        await service.delete_app(app_name)
        return {"status": "success", "message": f"App '{app_name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting app: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/apps/schema/generate-presigned-url")
async def generate_app_schema_presigned_url(request: PresignedUrlRequest, user=Depends(RequireRole("author")), service: SchemaService = Depends(get_schema_service)):
    """アプリスキーマ用の署名付きURLを生成する（author 以上）"""
    try:
        return await service.generate_schema_presigned_url(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating app schema presigned URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/apps/{app_name}/schema/generate")
async def generate_app_schema(app_name: str, request: SchemaGenerateRequest, user=Depends(RequireRole("author")), service: SchemaService = Depends(get_schema_service)):
    """アプリのスキーマを自動生成する（author 以上）"""
    try:
        return await service.generate_schema(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating app schema: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
