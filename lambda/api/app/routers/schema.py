from fastapi import APIRouter, HTTPException
import logging
from typing import Optional

from schemas import (
    AppCreateRequest, AppUpdateRequest, SchemaGenerateRequest,
    PresignedUrlRequest, CustomPromptRequest, SchemaSaveRequest
)
from services.schema_service import SchemaService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Schema & Apps"])

# スキーマサービスのインスタンス
schema_service = SchemaService()


# アプリ管理エンドポイント
@router.get("/apps")
async def get_apps():
    """アプリ一覧を取得する"""
    try:
        result = await schema_service.get_apps_list()
        return result
    except Exception as e:
        logger.error(f"Error getting apps list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/apps/{app_name}")
async def get_app_details(app_name: str):
    """アプリ詳細を取得する"""
    try:
        result = await schema_service.get_app_details(app_name)
        return result
    except Exception as e:
        logger.error(f"Error getting app details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/apps/{app_name}/fields")
async def get_app_fields(app_name: str):
    """アプリのフィールド一覧を取得する"""
    try:
        result = await schema_service.get_app_fields(app_name)
        return result
    except Exception as e:
        logger.error(f"Error getting app fields: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/apps/{app_name}/custom-prompt")
async def get_custom_prompt(app_name: str):
    """カスタムプロンプトを取得する"""
    try:
        result = await schema_service.get_custom_prompt(app_name)
        return result
    except Exception as e:
        logger.error(f"Error getting custom prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.put("/apps/{app_name}/custom-prompt")
async def update_custom_prompt(app_name: str, request: CustomPromptRequest):
    """カスタムプロンプトを更新する"""
    try:
        await schema_service.update_custom_prompt(app_name, request)
        return {"status": "success", "message": "Custom prompt updated successfully"}
    except Exception as e:
        logger.error(f"Error updating custom prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/apps")
async def create_app(app_data: dict):
    """新しいアプリを作成または更新する"""
    try:
        result = await schema_service.create_app(app_data)
        return result
    except Exception as e:
        logger.error(f"Error creating app: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/apps/{app_name}")
async def delete_app(app_name: str):
    """アプリを削除する"""
    try:
        await schema_service.delete_app(app_name)
        return {"status": "success", "message": f"App '{app_name}' deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting app: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# スキーマ管理エンドポイント
@router.post("/schema/save")
async def save_schema(request: SchemaSaveRequest):
    """スキーマを保存する"""
    try:
        result = await schema_service.save_schema(request)
        return result
    except Exception as e:
        logger.error(f"Error saving schema: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/schema/generate-presigned-url")
async def generate_schema_presigned_url(request: PresignedUrlRequest):
    """スキーマ用の署名付きURLを生成する"""
    try:
        result = await schema_service.generate_schema_presigned_url(request)
        return result
    except Exception as e:
        logger.error(f"Error generating schema presigned URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/schema/generate")
async def generate_schema(request: SchemaGenerateRequest):
    """スキーマを自動生成する"""
    try:
        result = await schema_service.generate_schema(request)
        return result
    except Exception as e:
        logger.error(f"Error generating schema: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.put("/schema/update/{app_name}")
async def update_schema(app_name: str, request: SchemaSaveRequest):
    """既存のスキーマを更新する"""
    try:
        result = await schema_service.update_schema(app_name, request)
        return result
    except Exception as e:
        logger.error(f"Error updating schema: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
