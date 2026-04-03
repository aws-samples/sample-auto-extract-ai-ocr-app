"""サービス DI（FastAPI Depends 経由で各 Router に注入）"""
from fastapi import Request

from services.ocr_service import OcrService
from services.upload_service import UploadService
from services.extraction_service import ExtractionService
from services.schema_service import SchemaService
from services.s3_sync_service import S3SyncService
from services.agent_service import AgentService
from services.admin_service import AdminService
from services.user_service import UserService
from services.sharing_service import SharingService


def get_ocr_service(request: Request) -> OcrService:
    return request.app.state.ocr_service


def get_upload_service(request: Request) -> UploadService:
    return request.app.state.upload_service


def get_extraction_service(request: Request) -> ExtractionService:
    return request.app.state.extraction_service


def get_schema_service(request: Request) -> SchemaService:
    return request.app.state.schema_service


def get_s3_sync_service(request: Request) -> S3SyncService:
    return request.app.state.s3_sync_service


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_admin_service(request: Request) -> AdminService:
    return request.app.state.admin_service


def get_user_service(request: Request) -> UserService:
    return request.app.state.user_service


def get_sharing_service(request: Request) -> SharingService:
    return request.app.state.sharing_service
