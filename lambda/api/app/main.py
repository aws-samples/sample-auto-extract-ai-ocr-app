import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.ocr_service import OcrService
from services.upload_service import UploadService
from services.image_list_service import ImageListService
from services.extraction_service import ExtractionService
from services.schema_service import SchemaService
from services.s3_sync_service import S3SyncService
from services.agent_service import AgentService
from services.admin_service import AdminService
from services.user_service import UserService
from services.sharing_service import SharingService

from routers import health, images, jobs, system, tools, apps
from routers import admin, user, sharing
from errors import register_error_handlers

# アプリケーション全体のログレベル設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# エラーレスポンスを統一形（{detail, code}）に正規化するハンドラを登録
register_error_handlers(app)

# 全サービスを app.state に集約
app.state.ocr_service = OcrService()
app.state.upload_service = UploadService()
app.state.image_list_service = ImageListService()
app.state.extraction_service = ExtractionService()
app.state.schema_service = SchemaService()
app.state.s3_sync_service = S3SyncService()
app.state.agent_service = AgentService()
app.state.admin_service = AdminService()
app.state.user_service = UserService()
app.state.sharing_service = SharingService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーター登録
app.include_router(health.router)
app.include_router(images.router)
app.include_router(jobs.router)
app.include_router(system.router)
app.include_router(tools.router)
app.include_router(apps.router)
app.include_router(admin.router)
app.include_router(user.router)
app.include_router(sharing.router)
