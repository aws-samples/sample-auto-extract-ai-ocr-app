from background import BackgroundTaskExtension
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.ocr_service import OcrService
from services.upload_service import UploadService
from services.extraction_service import ExtractionService
from services.schema_service import SchemaService
from services.s3_sync_service import S3SyncService
from services.agent_service import AgentService
from services.admin_service import AdminService
from services.user_service import UserService
from services.sharing_service import SharingService

from routers import health, images, jobs, system, tools, apps
from routers import admin, user, sharing

# アプリケーション全体のログレベル設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# バックグラウンドタスク拡張機能を初期化
background_task = BackgroundTaskExtension()

# 全サービスを app.state に集約
app.state.ocr_service = OcrService()
app.state.upload_service = UploadService(background_task)
app.state.extraction_service = ExtractionService(background_task)
app.state.schema_service = SchemaService()
app.state.s3_sync_service = S3SyncService(upload_service=app.state.upload_service)
app.state.agent_service = AgentService()
app.state.admin_service = AdminService()
app.state.user_service = UserService()
app.state.sharing_service = SharingService()

# CORS 設定
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
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


# リクエスト完了時にバックグラウンドタスクに通知するミドルウェア
@app.middleware("http")
async def send_done_message(request, call_next):
    response = await call_next(request)
    background_task.done()
    return response
