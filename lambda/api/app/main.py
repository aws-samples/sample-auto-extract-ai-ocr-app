from routers.extraction import set_background_task as set_extraction_background_task
from routers.ocr import set_background_task
from background import BackgroundTaskExtension
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import health, ocr, upload, extraction, schema, s3_sync

# アプリケーション全体のログレベル設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# バックグラウンドタスク拡張機能を初期化
background_task = BackgroundTaskExtension()

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
app.include_router(ocr.router)
app.include_router(upload.router)
app.include_router(extraction.router)
app.include_router(schema.router)
app.include_router(s3_sync.router)

# バックグラウンドタスクをサービスに注入
set_background_task(background_task)
set_extraction_background_task(background_task)


# リクエスト完了時にバックグラウンドタスクに通知するミドルウェア
@app.middleware("http")
async def send_done_message(request, call_next):
    response = await call_next(request)
    background_task.done()
    return response
