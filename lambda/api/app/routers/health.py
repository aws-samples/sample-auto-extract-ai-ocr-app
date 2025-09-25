from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/")
def read_root():
    """ルートエンドポイント - API稼働確認"""
    return {"message": "API is running"}


@router.get("/health")
def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "ok"}
