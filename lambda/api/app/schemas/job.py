from pydantic import BaseModel


class JobStartResponse(BaseModel):
    """ジョブ開始レスポンス"""
    jobId: str
