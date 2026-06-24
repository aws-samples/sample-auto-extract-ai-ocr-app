from typing import Literal
from pydantic import BaseModel


class ShareUserRequest(BaseModel):
    user_id: str
    permission: Literal["viewer", "editor", "owner"] = "viewer"


class ShareGroupRequest(BaseModel):
    group_id: str
    permission: Literal["viewer", "editor"] = "viewer"
