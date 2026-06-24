from typing import Optional, Literal
from pydantic import BaseModel


class UserRoleUpdate(BaseModel):
    role: Literal["admin", "author", "reader"]


class GroupCreate(BaseModel):
    name: str
    description: Optional[str] = None


class GroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class GroupMemberUpdate(BaseModel):
    user_ids: list[str]


class ToolCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class ToolUserBody(BaseModel):
    user_id: str


class ToolGroupBody(BaseModel):
    group_id: str
