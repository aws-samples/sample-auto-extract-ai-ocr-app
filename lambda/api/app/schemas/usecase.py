"""Schemas for usecase-related requests"""
from pydantic import BaseModel


class UsecaseToolsUpdate(BaseModel):
    tool_ids: list[str]
