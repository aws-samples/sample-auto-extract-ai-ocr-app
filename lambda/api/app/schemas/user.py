from pydantic import BaseModel


class UpdateProfileRequest(BaseModel):
    display_name: str
