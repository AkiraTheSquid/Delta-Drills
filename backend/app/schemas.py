from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    created_at: datetime


class JobOut(BaseModel):
    id: UUID
    status: str
    original_filename: str
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None


class ChapterOut(BaseModel):
    id: UUID
    title: str
    start_page: int
    end_page: int
    filename: str
    file_size: int
    created_at: datetime
