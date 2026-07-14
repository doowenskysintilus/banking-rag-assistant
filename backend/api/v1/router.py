"""Aggregates all v1 routers under a single APIRouter mounted by main.py."""

from fastapi import APIRouter

from backend.api.v1.routers import (
    audit,
    auth,
    chat,
    documents,
    evaluation,
    health,
    users,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(evaluation.router)
api_router.include_router(users.router)
api_router.include_router(audit.router)
