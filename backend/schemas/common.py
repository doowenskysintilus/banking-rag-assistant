from pydantic import BaseModel, ConfigDict


class ORMModel(BaseModel):
    """Base schema for models that are read directly from SQLAlchemy ORM objects."""

    model_config = ConfigDict(from_attributes=True)


class HealthStatus(BaseModel):
    status: str
    version: str
    environment: str


class ReadinessCheck(BaseModel):
    name: str
    healthy: bool
    detail: str | None = None


class ReadinessStatus(BaseModel):
    status: str
    checks: list[ReadinessCheck]
