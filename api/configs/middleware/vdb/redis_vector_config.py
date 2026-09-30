from pydantic import Field
from pydantic_settings import BaseSettings


class RedisVectorConfig(BaseSettings):
    """
    Configuration settings for Redis as a vector store (requires Redis search, e.g. Redis 8)
    """

    REDIS_VECTOR_URL: str | None = Field(
        description="Redis URL for the vector store, e.g. 'redis://:password@localhost:6379/0' ('rediss://' for TLS)",
        default=None,
    )
