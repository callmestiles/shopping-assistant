import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SHOPIFY_SHOP_URL: str = Field(..., description="The URL of the Shopify shop")
    SHOPIFY_ACCESS_TOKEN: str = Field(..., description="Access token for Shopify API")
    SHOPIFY_API_VERSION: str = Field(default="2025-07", description="API version for Shopify")
    OPENAI_KEY: Optional[str] = Field(default=None, description="OpenAI API key for LLM features")


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()

def get_shopify_headers() -> dict:
    """Generate headers for Shopify API requests."""
    return {
        "X-Shopify-Access-Token": settings.SHOPIFY_ACCESS_TOKEN,
        "Content-Type": "application/json",
    }

def get_shopify_graphql_url() -> str:
    """Get the Shopify GraphQL API URL."""
    return f"https://{settings.SHOPIFY_SHOP_URL}/admin/api/{settings.SHOPIFY_API_VERSION}/graphql.json"

