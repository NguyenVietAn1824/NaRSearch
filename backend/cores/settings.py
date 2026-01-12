from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from loguru import logger
from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import PydanticBaseSettingsSource
from pydantic_settings import YamlConfigSettingsSource
from cores.services.web_discovery.settings import WebDiscoverySettings
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider


# test in local
load_dotenv(find_dotenv('.env'), override=True)

# Get the directory where this settings.py file is located
SETTINGS_DIR = Path(__file__).parent
YAML_FILE_PATH = SETTINGS_DIR / 'settings.yaml'


class GoogleThinkingConfig(BaseModel):
    """Google thinking configuration."""
    thinking_budget: int = 2048
    include_thoughts: bool = True


class ModelConfig(BaseModel):
    """Model configuration."""
    lead_agent_model_name: str = "gemini-2.0-flash"
    subagent_model_name: str = "gemini-2.0-flash"
    lead_thinking_budget: int = 8096
    subagent_thinking_budget: int = 2048
    include_thoughts: bool = False


class GeminiSettings(BaseModel):
    """Google Gemini API settings."""
    api_key: str = ""
    flash_rpm: int = 15
    pro_rpm: int = 10
    
    @field_validator('api_key', mode='before')
    @classmethod
    def get_api_key_from_env(cls, v: str) -> str:
        """Get API key from GOOGLE_API_KEY or GEMINI__API_KEY environment variable."""
        if v:
            return v
        # Try GOOGLE_API_KEY first (standard for Generative Language API)
        google_key = os.getenv('GOOGLE_API_KEY')
        if google_key:
            return google_key
        # Fall back to GEMINI__API_KEY (nested format)
        gemini_key = os.getenv('GEMINI__API_KEY')
        if gemini_key:
            return gemini_key
        return ""


class Settings(BaseSettings):

    web_discovery: WebDiscoverySettings
    model: ModelConfig = ModelConfig()
    gemini: GeminiSettings = GeminiSettings()

    class Config:
        env_nested_delimiter = '__'
        yaml_file = str(YAML_FILE_PATH)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @property
    def lead_research_model(self) -> Model:
        """Lead research agent model with higher thinking budget."""
        provider = GoogleProvider(api_key=self.gemini.api_key)
        logger.info(f"Using provider: {provider}")
        return GoogleModel(
            model_name=self.model.lead_agent_model_name,
        )

    @property
    def subagent_research_model(self) -> Model:
        """Subagent research model with lower thinking budget."""
        provider = GoogleProvider(api_key=self.gemini.api_key)
        logger.info(f"Using provider: {provider}")
        return GoogleModel(
            model_name=self.model.subagent_model_name,
        )


settings = Settings()