from __future__ import annotations

import os
from pathlib import Path
from loguru import logger
from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import PydanticBaseSettingsSource
from pydantic_settings import YamlConfigSettingsSource
from cores.services.web_discovery.settings import WebDiscoverySettings
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider


# test in local
load_dotenv(find_dotenv('.env'), override=True)

# Get the directory where this settings.py file is located
SETTINGS_DIR = Path(__file__).parent
YAML_FILE_PATH = SETTINGS_DIR / 'settings.yaml'

class GoogleThinkingConfig(BaseModel):
    thinking_budget: int = 2048
    include_thoughts: bool = True


class ModelConfig(BaseModel):
    lead_agent_model_name: str = "gemini-2.5-flash"
    subagent_model_name: str = "gemini-2.5-flash"
class GeminiSettings(BaseModel):
    api_key: str = ""
    flash_rpm: int = 15
    pro_rpm: int = 10

class Settings(BaseSettings):

    web_discovery: WebDiscoverySettings
    model: ModelConfig
    gemini: GeminiSettings 

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
        logger.debug(f"Configuring lead research model: {self.gemini.api_key}")
        provider = GoogleProvider(api_key=self.gemini.api_key)
        logger.info(f"Using provider: {provider}")
        return GoogleModel(
            self.model.lead_agent_model_name,
            provider=provider
        )

    @property
    def subagent_research_model(self) -> Model:
        """Subagent research model with lower thinking budget."""
        provider = GoogleProvider(api_key=self.gemini.api_key)
        logger.info(f"Using provider: {provider}")
        return GoogleModel(
            self.model.subagent_model_name,
            provider=provider
        )

settings = Settings()
print("Load API Key success")
print(settings.gemini.api_key)
