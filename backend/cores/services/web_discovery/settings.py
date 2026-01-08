from __future__ import annotations
from base import BaseModel

class WebDiscoverySettings(BaseModel):
    brave_search_url: str = "https://api.search.brave.com/res/v1/web/search"
    brave_api_key: str