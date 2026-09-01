import time
import base64
import logging

import requests
from langchain_openai import ChatOpenAI

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Module-level token cache
_token_cache = {"access_token": None, "expires_at": 0.0}
_REFRESH_MARGIN_SECONDS = 60


def _get_access_token() -> str:
    """Fetch (and cache) an OAuth2 client-credentials access token from Gloo AI Studio."""
    now = time.time()
    if _token_cache["access_token"] and now < (_token_cache["expires_at"] - _REFRESH_MARGIN_SECONDS):
        return _token_cache["access_token"]

    settings = get_settings()
    basic = base64.b64encode(
        f"{settings.gloo_client_id}:{settings.gloo_client_secret}".encode("utf-8")
    ).decode("utf-8")

    resp = requests.post(
        settings.gloo_token_url,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials", "scope": settings.gloo_scope},
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Gloo OAuth token request failed [{resp.status_code}]: {resp.text}"
        )

    payload = resp.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("Gloo OAuth response did not contain an access_token")

    expires_in = payload.get("expires_in", 3600)
    _token_cache["access_token"] = token
    _token_cache["expires_at"] = time.time() + float(expires_in)

    logger.info("Fetched new Gloo access token (expires_in=%ss)", expires_in)
    return token


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance pointed at the Gloo AI Studio OpenAI-compatible API.

    A fresh instance is built per call so the current (auto-refreshed) OAuth
    token is always used. Token fetching itself is cached in-process.
    """
    settings = get_settings()
    token = _get_access_token()

    logger.info(
        "Initializing Gloo ChatOpenAI | model: %s | base: %s",
        settings.gloo_model,
        settings.gloo_api_base,
    )

    return ChatOpenAI(
        base_url=settings.gloo_api_base,
        api_key=token,
        model=settings.gloo_model,
        temperature=temperature,
        max_tokens=1500,
        streaming=False,
    )
