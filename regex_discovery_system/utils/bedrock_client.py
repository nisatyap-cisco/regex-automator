from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import boto3
import yaml

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Best-effort load of .env file next to config.yaml."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            logger.debug("Loaded .env from %s", env_path)
    except ImportError:
        pass


def load_config(path: str = "config.yaml") -> dict:
    _load_dotenv()

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    profile = os.getenv("AWS_PROFILE", cfg.get("aws_profile", ""))
    region = os.getenv("AWS_REGION", cfg.get("region", "us-east-1"))
    tavily_key = os.getenv("TAVILY_API_KEY", cfg.get("tavily_api_key", ""))
    proxies_env = os.getenv("PROXIES", "")
    proxies = [p.strip() for p in proxies_env.split(",") if p.strip()] if proxies_env else cfg.get("proxies", [])

    cfg["aws_profile"] = profile
    cfg["region"] = region
    cfg["tavily_api_key"] = tavily_key
    cfg["proxies"] = proxies

    return cfg


def _build_session(config: dict) -> boto3.Session:
    """Build a boto3 session: use named profile for local dev, instance role for deployed.

    boto3 reads AWS_PROFILE from the environment even when profile_name is not
    passed explicitly.  If the .env sets AWS_PROFILE="" (empty), boto3 tries to
    find a profile named "" and raises "The config profile () could not be found".
    We therefore pop the env var when no profile is configured so that boto3 falls
    through to the instance-role / default credential chain.
    """
    profile = config.get("aws_profile") or ""
    region = config.get("region", "us-east-1")

    if profile:
        return boto3.Session(profile_name=profile, region_name=region)

    # No profile configured — use instance role / default credential chain.
    # Temporarily remove AWS_PROFILE from the environment so boto3 doesn't
    # try to resolve the empty-string profile name.
    _env_profile = os.environ.pop("AWS_PROFILE", None)
    try:
        return boto3.Session(region_name=region)
    finally:
        if _env_profile:          # only restore if it was a real non-empty value
            os.environ["AWS_PROFILE"] = _env_profile


def invoke_claude(
    prompt: str,
    config: dict,
    system: str = "",
    max_retries: Optional[int] = None,
) -> str:
    retries = max_retries if max_retries is not None else config.get("max_retries", 3)
    session = _build_session(config)
    client = session.client("bedrock-runtime", region_name=config["region"])

    messages = [{"role": "user", "content": prompt}]
    body: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": config.get("max_tokens", 4096),
        "temperature": config.get("temperature", 0.2),
    }
    if system:
        body["system"] = system

    for attempt in range(1, retries + 1):
        try:
            resp = client.invoke_model(
                modelId=config["model_id"],
                body=json.dumps(body),
            )
            result = json.loads(resp["body"].read())
            return result["content"][0]["text"]
        except Exception as exc:
            wait = 2 ** (attempt - 1)
            logger.warning(
                "Bedrock call failed (attempt %d/%d): %s — retrying in %ds",
                attempt,
                retries,
                exc,
                wait,
            )
            if attempt == retries:
                raise
            time.sleep(wait)

    raise RuntimeError("Exhausted retries for Bedrock invocation")
