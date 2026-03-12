from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import boto3
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    cfg = {}
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("aws_profile"):
        os.environ["AWS_PROFILE"] = cfg["aws_profile"]
    if cfg.get("region"):
        os.environ["AWS_REGION"] = cfg["region"]

    return cfg


def invoke_claude(
    prompt: str,
    config: dict,
    system: str = "",
    max_retries: Optional[int] = None,
) -> str:
    retries = max_retries if max_retries is not None else config.get("max_retries", 3)
    session = boto3.Session(profile_name=config.get("aws_profile", "strln"))
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
