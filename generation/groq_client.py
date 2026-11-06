from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from math import ceil
from typing import TypeVar

from groq import Groq, RateLimitError


T = TypeVar("T")
MAX_RATE_LIMIT_RETRIES = 2
MAX_AUTOMATIC_WAIT_SECONDS = 600.0


class GroqRateLimitError(RuntimeError):
    """A Groq rate limit could not be resolved with bounded retries."""

    def __init__(
        self,
        message: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class GroqStructuredOutputError(RuntimeError):
    """Groq could not complete a response constrained to valid JSON."""


def _retry_after_seconds(error: RateLimitError) -> float | None:
    headers = error.response.headers
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            return max(0.0, float(retry_after_ms) / 1000)
        except ValueError:
            pass

    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    body = error.body
    if isinstance(body, Mapping):
        error_body = body.get("error")
        message = (
            error_body.get("message", "")
            if isinstance(error_body, Mapping)
            else ""
        )
    else:
        message = str(body or error)

    match = re.search(
        r"try again in\s+(\d+(?:\.\d+)?)\s*"
        r"(milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|m)\b",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"m", "min", "mins", "minute", "minutes"}:
        return amount * 60
    if unit in {"ms", "millisecond", "milliseconds"}:
        return amount / 1000
    return amount


def call_with_rate_limit_retry(operation: Callable[[], T]) -> T:
    """Retry Groq rate limits up to twice when the suggested wait is <= 600s."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return operation()
        except RateLimitError as error:
            retry_after = _retry_after_seconds(error)
            if retry_after is None:
                retry_after = 0.5 * (2**attempt)

            if retry_after > MAX_AUTOMATIC_WAIT_SECONDS:
                raise GroqRateLimitError(
                    f"Groq is rate-limiting requests. Wait about "
                    f"{ceil(retry_after)} seconds before trying again. "
                    "Answer generation and response evaluation are mandatory. "
                    "For a smaller run, reduce the number of questions, selected "
                    "chunking strategies, embedding models, or retrieval methods.",
                    retry_after_seconds=retry_after,
                ) from error

            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise GroqRateLimitError(
                    "Groq is still rate-limiting requests after automatic "
                    "retries. Answer generation and response evaluation are "
                    "mandatory for every pipeline. Wait for Groq's rate limit "
                    "window to reset, then rerun with fewer questions or "
                    "selected chunking strategies, embedding models, and "
                    "retrieval methods. HyDE and Query Expansion add another "
                    "Groq request per question.",
                    retry_after_seconds=retry_after,
                ) from error

            time.sleep(retry_after)

    raise AssertionError("Rate-limit retry loop exited unexpectedly.")


def create_chat_completion(api_key: str, **request: object):
    """Make one chat completion with bounded rate-limit retries."""
    client = Groq(api_key=api_key, max_retries=0)
    try:
        return call_with_rate_limit_retry(
            lambda: client.chat.completions.create(**request)
        )
    finally:
        client.close()
