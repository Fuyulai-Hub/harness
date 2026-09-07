"""OpenAI-compatible HTTP client with retry, backoff and concurrency control.

This is the default and (out of the box) only :class:`Model` implementation.  It speaks
both the Chat Completions and Completions flavours of the OpenAI HTTP API and therefore
works against:

* the real OpenAI API,
* local inference servers (vLLM, Ollama, TGI, LM Studio, llama.cpp server),
* any reverse proxy that mimics the schema.

Design notes
------------
* The *synchronous* path uses :mod:`requests` because it is universally available on
  Windows and trivially debuggable.  It is the default executor.
* The *asynchronous* path uses :mod:`aiohttp` and is only imported when the runner
  selects ``executor: async``.  This keeps the dependency footprint minimal.
* Retry uses exponential backoff with jitter and honours ``Retry-After`` when the server
  returns it.  A bounded :class:`~asyncio.Semaphore` (per model) caps concurrency so the
  harness never overwhelms the endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, Iterator, List, Optional

from ..config import ModelConfig
from .base import Model, ModelResponse

log = logging.getLogger(__name__)

try:
    import requests  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'requests' package is required for HTTP model calls. "
        "Install it with `pip install requests`."
    ) from exc


class OpenAICompatibleModel(Model):
    """Talk to an OpenAI-style completions/chat endpoint.

    Parameters are taken from a :class:`~fylharness.config.ModelConfig`; the resolved
    API key is read lazily so that environment changes are picked up.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._semaphore: Optional[asyncio.Semaphore] = None

    # --------------------------------------------------------------- internals
    @property
    def _api_key(self) -> str:
        key = self.config.resolved_api_key()
        if not key:
            # Many local servers accept any non-empty string; warn but continue.
            log.warning(
                "Model '%s' has no API key (env %s unset). Using 'EMPTY' -- set a key "
                "for hosted endpoints.",
                self.config.name,
                self.config.api_key_env,
            )
            return "EMPTY"
        return key

    @property
    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return f"{base}/chat/completions" if self.config.chat else f"{base}/completions"

    def _build_body(self, payload: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ov = overrides or {}
        body: Dict[str, Any] = {
            "model": self.config.model or self.config.name,
            "temperature": ov.get("temperature", self.config.temperature),
            "max_tokens": ov.get("max_tokens", self.config.max_tokens),
        }
        if self.config.chat:
            messages = payload.get("messages")
            if messages is None:
                # Allow raw ``prompt`` to be wrapped as a single user turn.
                messages = [{"role": "user", "content": payload.get("prompt", "")}]
            body["messages"] = messages
        else:
            body["prompt"] = payload.get("prompt", "")
        body.update(self.config.extra)
        # Apply per-request reasoning_effort override (DeepSeek-R1 / o1 style).
        if ov.get("reasoning_effort"):
            body["reasoning_effort"] = ov["reasoning_effort"]
        elif "reasoning_effort" in self.config.extra:
            body["reasoning_effort"] = self.config.extra["reasoning_effort"]
        return body

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _extract_text(self, data: Dict[str, Any]) -> str:
        if self.config.chat:
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "") or ""
            return choices[0].get("text", "") if choices else ""
        choices = data.get("choices", [])
        return choices[0].get("text", "") if choices else ""

    # ----------------------------------------------------------------- retrying
    def _should_retry(self, status: int) -> bool:
        # Retry on rate-limit and transient server errors.
        return status in (408, 409, 429, 500, 502, 503, 504)

    def _backoff(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        base = self.config.retry_backoff
        # Exponential with full jitter (0 .. base * 2**attempt), capped at 60s.
        return min(random.uniform(0, base * (2 ** attempt)), 60.0)

    # ----------------------------------------------------- synchronous request
    def generate(self, payload: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> ModelResponse:
        url = self._endpoint
        headers = self._headers()
        body = self._build_body(payload, overrides=overrides)
        last_error: Optional[str] = None
        for attempt in range(self.config.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.config.timeout,
                )
            except requests.RequestException as exc:
                last_error = f"network: {exc!s}"
                log.warning("Model %s request error (attempt %d): %s",
                            self.config.name, attempt + 1, exc)
            else:
                latency = (time.perf_counter() - start) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage", {}) or {}
                    return ModelResponse(
                        text=self._extract_text(data),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        finish_reason=(
                            data.get("choices", [{}])[0].get("finish_reason", "")
                            if data.get("choices")
                            else ""
                        ),
                        latency_ms=latency,
                        raw=data,
                    )
                last_error = f"http {resp.status_code}: {resp.text[:200]}"
                if not self._should_retry(resp.status_code) or attempt == self.config.max_retries:
                    log.error("Model %s non-retryable/last error: %s",
                              self.config.name, last_error)
                    break
                retry_after = resp.headers.get("Retry-After")
                sleep = self._backoff(attempt, retry_after)
                log.warning("Model %s HTTP %d, retrying in %.1fs (attempt %d/%d)",
                            self.config.name, resp.status_code, sleep,
                            attempt + 1, self.config.max_retries)
                time.sleep(sleep)
                continue
            if attempt == self.config.max_retries:
                break
            sleep = self._backoff(attempt, None)
            log.warning("Model %s retrying in %.1fs (attempt %d/%d)",
                        self.config.name, sleep, attempt + 1, self.config.max_retries)
            time.sleep(sleep)
        return ModelResponse(error=last_error or "unknown error")

    # -------------------------------------------------------- asynchronous path
    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.config.concurrency))
        return self._semaphore

    async def agenerate(self, payload: Dict[str, Any]) -> ModelResponse:
        try:
            import aiohttp  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'aiohttp' package is required for async execution. "
                "Install it with `pip install aiohttp` or set executor: thread."
            ) from exc

        sem = self._get_semaphore()
        async with sem:
            return await self._agenerate_once(aiohttp, payload)

    async def _agenerate_once(self, aiohttp, payload: Dict[str, Any]) -> ModelResponse:
        url = self._endpoint
        headers = self._headers()
        body = self._build_body(payload)
        last_error: Optional[str] = None
        for attempt in range(self.config.max_retries + 1):
            start = time.perf_counter()
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as session:
                    async with session.post(url, headers=headers, json=body) as resp:
                        latency = (time.perf_counter() - start) * 1000
                        if resp.status == 200:
                            data = await resp.json()
                            usage = data.get("usage", {}) or {}
                            return ModelResponse(
                                text=self._extract_text(data),
                                prompt_tokens=usage.get("prompt_tokens", 0),
                                completion_tokens=usage.get("completion_tokens", 0),
                                finish_reason=(
                                    data.get("choices", [{}])[0].get("finish_reason", "")
                                    if data.get("choices")
                                    else ""
                                ),
                                latency_ms=latency,
                                raw=data,
                            )
                        text = await resp.text()
                        last_error = f"http {resp.status}: {text[:200]}"
                        if not self._should_retry(resp.status) or attempt == self.config.max_retries:
                            break
                        retry_after = resp.headers.get("Retry-After")
                        sleep = self._backoff(attempt, retry_after)
                        log.warning("Model %s async HTTP %d, retry in %.1fs (%d/%d)",
                                    self.config.name, resp.status, sleep,
                                    attempt + 1, self.config.max_retries)
                        await asyncio.sleep(sleep)
                        continue
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = f"network: {exc!s}"
                log.warning("Model %s async error (attempt %d): %s",
                            self.config.name, attempt + 1, exc)
                if attempt == self.config.max_retries:
                    break
                sleep = self._backoff(attempt, None)
                await asyncio.sleep(sleep)
        return ModelResponse(error=last_error or "unknown error")

    # --------------------------------------------------------- chat interface
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> ModelResponse:
        """Conversational completion over a message history.

        Accepts per-request overrides: ``temperature``, ``max_tokens``,
        ``reasoning_effort``.
        """

        return self.generate(
            {"messages": messages, "prompt": messages[-1].get("content", "") if messages else ""},
            overrides=kwargs,
        )

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs: Any) -> Iterator[str]:
        """Stream a chat completion token-by-token via the OpenAI SSE protocol.

        Accepts per-request overrides: ``temperature``, ``max_tokens``,
        ``reasoning_effort``.
        """

        url = self._endpoint
        headers = self._headers()
        body = self._build_body({"messages": messages}, overrides=kwargs)
        body["stream"] = True
        try:
            with requests.post(
                url, headers=headers, json=body, timeout=self.config.timeout, stream=True
            ) as resp:
                if resp.status_code != 200:
                    err = resp.text[:300]
                    yield f"[ERROR] http {resp.status_code}: {err}"
                    return
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield text
        except requests.RequestException as exc:
            yield f"[ERROR] network: {exc!s}"
