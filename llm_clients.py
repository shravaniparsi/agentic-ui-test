"""
LLM client supporting three backends:
  1. Direct OpenAI API (set OPENAI_API_KEY in .env)
  2. Direct Anthropic API (set ANTHROPIC_API_KEY in .env)
  3. WLM LLM Gateway (fallback — set REDACTED_GATEWAY_KEY in .env)

All backends share the same interface so verify.py doesn't need to change.
The correct client is selected based on the model's "provider" field in config.
"""
import base64
import json
import time
from pathlib import Path
from typing import Optional, Union

import requests
import urllib3

from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GATEWAY_OPENAI_URL,
    GATEWAY_ANTHROPIC_URL,
    GATEWAY_GEMINI_URL,
    GATEWAY_KEY,
    GATEWAY_HEADERS,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _encode_image(image_path: Union[str, Path]) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _detect_media_type(image_path: Union[str, Path]) -> str:
    suffix = Path(image_path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")


def _parse_response(raw_text: str) -> dict:
    """Extract JSON from model response.

    Handles: plain JSON, markdown-fenced JSON, and JSON embedded in
    free-form reasoning (common with Claude).
    """
    import re

    text = raw_text.strip()

    # 1. Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # 2. Try parsing the whole text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Find the last JSON object containing "verdict" (models sometimes
    #    write reasoning before the JSON block).  Use DOTALL so the match
    #    spans newlines inside the reasoning field.
    matches = list(re.finditer(r'\{[^{}]*"verdict"[^{}]*\}', text, re.DOTALL))
    if matches:
        try:
            return json.loads(matches[-1].group())
        except json.JSONDecodeError:
            pass

    # 4. Greedy fallback: find last '{' before "verdict" and last '}' in text
    verdict_pos = text.rfind('"verdict"')
    if verdict_pos != -1:
        brace_start = text.rfind('{', 0, verdict_pos)
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

    # 5. Try extracting from markdown fenced block anywhere in text
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    return {
        "verdict": "PARSE_ERROR",
        "confidence": 0,
        "reasoning": f"Failed to parse: {raw_text[:500]}",
    }


def _build_multimodal_messages(
    system_prompt: str,
    user_prompt: str,
    actual_screenshot: Union[str, Path],
    reference_screenshot: Optional[Union[str, Path]] = None,
) -> list[dict]:
    """Build OpenAI-format multimodal messages (shared by both backends)."""
    user_content = []

    if reference_screenshot:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{_encode_image(reference_screenshot)}",
                "detail": "high",
            },
        })

    user_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{_encode_image(actual_screenshot)}",
            "detail": "high",
        },
    })

    user_content.append({"type": "text", "text": user_prompt})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


class OpenAIDirectClient:
    """Client that calls the OpenAI API directly using the openai SDK."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def verify(
        self,
        model_id: str,
        model_version: str,
        api_version: str,
        system_prompt: str,
        user_prompt: str,
        actual_screenshot: Union[str, Path],
        reference_screenshot: Optional[Union[str, Path]] = None,
        supports_temperature: bool = True,
    ) -> dict:
        messages = _build_multimodal_messages(
            system_prompt, user_prompt, actual_screenshot, reference_screenshot
        )

        kwargs = {
            "model": model_id,
            "messages": messages,
            "max_completion_tokens": 512,
        }
        if supports_temperature:
            kwargs["temperature"] = 0

        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                response = self.client.chat.completions.create(**kwargs)
                latency = time.time() - t0

                raw = response.choices[0].message.content or ""
                parsed = _parse_response(raw)
                parsed["_raw"] = raw
                parsed["_latency_s"] = round(latency, 2)
                parsed["_input_tokens"] = response.usage.prompt_tokens if response.usage else 0
                parsed["_output_tokens"] = response.usage.completion_tokens if response.usage else 0
                return parsed

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {
                        "verdict": "API_ERROR",
                        "confidence": 0,
                        "reasoning": str(e),
                        "_raw": "",
                        "_latency_s": 0,
                        "_input_tokens": 0,
                        "_output_tokens": 0,
                    }
                time.sleep(2 ** attempt)

    def generate_text(
        self,
        model_id: str,
        model_version: str,
        api_version: str,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 200,
        supports_temperature: bool = True,
    ) -> str:
        kwargs = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        }
        if supports_temperature:
            kwargs["temperature"] = temperature

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)


class AnthropicDirectClient:
    """Client that calls the Anthropic API directly using the anthropic SDK."""

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def _build_anthropic_content(
        self,
        user_prompt: str,
        actual_screenshot: Union[str, Path],
        reference_screenshot: Optional[Union[str, Path]] = None,
    ) -> list[dict]:
        """Build Anthropic-format content blocks (images use source.type=base64)."""
        content = []
        if reference_screenshot:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _detect_media_type(reference_screenshot),
                    "data": _encode_image(reference_screenshot),
                },
            })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": _detect_media_type(actual_screenshot),
                "data": _encode_image(actual_screenshot),
            },
        })
        content.append({"type": "text", "text": user_prompt})
        return content

    def verify(
        self,
        model_id: str,
        model_version: str,
        api_version: str,
        system_prompt: str,
        user_prompt: str,
        actual_screenshot: Union[str, Path],
        reference_screenshot: Optional[Union[str, Path]] = None,
        supports_temperature: bool = True,
    ) -> dict:
        content = self._build_anthropic_content(
            user_prompt, actual_screenshot, reference_screenshot
        )

        kwargs = {
            "model": model_id,
            "max_tokens": 512,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }
        if supports_temperature:
            kwargs["temperature"] = 0

        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                response = self.client.messages.create(**kwargs)
                latency = time.time() - t0

                raw = response.content[0].text if response.content else ""
                parsed = _parse_response(raw)
                parsed["_raw"] = raw
                parsed["_latency_s"] = round(latency, 2)
                parsed["_input_tokens"] = response.usage.input_tokens if response.usage else 0
                parsed["_output_tokens"] = response.usage.output_tokens if response.usage else 0
                return parsed

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {
                        "verdict": "API_ERROR",
                        "confidence": 0,
                        "reasoning": str(e),
                        "_raw": "",
                        "_latency_s": 0,
                        "_input_tokens": 0,
                        "_output_tokens": 0,
                    }
                time.sleep(2 ** attempt)

    def generate_text(
        self,
        model_id: str,
        model_version: str,
        api_version: str,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 200,
        supports_temperature: bool = True,
    ) -> str:
        kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if supports_temperature:
            kwargs["temperature"] = temperature

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.messages.create(**kwargs)
                return (response.content[0].text if response.content else "").strip()
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)


class GatewayClient:
    """Client that calls OpenAI models through the WLM LLM Gateway."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GATEWAY_KEY}",
            **GATEWAY_HEADERS,
        })
        self.session.verify = False

    def verify(
        self,
        model_id: str,
        model_version: str,
        api_version: str,
        system_prompt: str,
        user_prompt: str,
        actual_screenshot: Union[str, Path],
        reference_screenshot: Optional[Union[str, Path]] = None,
        supports_temperature: bool = True,
    ) -> dict:
        messages = _build_multimodal_messages(
            system_prompt, user_prompt, actual_screenshot, reference_screenshot
        )

        model_params = {
            "max_completion_tokens": 512,
            "messages": messages,
        }
        if supports_temperature:
            model_params["temperature"] = 0

        payload = {
            "model": model_id,
            "model-version": model_version,
            "api-version": api_version,
            "task": "chat/completions",
            "streaming": False,
            "model-params": model_params,
        }

        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                resp = self.session.post(
                    GATEWAY_OPENAI_URL,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                latency = time.time() - t0
                resp.raise_for_status()
                data = resp.json()

                raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                parsed = _parse_response(raw)
                parsed["_raw"] = raw
                parsed["_latency_s"] = round(latency, 2)

                usage = data.get("usage") or data.get("model_usage") or {}
                parsed["_input_tokens"] = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                parsed["_output_tokens"] = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                return parsed

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {
                        "verdict": "API_ERROR",
                        "confidence": 0,
                        "reasoning": str(e),
                        "_raw": "",
                        "_latency_s": 0,
                        "_input_tokens": 0,
                        "_output_tokens": 0,
                    }
                time.sleep(2 ** attempt)

    def generate_text(
        self,
        model_id: str,
        model_version: str,
        api_version: str,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 200,
        supports_temperature: bool = True,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]

        model_params = {
            "max_completion_tokens": max_tokens,
            "messages": messages,
        }
        if supports_temperature:
            model_params["temperature"] = temperature

        payload = {
            "model": model_id,
            "model-version": model_version,
            "api-version": api_version,
            "task": "chat/completions",
            "streaming": False,
            "model-params": model_params,
        }

        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.post(
                    GATEWAY_OPENAI_URL,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)


class GatewayAnthropicClient:
    """Client that calls Anthropic models through the WLM LLM Gateway's /v1/messages endpoint."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-api-key": GATEWAY_KEY,
            **GATEWAY_HEADERS,
        })
        self.session.verify = False

    def _build_anthropic_content(
        self,
        user_prompt: str,
        actual_screenshot: Union[str, Path],
        reference_screenshot: Optional[Union[str, Path]] = None,
    ) -> list[dict]:
        content = []
        if reference_screenshot:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _detect_media_type(reference_screenshot),
                    "data": _encode_image(reference_screenshot),
                },
            })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": _detect_media_type(actual_screenshot),
                "data": _encode_image(actual_screenshot),
            },
        })
        content.append({"type": "text", "text": user_prompt})
        return content

    def verify(
        self,
        model_id: str,
        model_version: str,
        api_version: str,
        system_prompt: str,
        user_prompt: str,
        actual_screenshot: Union[str, Path],
        reference_screenshot: Optional[Union[str, Path]] = None,
        supports_temperature: bool = True,
    ) -> dict:
        content = self._build_anthropic_content(
            user_prompt, actual_screenshot, reference_screenshot
        )

        payload = {
            "model": model_id,
            "max_tokens": 512,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }
        if supports_temperature:
            payload["temperature"] = 0

        headers = {"anthropic-version": api_version}

        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                resp = self.session.post(
                    GATEWAY_ANTHROPIC_URL,
                    json=payload,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
                latency = time.time() - t0
                resp.raise_for_status()
                data = resp.json()

                raw = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        raw = block.get("text", "")
                        break

                parsed = _parse_response(raw)
                parsed["_raw"] = raw
                parsed["_latency_s"] = round(latency, 2)

                usage = data.get("usage", {})
                parsed["_input_tokens"] = usage.get("input_tokens", 0)
                parsed["_output_tokens"] = usage.get("output_tokens", 0)
                return parsed

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {
                        "verdict": "API_ERROR",
                        "confidence": 0,
                        "reasoning": str(e),
                        "_raw": "",
                        "_latency_s": 0,
                        "_input_tokens": 0,
                        "_output_tokens": 0,
                    }
                time.sleep(2 ** attempt)

    def generate_text(
        self,
        model_id: str,
        model_version: str,
        api_version: str,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 200,
        supports_temperature: bool = True,
    ) -> str:
        payload = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if supports_temperature:
            payload["temperature"] = temperature

        headers = {"anthropic-version": api_version}

        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.post(
                    GATEWAY_ANTHROPIC_URL,
                    json=payload,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        return block.get("text", "").strip()
                return ""
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)


class GatewayGeminiClient:
    """Client that calls Gemini models through the WLM LLM Gateway's Vertex-style endpoint."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-goog-api-key": GATEWAY_KEY,
            **GATEWAY_HEADERS,
        })
        self.session.verify = False

    def _build_gemini_parts(
        self,
        user_prompt: str,
        actual_screenshot: Union[str, Path],
        reference_screenshot: Optional[Union[str, Path]] = None,
    ) -> list[dict]:
        """Build Gemini-format parts list (inline_data for images, text for prompt)."""
        parts = []
        if reference_screenshot:
            parts.append({
                "inline_data": {
                    "mime_type": _detect_media_type(reference_screenshot),
                    "data": _encode_image(reference_screenshot),
                },
            })
        parts.append({
            "inline_data": {
                "mime_type": _detect_media_type(actual_screenshot),
                "data": _encode_image(actual_screenshot),
            },
        })
        parts.append({"text": user_prompt})
        return parts

    def verify(
        self,
        model_id: str,
        model_version: str,
        api_version: str,
        system_prompt: str,
        user_prompt: str,
        actual_screenshot: Union[str, Path],
        reference_screenshot: Optional[Union[str, Path]] = None,
        supports_temperature: bool = True,
    ) -> dict:
        parts = self._build_gemini_parts(
            user_prompt, actual_screenshot, reference_screenshot
        )

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generation_config": {
                "max_output_tokens": 512,
            },
        }
        if supports_temperature:
            payload["generation_config"]["temperature"] = 0.01

        url = f"{GATEWAY_GEMINI_URL}/{model_id}:generateContent"

        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                resp = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
                latency = time.time() - t0
                resp.raise_for_status()
                data = resp.json()

                raw = ""
                for candidate in data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "text" in part:
                            raw += part["text"]

                parsed = _parse_response(raw)
                parsed["_raw"] = raw
                parsed["_latency_s"] = round(latency, 2)

                usage = data.get("usageMetadata", {})
                parsed["_input_tokens"] = usage.get("promptTokenCount", 0)
                parsed["_output_tokens"] = usage.get("candidatesTokenCount", 0)
                return parsed

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {
                        "verdict": "API_ERROR",
                        "confidence": 0,
                        "reasoning": str(e),
                        "_raw": "",
                        "_latency_s": 0,
                        "_input_tokens": 0,
                        "_output_tokens": 0,
                    }
                time.sleep(2 ** attempt)


def get_client(provider: str = "openai"):
    """Return the appropriate client based on the model's provider field.

    Priority for OpenAI models: direct API key > gateway.
    Priority for Anthropic models: direct API key > gateway (Anthropic endpoint).
    Gateway provider: always uses the WLM gateway (OpenAI-format endpoint).
    """
    if provider == "gateway":
        print("[LLM Client] Using WLM LLM Gateway (OpenAI-format)")
        return GatewayClient()

    if provider == "gemini":
        print("[LLM Client] Using WLM LLM Gateway (Gemini Vertex-format)")
        return GatewayGeminiClient()

    if provider == "anthropic":
        if ANTHROPIC_API_KEY:
            print("[LLM Client] Using Anthropic API directly")
            return AnthropicDirectClient()
        print("[LLM Client] Using WLM LLM Gateway (Anthropic /v1/messages)")
        return GatewayAnthropicClient()

    # Default: openai provider
    if OPENAI_API_KEY:
        print("[LLM Client] Using OpenAI API directly")
        return OpenAIDirectClient()
    print("[LLM Client] Using WLM LLM Gateway (OpenAI)")
    return GatewayClient()
