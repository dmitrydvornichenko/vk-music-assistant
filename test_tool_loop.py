#!/usr/bin/env python3
"""
Manual tool-calling loop test.

Supports local llama-server and cloud LLMs via environment variables:

  LLM_PROVIDER  local | openai | anthropic | openrouter | ollama | custom
                default: local
  LLM_API_KEY   API key for cloud providers (not needed for local/ollama)
  LLM_URL       Override base URL (optional; sensible defaults are set per provider)
  LLM_MODEL     Model name
  MCPO_URL      mcpo endpoint (default: http://localhost:8001)

Examples:
  # local llama-server (default)
  python test_tool_loop.py "play born to be wild"

  # OpenAI
  LLM_PROVIDER=openai LLM_API_KEY=sk-... LLM_MODEL=gpt-4o python test_tool_loop.py "play born to be wild"

  # Anthropic Claude
  LLM_PROVIDER=anthropic LLM_API_KEY=sk-ant-... LLM_MODEL=claude-opus-4-5 python test_tool_loop.py "play born to be wild"

  # OpenRouter
  LLM_PROVIDER=openrouter LLM_API_KEY=sk-or-... LLM_MODEL=openai/gpt-4o python test_tool_loop.py "play born to be wild"

  # Ollama (local, no key)
  LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:32b python test_tool_loop.py "play born to be wild"
"""

import json
import os
import sys
import requests

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "local").lower()
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_URL      = os.environ.get("LLM_URL", "").rstrip("/")
MODEL        = os.environ.get("LLM_MODEL", "gpt-oss-20b-Q4_K_M.gguf")
MCPO_URL     = os.environ.get("MCPO_URL", "http://localhost:8001")
MAX_ITERS    = 6

# Default base URLs for OpenAI-compatible providers (can be overridden via LLM_URL)
_DEFAULT_BASES: dict = {
    "openai":     "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama":     "http://localhost:11434/v1",
    "local":      "http://localhost:8000/v1",
}

SYSTEM_PROMPT = """\
When the user asks to play music — do NOT ask for confirmation and do NOT show \
intermediate results. Always do the following in one go:
1. Call search_tracks first.
2. Immediately call play_music with the best matching result.
3. Only then tell the user what you started playing.

Always pass access_key to play_music if search_tracks returned it.
Pick the track that best matches: correct artist + correct album (if specified). \
Prefer tracks where main_artists is present and matches the requested artist.

CRITICAL: search_tracks does NOT play music. Music is NOT playing until you \
explicitly call play_music. Calling search_tracks without calling play_music \
afterwards = FAILURE."""


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def fetch_tools() -> list:
    """Fetch tool definitions from mcpo OpenAPI schema and convert to OpenAI format."""
    resp = requests.get(f"{MCPO_URL}/openapi.json", timeout=10)
    resp.raise_for_status()
    schema = resp.json()

    tools = []
    for path, path_item in schema.get("paths", {}).items():
        op = path_item.get("post")
        if not op:
            continue

        name = path.lstrip("/")
        description = op.get("description") or op.get("summary") or ""

        body = op.get("requestBody", {})
        json_schema = (
            body.get("content", {})
                .get("application/json", {})
                .get("schema", {})
        )

        if "$ref" in json_schema:
            ref = json_schema["$ref"].lstrip("#/").split("/")
            node = schema
            for part in ref:
                node = node[part]
            json_schema = node

        parameters = {
            "type": "object",
            "properties": json_schema.get("properties", {}),
        }
        if json_schema.get("required"):
            parameters["required"] = json_schema["required"]

        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })

    print(f"[tools] Loaded {len(tools)} tools from mcpo: {[t['function']['name'] for t in tools]}")
    return tools


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (local, openai, openrouter, ollama, custom)
# ---------------------------------------------------------------------------

def _openai_base() -> str:
    if LLM_URL:
        return LLM_URL
    return _DEFAULT_BASES.get(LLM_PROVIDER, "http://localhost:8000/v1")


def _llm_request_openai(messages: list, tools: list) -> dict:
    base = _openai_base()
    headers: dict = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    resp = requests.post(
        f"{base}/chat/completions",
        headers=headers,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Anthropic provider (Claude)
# ---------------------------------------------------------------------------

def _tools_to_anthropic(tools: list) -> list:
    """Convert OpenAI-format tool list to Anthropic format."""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


def _messages_to_anthropic(messages: list) -> tuple:
    """
    Extract system prompt and convert message list to Anthropic format.

    OpenAI format:
      [{role:system}, {role:user}, {role:assistant, tool_calls:[...]}, {role:tool}, ...]

    Anthropic format (system is separate, tool results are grouped into user turns):
      system="..."
      [{role:user}, {role:assistant, content:[tool_use]}, {role:user, content:[tool_result]}, ...]
    """
    system = ""
    result: list = []
    pending_tool_results: list = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system = msg.get("content", "")

        elif role == "user":
            if pending_tool_results:
                result.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []
            result.append({"role": "user", "content": msg.get("content", "")})

        elif role == "assistant":
            if pending_tool_results:
                result.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []
            blocks: list = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls", []):
                blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"]),
                })
            result.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})

        elif role == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": msg["content"],
            })

    if pending_tool_results:
        result.append({"role": "user", "content": pending_tool_results})

    return system, result


def _anthropic_to_openai(response: dict) -> dict:
    """Normalise Anthropic response to OpenAI-compatible structure used by the loop."""
    blocks = response.get("content", [])
    text = ""
    tool_calls: list = []

    for block in blocks:
        if block.get("type") == "text":
            text += block["text"]
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block["input"]),
                },
            })

    message: dict = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = "tool_calls" if tool_calls else "stop"
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


def _llm_request_anthropic(messages: list, tools: list) -> dict:
    base = LLM_URL if LLM_URL else "https://api.anthropic.com"
    system, ant_messages = _messages_to_anthropic(messages)
    ant_tools = _tools_to_anthropic(tools)

    headers = {
        "x-api-key": LLM_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body: dict = {
        "model": MODEL,
        "max_tokens": 4096,
        "messages": ant_messages,
        "tools": ant_tools,
    }
    if system:
        body["system"] = system

    resp = requests.post(
        f"{base}/v1/messages",
        headers=headers,
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    return _anthropic_to_openai(resp.json())


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def llm_request(messages: list, tools: list) -> dict:
    if LLM_PROVIDER == "anthropic":
        return _llm_request_anthropic(messages, tools)
    return _llm_request_openai(messages, tools)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def tool_request(name: str, arguments: dict) -> str:
    resp = requests.post(
        f"{MCPO_URL}/{name}",
        json=arguments,
        timeout=60,
    )
    resp.raise_for_status()
    return json.dumps(resp.json(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(user_query: str):
    tools = fetch_tools()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]

    print(f"\n{'='*60}")
    print(f"Provider : {LLM_PROVIDER}")
    print(f"Model    : {MODEL}")
    print(f"User     : {user_query}")
    print(f"{'='*60}\n")

    for i in range(MAX_ITERS):
        print(f"[{i+1}] → LLM ({LLM_PROVIDER})...")
        data   = llm_request(messages, tools)
        choice = data["choices"][0]
        msg    = choice["message"]
        reason = choice.get("finish_reason", "?")

        print(f"[{i+1}] finish_reason = {reason}")

        if msg.get("reasoning_content"):
            print(f"[{i+1}] thinking: {msg['reasoning_content']}")

        assistant_turn: dict = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            assistant_turn["reasoning_content"] = msg["reasoning_content"]
        if msg.get("tool_calls"):
            assistant_turn["tool_calls"] = msg["tool_calls"]
        messages.append(assistant_turn)

        if reason == "tool_calls" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc["function"]
                name = func["name"]
                args = json.loads(func["arguments"])

                print(f"[{i+1}] tool_call → {name}({json.dumps(args, ensure_ascii=False)})")

                try:
                    result = tool_request(name, args)
                except Exception as e:
                    result = json.dumps({"error": str(e)})

                print(f"[{i+1}] tool_result → {result}")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      result,
                })
        else:
            print(f"\n{'='*60}")
            print(f"Assistant: {msg.get('content', '')}")
            print(f"{'='*60}\n")
            return

    print("[!] Reached max iterations without final response")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "play born to be wild steppenwolf"
    run(query)
