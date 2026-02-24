#!/usr/bin/env python3
"""
Manual tool-calling loop test.
Talks directly to llama-server + mcpo.
"""

import json
import sys
import requests

LLM_URL  = "http://localhost:8000"
MCPO_URL = "http://localhost:8001"
MODEL    = "gpt-oss-20b-Q4_K_M.gguf"
MAX_ITERS = 6

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

        # Resolve $ref if present
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


def llm_request(messages: list, tools: list) -> dict:
    resp = requests.post(
        f"{LLM_URL}/v1/chat/completions",
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


def tool_request(name: str, arguments: dict) -> str:
    resp = requests.post(
        f"{MCPO_URL}/{name}",
        json=arguments,
        timeout=60,
    )
    resp.raise_for_status()
    return json.dumps(resp.json(), ensure_ascii=False)


def run(user_query: str):
    tools = fetch_tools()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]

    print(f"\n{'='*60}")
    print(f"User: {user_query}")
    print(f"{'='*60}\n")

    for i in range(MAX_ITERS):
        print(f"[{i+1}] → LLM...")
        data    = llm_request(messages, tools)
        choice  = data["choices"][0]
        msg     = choice["message"]
        reason  = choice.get("finish_reason", "?")

        print(f"[{i+1}] finish_reason = {reason}")

        if msg.get("reasoning_content"):
            print(f"[{i+1}] thinking: {msg['reasoning_content']}")

        # Build assistant turn preserving reasoning_content for next iteration
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
