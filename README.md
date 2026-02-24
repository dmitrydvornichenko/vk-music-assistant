# vk-music-assistant

MCP server that searches and plays music from VK Audio.  
Exposes tools over HTTP via [mcpo](https://github.com/open-webui/mcpo), compatible with Open WebUI, Claude Desktop, and any OpenAI-tool-calling LLM.

## Tools

| Tool | Description |
|---|---|
| `search_tracks` | Search VK Audio by artist / title |
| `search_album` | Search albums and list their tracks |
| `play_music` | Queue and play one or several tracks |
| `stop_music` | Stop playback immediately |
| `music_health` | Health check: token status, current mode |

## Quick start

### 1. Clone & configure

```bash
git clone https://github.com/yourname/vk-music-assistant.git
cd vk-music-assistant
cp .env.example .env
# edit .env — at minimum set VK_TOKEN
```

---

### Option A — local run (no Docker)

Prerequisites: Python 3.11+, `ffmpeg`, `paplay` (pulseaudio-utils).

```bash
pip install -r requirements.txt
```

PulseAudio runs on the same host, so no extra configuration is needed.  
`PULSE_SERVER` defaults to `unix:$XDG_RUNTIME_DIR/pulse/native` (i.e. `/run/user/1000/pulse/native`), which is the standard socket for the current user session.

```bash
# optional: override only if your socket is in a non-standard location
export PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native

VK_TOKEN=your_token mcpo --port 8001 -- python music_mcp.py
```

The OpenAPI-compatible endpoint is available at `http://localhost:8001`.

---

### Option B — Docker

The container plays audio through the **host** PulseAudio daemon.  
Set `PULSE_SERVER` in `.env` to point to your PulseAudio instance and mount any required sockets yourself.

**Linux (Unix socket example)**

```
PULSE_SERVER=unix:/tmp/pulse-native
```

Mount the host socket into the container by extending `docker-compose.yml` with an extra volume, e.g.:
```yaml
volumes:
  - /run/user/1000/pulse/native:/tmp/pulse-native:ro
```

**Windows / macOS (TCP)**

Enable TCP in PulseAudio and set:
```
PULSE_SERVER=tcp:host.docker.internal:4713
```

```bash
docker compose up --build -d
```

---

## Playback modes

| `MODE` | Behaviour |
|---|---|
| `stream` | Decrypt and pipe HLS segments directly to PulseAudio in real time |
| `file` | Download and convert the full track to MP3 in `TEMP_DIR`, then play |

## Environment variables

See [`.env.example`](.env.example) for all options.

| Variable | Default | Description |
|---|---|---|
| `VK_TOKEN` | — | VK API token (required) |
| `PULSE_SERVER` | `unix:$XDG_RUNTIME_DIR/pulse/native` | PulseAudio server address |
| `MODE` | `stream` | Playback mode: `stream` or `file` |
| `TEMP_DIR` | `/tmp/music-mcp` | Temp dir for downloaded files (`file` mode) |

## Test script

`test_tool_loop.py` — manual tool-calling loop that talks directly to a local llama-server + mcpo without Open WebUI:

```bash
python test_tool_loop.py "включи born to be wild steppenwolf"
```

Requires `llama-server` running on `localhost:8000` and the MCP server on `localhost:8001`.

## License

[MIT](LICENSE)
