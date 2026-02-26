# vk-music-assistant

MCP server that searches and plays music from VK Audio.  
Exposes tools over HTTP via [mcpo](https://github.com/open-webui/mcpo).

## Tools

| Tool | Description |
|---|---|
| `search_tracks` | Search VK Audio by artist / title |
| `search_album` | Search albums and list their tracks |
| `play_music` | Queue and play one or several tracks by ID |
| `play_album` | Queue all tracks from an album/playlist by ID (safe for large albums) |
| `play_user_audio` | Queue a user's full audio library (own or by ID), with optional shuffle |
| `find_user` | Find VK user by name — friends first, then global search |
| `whoami` | Return info about the account whose token is in use |
| `pause_music` | Pause playback, saving current position |
| `resume_music` | Resume from where playback was paused |
| `stop_music` | Stop playback and clear the queue |
| `skip_tracks` | Skip one or more tracks in the queue |
| `get_queue` | Show currently playing track and upcoming queue |
| `volume_up` | Increase system volume by 10% |
| `volume_down` | Decrease system volume by 10% |
| `music_health` | Health check: token status, current mode |

## Quick start

### 1. Clone & configure

```bash
git clone https://github.com/dmitrydvornichenko/vk-music-assistant.git
cd vk-music-assistant
cp .env.example .env
# edit .env — at minimum set VK_TOKEN
```

---

### Run locally

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

### Run in Docker

The container plays audio through the **host** PulseAudio daemon.  
Set `PULSE_SERVER` in `.env` to point to your PulseAudio instance and mount any required sockets yourself.

**Linux (Unix socket)**

```
PULSE_SERVER=unix:/tmp/pulse-native
```

Mount the host socket into the container by extending `docker-compose.yml` with an extra volume, e.g.:
```yaml
volumes:
  - /run/user/1000/pulse/native:/tmp/pulse-native:ro
```

**Linux (TCP)**

```
PULSE_SERVER=localhost:4713
```
Make sure to enable TCP in PulseAudio in `/etc/pulse/default.pa`

**Windows / macOS (TCP)**

Enable TCP in PulseAudio and set:
```
PULSE_SERVER=tcp:host.docker.internal:4713
```

```bash
docker compose up --build -d
```

---

### Auth server (token management)

`auth_server.py` runs as a **separate container** (`auth`, port `8080`).  
It handles VK login and keeps the token fresh automatically — no manual token copying needed.

**How it works:**

1. Open `http://your-host:8080/auth` in a browser.
2. A headless Chromium streams the VK login page to your browser via WebSocket (remote desktop for a single tab). Log in once — handles password, 2FA, CAPTCHA.
3. After login the browser session state (cookies) is saved to `playwright_state.json`.
4. A background task silently navigates through VK OAuth every 6 hours, captures the new `access_token`, and writes it to `.vk_token`.
5. `music-mcp` hot-reloads the token from `.vk_token` on every API call — no restart needed.

Both containers share a named Docker volume (`vk_data`) for `.vk_token` and `playwright_state.json`.

**Endpoints:**

| Endpoint | Description |
|---|---|
| `GET /auth` | Remote-browser login UI |
| `GET /status` | JSON: token present, session saved, next refresh in N seconds |

**Auth server environment variables:**

| Variable | Default | Description |
|---|---|---|
| `AUTH_PORT` | `8080` | HTTP port |
| `TOKEN_FILE` | `.vk_token` | Where to write the refreshed token |
| `STATE_FILE` | `playwright_state.json` | Playwright browser session state |
| `ENV_FILE` | `.env` | `.env` file to patch `VK_TOKEN` in (set to `""` in Docker) |
| `TOKEN_REFRESH_INTERVAL` | `21600` | Refresh interval in seconds (default 6 h) |

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

---

## Test script

`cli.py` — CLI client with LLM tool-calling loop that talks to an LLM + mcpo.  
Supports **local llama-server** and any major cloud provider.

### LLM environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `local` | `local` · `openai` · `anthropic` · `openrouter` · `ollama` · `custom` |
| `LLM_API_KEY` | — | API key (required for cloud providers) |
| `LLM_URL` | *(auto)* | Override base URL; useful for custom/self-hosted endpoints |
| `LLM_MODEL` | `gpt-oss-20b-Q4_K_M.gguf` | Model name |
| `MCPO_URL` | `http://localhost:8001` | mcpo endpoint |

Default base URLs (used when `LLM_URL` is not set):

| Provider | Default base URL |
|---|---|
| `local` | `http://localhost:8000/v1` |
| `openai` | `https://api.openai.com/v1` |
| `anthropic` | `https://api.anthropic.com` |
| `openrouter` | `https://openrouter.ai/api/v1` |
| `ollama` | `http://localhost:11434/v1` |

### Usage examples

```bash
# local llama-server (default)
python cli.py "play born to be wild steppenwolf"

# OpenAI
LLM_PROVIDER=openai LLM_API_KEY=sk-... LLM_MODEL=gpt-4o \
  python cli.py "play born to be wild"

# Anthropic Claude
LLM_PROVIDER=anthropic LLM_API_KEY=sk-ant-... LLM_MODEL=claude-opus-4-5 \
  python cli.py "play born to be wild"

# OpenRouter (access to many models via one key)
LLM_PROVIDER=openrouter LLM_API_KEY=sk-or-... LLM_MODEL=openai/gpt-4o \
  python cli.py "play born to be wild"

# Ollama (local, no key required)
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:32b \
  python cli.py "play born to be wild"

# Any OpenAI-compatible endpoint
LLM_PROVIDER=custom LLM_URL=https://my-gateway.example.com/v1 LLM_API_KEY=... \
  LLM_MODEL=my-model python cli.py "play born to be wild"
```

Or configure via `.env`:

```bash
cp .env.example .env
# set LLM_PROVIDER, LLM_API_KEY, LLM_MODEL in .env
export $(grep -v '^#' .env | xargs)
python cli.py "play born to be wild"
```

### How it works

1. Fetches tool schemas from mcpo (`/openapi.json`) and converts them to OpenAI tool-calling format.
2. Sends user query to the LLM with the tools.
3. Executes tool calls returned by the LLM against mcpo.
4. Feeds results back to the LLM until it produces a final text response.

For the **Anthropic** provider the script transparently converts between OpenAI-style tool calls and Anthropic's native `tool_use` / `tool_result` format — no extra dependencies needed beyond `requests`.

## License

[MIT](LICENSE)
