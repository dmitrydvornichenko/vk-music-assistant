"""
auth_server.py — VK token auth server with remote-browser first-auth and auto-refresh.

How it works
------------
First auth (one time):
  1. User opens http://host:PORT/auth in their browser.
  2. Server starts a headless Chromium via Playwright and opens the VK OAuth page.
  3. The /auth page streams JPEG screenshots over WebSocket and forwards the user's
     mouse / keyboard input back to Chromium — effectively a lightweight remote desktop
     for a single browser tab.
  4. User logs in (handles 2FA, CAPTCHA, etc.) through this remote browser.
  5. When VK redirects to oauth.vk.com/blank.html#access_token=..., the server:
       - Extracts the access_token.
       - Saves the full browser session state (cookies, localStorage) to STATE_FILE.
       - Writes the token to TOKEN_FILE and patches VK_TOKEN in ENV_FILE.

Auto-refresh (every REFRESH_INTERVAL seconds, default 6 h):
  1. A background asyncio task loads the saved session state into a fresh context.
  2. Navigates to the VK OAuth URL.
  3. If the VK session is still alive, VK redirects immediately without showing a
     login page — new access_token is captured silently.
  4. If the session has expired, a warning is logged and the user must re-auth.

Note: requesting scope=audio,offline with response_type=token returns expires_in=0
(token never expires) on Standalone VK apps.  The refresh loop is kept as a safety
net in case VK changes this behaviour.

Endpoints
---------
  GET /          → 302 → /auth
  GET /auth      → remote-browser HTML page
  WS  /ws        → WebSocket: JPEG frame stream + input forwarding
  GET /status    → JSON status (token present, session state saved, next refresh)
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
from pathlib import Path

import aiohttp
from aiohttp import web
from playwright.async_api import async_playwright

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TOKEN_FILE       = Path(os.environ.get("TOKEN_FILE",  ".vk_token"))
STATE_FILE       = Path(os.environ.get("STATE_FILE",  "playwright_state.json"))
ENV_FILE         = Path(os.environ.get("ENV_FILE",    ".env"))
REFRESH_INTERVAL = int(os.environ.get("TOKEN_REFRESH_INTERVAL", str(6 * 3600)))  # 6 h
PORT             = int(os.environ.get("AUTH_PORT", "8080"))

# Remote-browser viewport — must match the canvas dimensions in AUTH_HTML
VIEWPORT = {"width": 1280, "height": 800}

# Same app_id and scope bitmask as vkhost.github.io — known to have audio access.
VK_APP_ID    = "6287487"
REDIRECT_URI = "https://oauth.vk.com/blank.html"
SCOPES       = "408861919"

# ── Module-level mutable state ────────────────────────────────────────────────

_playwright      = None
_browser         = None
_session_lock    = None        # asyncio.Lock, created in startup()
_last_refresh_ts: float = 0.0  # epoch of last successful refresh

# ── Helpers ───────────────────────────────────────────────────────────────────

def _oauth_url() -> str:
    return (
        "https://oauth.vk.com/authorize"
        f"?client_id={VK_APP_ID}"
        "&display=page"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope={SCOPES}"
        "&response_type=token"
        "&v=5.131"
    )


def _parse_token_url(url: str):
    """Return (token: str, expires_in: int) if the URL is a successful OAuth redirect, else None."""
    if "blank.html" not in url or "access_token=" not in url:
        return None
    m = re.search(r"[#&]access_token=([^&\s]+)", url)
    if not m:
        return None
    token = m.group(1)
    exp = re.search(r"expires_in=(\d+)", url)
    expires_in = int(exp.group(1)) if exp else 86400
    return token, expires_in


def _save_token(token: str, expires_in: int) -> None:
    global _last_refresh_ts
    TOKEN_FILE.write_text(token)
    _last_refresh_ts = time.time()
    _patch_env("VK_TOKEN", token)
    ttl = f"{expires_in}s" if expires_in else "never (offline token)"
    log.info("Token saved → %s  (ttl=%s)", TOKEN_FILE, ttl)


def _patch_env(key: str, value: str) -> None:
    if not ENV_FILE.is_file():   # Path("") or missing file → skip silently
        return
    text = ENV_FILE.read_text()
    if re.search(rf"^{key}=", text, re.MULTILINE):
        text = re.sub(rf"^{key}=.*$", f"{key}={value}", text, flags=re.MULTILINE)
    else:
        text = text.rstrip("\n") + f"\n{key}={value}\n"
    ENV_FILE.write_text(text)
    log.info("Patched %s in %s", key, ENV_FILE)


async def _handle_input(page, data: dict) -> None:
    """Forward a single user-input event to the Playwright page."""
    t = data.get("type")
    try:
        if t == "click":
            await page.mouse.click(data["x"], data["y"])
        elif t == "dblclick":
            await page.mouse.dblclick(data["x"], data["y"])
        elif t == "move":
            await page.mouse.move(data["x"], data["y"])
        elif t == "wheel":
            await page.mouse.wheel(data.get("dx", 0), data.get("dy", 0))
        elif t == "key":
            await page.keyboard.press(data["key"])
        elif t == "type":
            await page.keyboard.type(data["text"])
    except Exception as e:
        log.debug("Input error (%s): %s", t, e)


# ── Silent token refresh ──────────────────────────────────────────────────────

async def _silent_refresh() -> tuple | None:
    """
    Open a throwaway Chromium context with the saved session state, navigate to
    the VK OAuth URL, and capture the new access_token from the redirect URL.

    Returns (token, expires_in) on success, or None if the session has expired.
    """
    if not STATE_FILE.exists():
        log.debug("No session state file — skipping silent refresh")
        return None

    context = await _browser.new_context(
        storage_state=str(STATE_FILE),
        viewport=VIEWPORT,
    )
    page = await context.new_page()
    try:
        await page.goto(_oauth_url(), wait_until="domcontentloaded", timeout=20_000)
        # Poll for 15 s
        for _ in range(150):
            await asyncio.sleep(0.1)
            result = _parse_token_url(page.url)
            if result:
                return result
            cur = page.url
            if "login.vk.com" in cur or "/login" in cur or "act=login" in cur:
                log.warning("Silent refresh: VK session expired — re-auth required at /auth")
                return None
        log.warning("Silent refresh: timed out waiting for blank.html redirect")
        return None
    finally:
        await page.close()
        await context.close()


async def refresh_loop() -> None:
    """Background task: refresh the VK token every REFRESH_INTERVAL seconds."""
    while _browser is None:
        await asyncio.sleep(2)

    # Brief initial delay — let the server settle after startup
    await asyncio.sleep(60)

    while True:
        try:
            result = await _silent_refresh()
            if result:
                _save_token(*result)
                log.info("Background token refresh OK")
        except Exception:
            log.exception("Background token refresh error")
        await asyncio.sleep(REFRESH_INTERVAL)


# ── HTTP handlers ─────────────────────────────────────────────────────────────

async def handle_root(request: web.Request) -> web.Response:
    raise web.HTTPFound("/auth")


async def handle_status(request: web.Request) -> web.Response:
    token = TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None
    next_refresh = None
    if _last_refresh_ts:
        next_refresh = int(_last_refresh_ts + REFRESH_INTERVAL - time.time())
    return web.json_response({
        "token_present":      bool(token),
        "token_preview":      (token[:16] + "…") if token else None,
        "session_state_saved": STATE_FILE.exists(),
        "refresh_interval_sec": REFRESH_INTERVAL,
        "next_refresh_in_sec":  next_refresh,
        "vk_app_id_set":       bool(VK_APP_ID),
    })


async def handle_auth_page(request: web.Request) -> web.Response:
    html = AUTH_HTML.replace("__VPW__", str(VIEWPORT["width"])).replace("__VPH__", str(VIEWPORT["height"]))
    return web.Response(text=html, content_type="text/html")


async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    log.info("WebSocket connected from %s", request.remote)

    if _browser is None:
        await ws.send_json({"type": "error", "message": "Browser is not ready yet — retry in a few seconds"})
        await ws.close()
        return ws

    if _session_lock.locked():
        await ws.send_json({"type": "error", "message": "Another auth session is already active — try again later"})
        await ws.close()
        return ws

    async with _session_lock:
        await _run_auth_session(ws)

    return ws


# ── Remote-browser auth session ───────────────────────────────────────────────

async def _run_auth_session(ws: web.WebSocketResponse) -> None:
    """
    Core remote-browser loop:
      - Opens a new Playwright page (reusing saved session state if available).
      - Two concurrent asyncio tasks run in parallel:
          sender — takes JPEG screenshots every ~150 ms, checks for auth completion.
          reader — reads WebSocket messages from the client and forwards them as
                   mouse/keyboard input to Playwright.
      - When VK redirects to blank.html with access_token, sender saves state,
        writes the token, and sends a "done" message to the client.
    """
    context = None
    page    = None
    done    = asyncio.Event()

    try:
        storage = str(STATE_FILE) if STATE_FILE.exists() else None
        context = await _browser.new_context(
            storage_state=storage,
            viewport=VIEWPORT,
            locale="ru-RU",
        )
        page = await context.new_page()

        await ws.send_json({"type": "status", "message": "Opening VK login page…"})
        await page.goto(_oauth_url(), wait_until="domcontentloaded", timeout=30_000)

        # ── Sender task ───────────────────────────────────────────────────────
        async def sender() -> None:
            while not ws.closed and not done.is_set():
                result = _parse_token_url(page.url)
                if result:
                    token, expires_in = result
                    await context.storage_state(path=str(STATE_FILE))
                    log.info("Auth complete — session state saved to %s", STATE_FILE)
                    _save_token(token, expires_in)
                    ttl = f"expires_in={expires_in}s" if expires_in else "offline (never expires)"
                    await ws.send_json({
                        "type":       "done",
                        "message":    f"Authorized! {ttl}",
                        "expires_in": expires_in,
                    })
                    done.set()
                    break

                try:
                    shot = await page.screenshot(type="jpeg", quality=65, timeout=5_000)
                    await ws.send_json({"type": "frame", "data": base64.b64encode(shot).decode()})
                except Exception as e:
                    log.debug("Screenshot error: %s", e)

                await asyncio.sleep(0.15)

        # ── Reader task ───────────────────────────────────────────────────────
        async def reader() -> None:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        await _handle_input(page, json.loads(msg.data))
                    except Exception as e:
                        log.debug("Input handling error: %s", e)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                    break
            done.set()

        sender_t = asyncio.create_task(sender())
        reader_t = asyncio.create_task(reader())
        await asyncio.wait([sender_t, reader_t], return_when=asyncio.FIRST_COMPLETED)
        sender_t.cancel()
        reader_t.cancel()
        await asyncio.gather(sender_t, reader_t, return_exceptions=True)

    except Exception:
        log.exception("Auth session error")
        if not ws.closed:
            await ws.send_json({"type": "error", "message": "Auth session failed — check server logs"})
    finally:
        if page and not page.is_closed():
            await page.close()
        if context:
            await context.close()
        log.info("Auth session ended")


# ── App lifecycle ─────────────────────────────────────────────────────────────

async def startup(app: web.Application) -> None:
    global _playwright, _browser, _session_lock
    _session_lock = asyncio.Lock()
    log.info("Launching Playwright / Chromium (headless)…")
    _playwright = await async_playwright().start()
    _browser    = await _playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-extensions",
        ],
    )
    log.info("Chromium ready")
    asyncio.create_task(refresh_loop())

    if TOKEN_FILE.exists():
        log.info("Existing token found in %s", TOKEN_FILE)
    else:
        log.info("No token yet — visit http://localhost:%d/auth to authorize", PORT)


async def shutdown(app: web.Application) -> None:
    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()
    log.info("Playwright shut down")


# ── Embedded client HTML ──────────────────────────────────────────────────────

AUTH_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>VK Music — Connect Account</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f0f1a;
      color: #cdd6f4;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
      gap: 14px;
    }
    h1 { font-size: 1.1rem; color: #89b4fa; letter-spacing: .05em; }
    #status {
      font-size: 0.85rem;
      color: #89dceb;
      min-height: 1.2em;
      text-align: center;
      max-width: 660px;
    }
    #canvas-wrap {
      border: 2px solid #313244;
      border-radius: 10px;
      overflow: hidden;
      cursor: crosshair;
      line-height: 0;
      box-shadow: 0 8px 32px #000a;
    }
    canvas { display: block; }
    #done-banner {
      display: none;
      padding: 14px 32px;
      background: #1e3a2f;
      border: 1px solid #40a870;
      border-radius: 8px;
      color: #a6e3a1;
      font-size: 1rem;
      text-align: center;
    }
    #hint {
      font-size: 0.75rem;
      color: #585b70;
      text-align: center;
      max-width: 500px;
    }
  </style>
</head>
<body>
  <h1>VK Music Assistant — Connect VK Account</h1>
  <div id="status">Connecting to remote browser…</div>
  <div id="canvas-wrap">
    <canvas id="c" width="__VPW__" height="__VPH__"></canvas>
  </div>
  <div id="done-banner"></div>
  <p id="hint">Click and type inside the browser above. Auth happens once; the token refreshes automatically every 6 hours.</p>

  <script>
    const VPW = __VPW__, VPH = __VPH__;
    const canvas = document.getElementById('c');
    const ctx    = canvas.getContext('2d');
    const status = document.getElementById('status');
    const done   = document.getElementById('done-banner');

    /* ── Scale canvas to fit the page ──────────────────────────────────── */
    function resize() {
      const maxW  = Math.min(window.innerWidth  - 40, VPW);
      const maxH  = Math.min(window.innerHeight * 0.78, VPH);
      const scale = Math.min(maxW / VPW, maxH / VPH);
      canvas.style.width  = Math.round(VPW * scale) + 'px';
      canvas.style.height = Math.round(VPH * scale) + 'px';
    }
    resize();
    window.addEventListener('resize', resize);

    /* ── WebSocket ──────────────────────────────────────────────────────── */
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws    = new WebSocket(proto + '//' + location.host + '/ws');
    ws.onopen  = () => status.textContent = 'Remote browser starting…';
    ws.onerror = () => status.textContent = '✗ WebSocket error — reload the page';
    ws.onclose = () => { if (!done.style.display || done.style.display === 'none') status.textContent = 'Disconnected'; }

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'frame') {
        const img = new Image();
        img.onload = () => ctx.drawImage(img, 0, 0, VPW, VPH);
        img.src = 'data:image/jpeg;base64,' + msg.data;
      } else if (msg.type === 'status') {
        status.textContent = msg.message;
      } else if (msg.type === 'done') {
        status.textContent = '✓ ' + msg.message;
        done.style.display = 'block';
        done.textContent   = '✓ Authorization complete! You can close this tab. Token will auto-refresh every 6 h.';
        canvas.style.opacity = '0.25';
        canvas.style.pointerEvents = 'none';
      } else if (msg.type === 'error') {
        status.textContent = '✗ ' + msg.message;
      }
    };

    /* ── Input forwarding ───────────────────────────────────────────────── */
    function toVP(e) {
      const r = canvas.getBoundingClientRect();
      return {
        x: Math.round((e.clientX - r.left) / r.width  * VPW),
        y: Math.round((e.clientY - r.top)  / r.height * VPH),
      };
    }

    function send(obj) { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); }

    canvas.addEventListener('click',     e => send({ type: 'click',    ...toVP(e) }));
    canvas.addEventListener('dblclick',  e => send({ type: 'dblclick', ...toVP(e) }));
    canvas.addEventListener('mousemove', e => send({ type: 'move',     ...toVP(e) }));
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      send({ type: 'wheel', dx: e.deltaX, dy: e.deltaY });
    }, { passive: false });

    /* Keyboard — captured globally so the user doesn't need to click canvas first */
    document.addEventListener('keydown', e => {
      if (['F5', 'F12', 'F11'].includes(e.key)) return;
      if (e.ctrlKey && ['r', 'l', 't', 'w', 'n'].includes(e.key.toLowerCase())) return;
      e.preventDefault();
      /* Map JS key names → Playwright key names (mostly 1:1, handle a few specials) */
      const map = { ' ': 'Space', 'ArrowUp': 'ArrowUp', 'ArrowDown': 'ArrowDown',
                    'ArrowLeft': 'ArrowLeft', 'ArrowRight': 'ArrowRight' };
      send({ type: 'key', key: map[e.key] ?? e.key });
    });

    canvas.setAttribute('tabindex', '0');
    canvas.addEventListener('mousedown', () => canvas.focus());
  </script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = web.Application()
    app.router.add_get("/",       handle_root)
    app.router.add_get("/auth",   handle_auth_page)
    app.router.add_get("/ws",     handle_ws)
    app.router.add_get("/status", handle_status)
    app.on_startup.append(startup)
    app.on_shutdown.append(shutdown)
    log.info("Auth server starting on port %d", PORT)
    log.info("Open http://localhost:%d/auth to authorize VK", PORT)
    web.run_app(app, port=PORT, access_log=None)


if __name__ == "__main__":
    main()
