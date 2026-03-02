#!/usr/bin/env python3
"""
MCP Server - Music Tools

Provides music search and playback via VK Music.
"""

import errno
import os
import subprocess
import tempfile
import logging
import re
import threading
import time
import sys

from mcp.server.fastmcp import FastMCP

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("music_mcp")

if log_level != "DEBUG":
    for _lib in ("httpx", "httpcore", "urllib3", "requests"):
        logging.getLogger(_lib).setLevel(logging.WARNING)

VK_TOKEN = os.environ.get("VK_TOKEN", "")

# Hot-reload: watch TOKEN_FILE for changes and update VK_TOKEN without restart.
# TOKEN_FILE env var allows Docker to point at a shared volume path (/data/.vk_token).
# Falls back to .vk_token in the script directory for local runs.
_TOKEN_FILE = os.environ.get(
    "TOKEN_FILE",
    os.path.join(os.path.dirname(__file__) or ".", ".vk_token"),
)
_token_file_mtime: float = 0.0
_token_lock_reload = threading.Lock()


def _reload_token_if_changed():
    """Check TOKEN_FILE mtime; if changed, reload VK_TOKEN from the file."""
    global VK_TOKEN, _token_file_mtime, _cached_vk_user
    try:
        mtime = os.path.getmtime(_TOKEN_FILE)
    except OSError:
        return
    if mtime <= _token_file_mtime:
        return
    with _token_lock_reload:
        # Re-check inside lock (another thread might have already reloaded)
        try:
            mtime = os.path.getmtime(_TOKEN_FILE)
        except OSError:
            return
        if mtime <= _token_file_mtime:
            return
        try:
            new_token = open(_TOKEN_FILE).read().strip()
            if new_token and new_token != VK_TOKEN:
                VK_TOKEN = new_token
                _cached_vk_user = None  # invalidate user info cache
                log.info("VK_TOKEN hot-reloaded from %s", _TOKEN_FILE)
            _token_file_mtime = mtime
        except Exception as e:
            log.warning("Failed to reload token from %s: %s", _TOKEN_FILE, e)
_pulse_default = "unix:{}/pulse/native".format(
    os.environ.get("XDG_RUNTIME_DIR", "/run/user/1000")
)
PULSE_SERVER = os.environ.get("PULSE_SERVER", _pulse_default)
MODE = os.environ.get("MODE", "stream").lower()
TEMP_DIR = os.environ.get("TEMP_DIR", "/tmp/music-mcp")

os.makedirs(TEMP_DIR, exist_ok=True)

active_music_processes: list = []
active_music_lock = threading.Lock()

# Playback queue and worker
playback_queue: list = []
playback_lock = threading.Lock()
playback_thread: threading.Thread | None = None
playback_running = False

# Playback position tracking (updated by worker thread)
current_track_info: dict | None = None
current_track_url: str = ""   # HLS manifest URL — saved separately for resume re-fetch
current_chunks: list = []
current_chunk_idx: int = 0

# Pause / resume state
pause_state: dict | None = None
playback_paused: bool = False


# Load token from file immediately on startup (before first API call)
_reload_token_if_changed()
if not VK_TOKEN:
    log.warning("VK_TOKEN not set — waiting for auth server to write %s", _TOKEN_FILE)

mcp = FastMCP("music-server")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def _wait_or_kill(proc: subprocess.Popen, timeout: float, label: str):
    """Wait for a process to finish; kill it if it doesn't within timeout seconds."""
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("%s did not finish in %.0fs, killing", label, timeout)
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def _get_track_chunks(track):
    """Parse a VK HLS playlist and return a list of segment descriptors.

    How it works
    ------------
    VK serves audio as an HLS (HTTP Live Streaming) playlist — a plain-text .m3u8
    file that describes a list of short MPEG-TS segments (.ts files), each typically
    4-10 seconds long.  The segments may be AES-128 encrypted; each encrypted group
    shares a key URL that is fetched once and reused.

    Step 1 — fetch the playlist text from track["url"].

    Step 2 — split into key-sections.  Each section begins with an
      #EXT-X-KEY:METHOD=...,URI=...
    line and contains all segments encrypted with that key.  We iterate over sections
    using a regex that captures everything up to the next #EXT-X-KEY tag.

    Step 3 — inside each section find every .ts segment line and build a chunk dict:
      {
        "file_url":   full URL of the .ts segment (base dir + filename + query tail),
        "key_method": encryption method ("AES-128" or "NONE"),
        "key_url":    URL of the AES key (None for unencrypted),
        "file":       bare filename (used for deduplication / logging),
        "extinf":     segment duration in seconds from #EXTINF tag (float),
      }

    Step 4 — for each unique key URL, download the raw AES-128 key bytes once and
    cache them; store the bytes in chunk["key"] so _decrypt_chunk can use them
    without a redundant network request.

    Step 5 — extract #EXTINF:<seconds>, tags from the playlist in order.  These are
    guaranteed to appear in the same order as the segment lines, so extinf_values[i]
    corresponds directly to chunks[i].  The extinf value enables precise remaining-
    duration calculation for pause/resume and for the process-kill timeout.

    Returns
    -------
    list of chunk dicts ready to be passed to _decrypt_chunk() one by one.
    """
    import requests as req
    # accept either object with .url or dict with 'url'
    url = getattr(track, "url", None) or (track.get("url") if isinstance(track, dict) else None)
    if not url:
        raise ValueError("Track has no url")
    urldir = os.path.dirname(url)
    playlist = req.get(url).text

    chunks = []
    for match in re.finditer(r'(#EXT-X-KEY:(?:(?!#EXT-X-KEY:).)*)', playlist, re.DOTALL):
        section = match.group(0)
        section_match = re.match(
            r'.*?METHOD=(?P<METHOD>[^,\n\r]+)(?:,URI=(?P<URI>[^\n]+)|)',
            section, re.IGNORECASE,
        )
        if section_match is None:
            continue
        key_method = section_match.group("METHOD")
        key_url = section_match.group("URI")
        if key_url:
            key_url = key_url.strip("'\"")
            if not key_url.startswith(("http://", "https://")):
                key_url = f"{urldir}/{key_url}"

        for match2 in re.finditer(
            r'.*?(?P<file>^[^\r\n]+\.ts)(?P<extra>[^\n\r]+)',
            section, re.IGNORECASE | re.DOTALL | re.MULTILINE,
        ):
            file = match2.group("file")
            extra = match2.group("extra")
            file_url = f"{urldir}/{file}{extra}"
            chunks.append({"file_url": file_url, "key_method": key_method, "key_url": key_url,
                           "file": file, "extinf": 0.0})

    # Step 5: assign #EXTINF durations in order — same index as segments in playlist
    extinf_values = [float(m.group(1)) for m in re.finditer(r'#EXTINF:([\d.]+)', playlist)]
    for i, chunk in enumerate(chunks):
        if i < len(extinf_values):
            chunk["extinf"] = extinf_values[i]

    cached_keys: dict = {}
    for chunk in chunks:
        key_url = chunk.get("key_url")
        if not key_url:
            chunk["key"] = None
            continue
        if key_url not in cached_keys:
            import requests as req
            cached_keys[key_url] = req.get(key_url).content
        chunk["key"] = cached_keys[key_url]

    total_dur = sum(c["extinf"] for c in chunks)
    log.debug("Found %d segments, total duration %.1fs", len(chunks), total_dur)
    return chunks


def _decrypt_chunk(chunk, retries: int = 4, backoff: float = 1.5) -> bytes:
    from urllib.error import URLError
    url = chunk["file_url"]
    last_exc: Exception | None = None
    for attempt in range(retries):
        if attempt:
            delay = backoff * attempt
            log.warning("Retry %d/%d for segment %s (delay %.1fs)", attempt, retries - 1, url, delay)
            time.sleep(delay)
        try:
            if chunk["key"]:
                from urllib.request import urlopen
                from Crypto.Cipher import AES
                from Crypto.Util.Padding import unpad
                key = chunk["key"]
                file_in = urlopen(url, timeout=15)
                iv = file_in.read(16)
                ciphered_data = file_in.read()
                file_in.close()
                cipher = AES.new(key, AES.MODE_CBC, iv=iv)
                return unpad(cipher.decrypt(ciphered_data), AES.block_size)
            import requests as req
            resp = req.get(url, timeout=15)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            last_exc = e
            log.warning("Segment fetch error (attempt %d/%d): %s", attempt + 1, retries, e)
    raise last_exc


def _download_and_save_mp3(output_path: str, chunks: list):
    out_dir = os.path.dirname(os.path.abspath(output_path))
    fd, temp_path = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    os.close(fd)
    try:
        ffmpeg_proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "mpegts", "-i", "pipe:0", "-f", "mp3", "-c:a", "libmp3lame", "-q:a", "2", temp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for i, chunk in enumerate(chunks):
            if ffmpeg_proc.poll() is not None:
                stderr = ffmpeg_proc.stderr.read().decode("utf-8", errors="ignore")
                raise RuntimeError(
                    f"ffmpeg terminated unexpectedly at segment {i+1} "
                    f"with code {ffmpeg_proc.returncode}. stderr: {stderr}"
                )
            log.debug("Processing segment %d/%d", i + 1, len(chunks))
            data = _decrypt_chunk(chunk)
            try:
                ffmpeg_proc.stdin.write(data)
                ffmpeg_proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                if getattr(e, "errno", None) == errno.EPIPE:
                    stderr = ffmpeg_proc.stderr.read().decode("utf-8", errors="ignore")
                    raise RuntimeError(
                        f"ffmpeg died at segment {i+1} (broken pipe). stderr: {stderr}"
                    ) from e
                raise

        ffmpeg_proc.stdin.close()
        ret = ffmpeg_proc.wait()
        if ret != 0:
            stderr = ffmpeg_proc.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg failed with code {ret}. stderr: {stderr}")

        os.replace(temp_path, output_path)
        log.debug("Saved MP3 to %s", output_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _vk_call(method: str, params: dict) -> dict:
    """Call VK API method and return parsed JSON response (raises on error)."""
    import requests

    _reload_token_if_changed()

    base = f"https://api.vk.com/method/{method}"
    req_params = dict(params or {})
    if VK_TOKEN:
        req_params["access_token"] = VK_TOKEN
    req_params.setdefault("v", "5.131")
    # Many audio endpoints accept extended=1 to include extra data
    if "extended" not in req_params:
        req_params["extended"] = 1

    resp = requests.get(base, params=req_params, timeout=15)
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"VK API returned non-JSON response: {e}")
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"VK API error {err.get('error_code')}: {err.get('error_msg')}")
    return data.get("response", {})


_cached_vk_user: dict | None = None


def _fetch_vk_user() -> dict | None:
    """Fetch and cache VK user info for the current token. Returns user dict or None."""
    global _cached_vk_user
    if _cached_vk_user is not None:
        return _cached_vk_user
    if not VK_TOKEN:
        return None
    try:
        resp = _vk_call("users.get", {"fields": "photo_50"})
        users = resp if isinstance(resp, list) else []
        if users:
            _cached_vk_user = users[0]
            return _cached_vk_user
    except Exception as e:
        log.warning("Failed to fetch VK user info: %s", e)
    return None


def _validate_track_item(item: dict) -> bool:
    """Validate that VK track item has a reachable URL. Returns True if playable."""
    import requests

    url = item.get("url")
    if not url:
        return False
    try:
        # Try HEAD first
        resp = requests.head(url, timeout=5, allow_redirects=True)
        if resp.status_code == 200:
            return True
        # Fallback to small GET
        headers = {"Range": "bytes=0-1023"}
        resp = requests.get(url, headers=headers, timeout=7, stream=True)
        if resp.status_code in (200, 206):
            return True
    except Exception:
        return False
    return False


def _enqueue_tracks(track_objs: list, play_now: bool = False):
    """Add tracks (dicts with 'url' etc) to queue and ensure worker is running."""
    global playback_thread, playback_running
    with playback_lock:
        if play_now:
            # insert at front preserving order
            for t in reversed(track_objs):
                playback_queue.insert(0, t)
        else:
            playback_queue.extend(track_objs)

        if not playback_running:
            playback_running = True
            playback_thread = threading.Thread(target=_playback_loop, daemon=True)
            playback_thread.start()


def _playback_loop():
    """Worker loop: consume queue and play each track sequentially."""
    global playback_running, current_track_info, current_track_url, current_chunks, current_chunk_idx
    try:
        while True:
            with playback_lock:
                if playback_paused:
                    playback_running = False
                    return
                if not playback_queue:
                    playback_running = False
                    return
                track = playback_queue.pop(0)

            # Use pre-loaded chunks for resumed tracks, otherwise fetch from URL
            if "_resume_chunks" in track:
                chunks = track["_resume_chunks"]
            else:
                try:
                    chunks = _get_track_chunks(track)
                except Exception as e:
                    log.warning("Failed to get chunks for queued track: %s", e)
                    continue

            if not chunks:
                log.warning("No segments for queued track, skipping")
                continue

            current_track_url = track.get("url", "")
            # _duration is pre-calculated remaining duration for resumed tracks
            play_duration = float(track.get("_duration") or track.get("duration") or 0)
            current_track_info = {k: v for k, v in track.items()
                                  if k not in ("url", "_resume_chunks", "_duration")}
            current_chunks = chunks
            current_chunk_idx = 0

            try:
                if MODE == "file":
                    _file_music_thread(chunks, duration=play_duration)
                else:
                    _stream_music_thread(chunks, duration=play_duration)
            except Exception as e:
                log.exception("Error while playing queued track: %s", e)
                continue
    finally:
        playback_running = False
        current_track_info = None
        current_track_url = ""
        current_chunks = []
        current_chunk_idx = 0


def _file_music_thread(chunks: list, duration: float = 0.0):
    """Download chunks to a temp MP3 file, then play it via ffmpeg+paplay."""
    temp_path = None
    ffmpeg_dec = None
    paplay_proc = None
    try:
        fd, temp_path = tempfile.mkstemp(dir=TEMP_DIR, suffix=".mp3")
        os.close(fd)

        log.info("Downloading %d segments to %s ...", len(chunks), temp_path)
        _download_and_save_mp3(temp_path, chunks)
        log.info("Download complete, playing from file...")

        env = os.environ.copy()
        if PULSE_SERVER:
            env["PULSE_SERVER"] = PULSE_SERVER

        ffmpeg_dec = subprocess.Popen(
            ["ffmpeg", "-i", temp_path, "-f", "s16le", "-ar", "44100", "-ac", "2", "pipe:1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        paplay_proc = subprocess.Popen(
            ["paplay", "--raw", "--format=s16le", "--rate=44100", "--channels=2"],
            stdin=ffmpeg_dec.stdout,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        with active_music_lock:
            active_music_processes[:] = [ffmpeg_dec, paplay_proc]

        play_timeout = max(10.0, duration + 1.0) if duration else 60.0
        _wait_or_kill(ffmpeg_dec, timeout=play_timeout, label="ffmpeg (file)")
        _wait_or_kill(paplay_proc, timeout=play_timeout, label="paplay (file)")
        log.info("File playback complete")
    except Exception as e:
        log.exception("File mode playback error: %s", e)
        for proc in [ffmpeg_dec, paplay_proc]:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
    finally:
        with active_music_lock:
            active_music_processes.clear()
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                log.debug("Removed temp file %s", temp_path)
            except Exception as e:
                log.warning("Could not remove temp file %s: %s", temp_path, e)


def _stream_music_thread(chunks: list, duration: float = 0.0):
    global current_chunk_idx
    ffmpeg_proc = None
    paplay_proc = None
    stream_start = time.time()
    try:
        env = os.environ.copy()
        if PULSE_SERVER:
            env["PULSE_SERVER"] = PULSE_SERVER

        log.info("Streaming %d segments via ffmpeg/paplay...", len(chunks))

        ffmpeg_proc = subprocess.Popen(
            ["ffmpeg", "-f", "mpegts", "-i", "pipe:0",
             "-f", "s16le", "-ar", "44100", "-ac", "2", "pipe:1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        paplay_proc = subprocess.Popen(
            ["paplay", "--raw", "--format=s16le", "--rate=44100", "--channels=2"],
            stdin=ffmpeg_proc.stdout,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        with active_music_lock:
            active_music_processes[:] = [ffmpeg_proc, paplay_proc]

        for i, chunk in enumerate(chunks):
            current_chunk_idx = i

            if ffmpeg_proc.poll() is not None:
                raise RuntimeError(
                    f"ffmpeg died at segment {i+1} with code {ffmpeg_proc.returncode}"
                )
            if paplay_proc.poll() is not None:
                stderr = paplay_proc.stderr.read().decode("utf-8", errors="ignore")
                raise RuntimeError(
                    f"paplay died at segment {i+1} with code {paplay_proc.returncode}. stderr: {stderr}"
                )

            log.debug("Streaming segment %d/%d", i + 1, len(chunks))
            data = _decrypt_chunk(chunk)

            try:
                ffmpeg_proc.stdin.write(data)
                ffmpeg_proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                if getattr(e, "errno", None) == errno.EPIPE:
                    if ffmpeg_proc.poll() is not None:
                        # process was killed externally (e.g. stop_music) — exit cleanly
                        log.debug("ffmpeg was killed externally at segment %d, stopping", i + 1)
                        return
                    stderr = ffmpeg_proc.stderr.read().decode("utf-8", errors="ignore")
                    raise RuntimeError(
                        f"ffmpeg died at segment {i+1} (broken pipe). stderr: {stderr}"
                    ) from e
                raise

        ffmpeg_proc.stdin.close()
        log.info("Music streaming completed, waiting for processes to finish...")
        elapsed = time.time() - stream_start
        # Use sum of #EXTINF durations for accuracy; fall back to passed track duration
        extinf_total = sum(c.get("extinf", 0.0) for c in chunks)
        effective_duration = extinf_total if extinf_total > 0 else duration
        remaining_s = max(5.0, effective_duration - elapsed + 1.0) if effective_duration else 30.0
        _wait_or_kill(ffmpeg_proc, timeout=min(30.0, remaining_s + 5), label="ffmpeg (stream)")
        _wait_or_kill(paplay_proc, timeout=remaining_s, label="paplay (stream)")
        log.info("Processes finished")
    except Exception as e:
        log.exception("Streaming error: %s", e)
        try:
            if ffmpeg_proc is not None:
                ffmpeg_proc.kill()
            if paplay_proc is not None:
                paplay_proc.kill()
        except Exception:
            pass
    finally:
        with active_music_lock:
            active_music_processes.clear()


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_tracks(query: str) -> dict:
    """Search tracks by a free-form query (artist or title or both).

    Returns up to 10 results. Each result contains:
      - artist, title, duration (seconds), track_id, owner_id
      - access_key (if available; required for playback of some tracks)
      - album_title (album name, if available)
      - main_artists (list of primary artist names, if available)

    Args:
        query: Free-form query (artist and/or title). At least one value required.
    """
    q = (query or "").strip()
    if not q:
        return {"error": "No query specified"}

    log.info("search_tracks: query=%r", q)
    try:
        resp = _vk_call("audio.search", {
            "q": q,
            "count": 20,
            "offset": 0,
            "sort": 0,
            "autocomplete": 1,
        })
        items = resp.get("items", [])
        results = []
        for item in items[:10]:
            album_obj = item.get("album") or {}
            main_artists = [a["name"] for a in item.get("main_artists", []) if a.get("name")]
            entry = {
                "artist": item.get("artist"),
                "title": item.get("title"),
                "duration": item.get("duration"),
                "track_id": item.get("id"),
                "owner_id": item.get("owner_id"),
            }
            if item.get("access_key"):
                entry["access_key"] = item["access_key"]
            if album_obj.get("title"):
                entry["album_title"] = album_obj["title"]
            if main_artists:
                entry["main_artists"] = main_artists
            results.append(entry)
        return {"status": "ok", "results": results}
    except Exception as e:
        log.exception("search_tracks error: %s", e)
        return {"error": str(e)}


@mcp.tool()
def play_music(tracks: list) -> dict:
    """
    Play one or several music tracks (songs). If user asks to play tracks,
    you should search for the track, retrieve owner_id, track_id (and access_key if available)
    and pass them to play_music. Always pass access_key when search_tracks returned it.

    Args:
        tracks: required list of dicts with keys 'owner_id', 'track_id', and optionally 'access_key'.
                Example: [{"owner_id":652168798,"track_id":456239274,"access_key":"abc123"}, ...]
    """
    if not tracks:
        return {"error": "No tracks specified. Use search_tracks first and pass selected track ids to play_music."}

    try:
        # normalize and collect owner_track strings
        resolved = []
        for t in tracks:
            owner = t.get("owner_id") or t.get("owner")
            tid = t.get("track_id") or t.get("id")
            acc = t.get("access_key")
            if owner and tid:
                if acc:
                    resolved.append(f"{owner}_{tid}_{acc}")
                else:
                    resolved.append(f"{owner}_{tid}")

        if not resolved:
            return {"error": "No valid tracks specified"}

        # Fetch detailed info for all tracks in one call
        audios_param = ",".join(resolved)
        resp = _vk_call("audio.getById", {"audios": audios_param})
        items = resp.get("items", resp) if isinstance(resp, dict) else resp

        queued = []
        for item in items:
            url = item.get("url")
            if not url:
                # skip if VK didn't return url
                log.warning("No url for track %s_%s, skipping", item.get("owner_id"), item.get("id"))
                continue
            # minimal track object for queue
            track_obj = {"url": url, "artist": item.get("artist"), "title": item.get("title"), "duration": item.get("duration")}
            # validate url/playability (quick check)
            try:
                ok = _validate_track_item(item)
            except Exception:
                ok = False
            if not ok:
                log.warning("Track %s_%s not playable, skipping", item.get("owner_id"), item.get("id"))
                continue
            queued.append({"artist": item.get("artist"), "title": item.get("title"), "duration": item.get("duration"), "owner_id": item.get("owner_id"), "track_id": item.get("id")})
            _enqueue_tracks([track_obj], play_now=False)

        if not queued:
            return {"error": "No playable tracks found"}

        return {"status": "queued", "tracks": queued}
    except Exception as e:
        log.exception("play_music error: %s", e)
        return {"error": str(e)}


@mcp.tool()
def search_album(query: str) -> dict:
    """
    Search for an album (by album title and/or artist) and return its tracks.
    """
    q = (query or "").strip()
    if not q:
        return {"error": "No query specified"}
    try:
        resp = _vk_call("audio.searchAlbums", {"q": q, "count": 5, "offset": 0})
        albums = resp.get("items", [])
        results = []
        for album in albums:
            owner = album.get("owner_id")
            album_id = album.get("id") or album.get("playlist_id")
            title = album.get("title")
            if not owner or not album_id:
                continue
            # fetch tracks in album
            try:
                tracks_resp = _vk_call("audio.get", {"owner_id": owner, "album_id": album_id, "count": 500})
                tracks = tracks_resp.get("items", [])
            except Exception as e:
                tracks = []
                log.warning("Failed to fetch album tracks %s/%s: %s", owner, album_id, e)
            results.append({
                "album_title": title,
                "owner_id": owner,
                "album_id": album_id,
                "access_key": album.get("access_key", ""),
                "tracks": [
                    {
                        "artist": t.get("artist"),
                        "title": t.get("title"),
                        "duration": t.get("duration"),
                        "track_id": t.get("id"),
                        "owner_id": t.get("owner_id"),
                        # 'url': t.get("url"),  # commented out: URL is retrieved only by play_music
                    }
                    for t in tracks
                ],
            })
        return {"status": "ok", "albums": results}
    except Exception as e:
        log.exception("search_album error: %s", e)
        return {"error": str(e)}


@mcp.tool()
def stop_music() -> dict:
    """Stop all currently playing music and clear the playback queue.

    Returns:
        Dict with status and the number of processes that were stopped.
    """
    global pause_state, playback_paused

    with playback_lock:
        playback_queue.clear()
        playback_paused = False
    pause_state = None

    with active_music_lock:
        procs = list(active_music_processes)
        active_music_processes.clear()

    stopped = 0
    for proc in procs:
        try:
            proc.kill()
            stopped += 1
        except Exception:
            pass

    log.info("Stopped %d music processes, queue cleared", stopped)
    return {"status": "stopped", "count": stopped}


@mcp.tool()
def music_health() -> dict:
    """Return health status of the music service.

    Returns:
        Dict with status, whether VK token is set, and current playback mode.
    """
    _reload_token_if_changed()
    result = {
        "status": "ok",
        "vk_token_set": bool(VK_TOKEN),
        "mode": MODE,
    }
    if MODE == "file":
        result["temp_dir"] = TEMP_DIR
    return result


@mcp.tool()
def pause_music() -> dict:
    """Pause the currently playing music. Saves position so playback can be resumed."""
    global pause_state, playback_paused

    with playback_lock:
        if playback_paused:
            return {"status": "already_paused"}
        # Snapshot current state before setting the flag
        snap_track = current_track_info
        snap_chunks = list(current_chunks)
        snap_idx = current_chunk_idx
        playback_paused = True

    if not snap_track and not snap_chunks:
        with playback_lock:
            playback_paused = False
        return {"error": "Nothing is playing"}

    pause_state = {
        "track":        snap_track,
        "chunks":       snap_chunks,
        "chunk_index":  snap_idx,
        "total_chunks": len(snap_chunks),
        "url":          current_track_url,
    }

    with active_music_lock:
        procs = list(active_music_processes)
    for proc in procs:
        try:
            proc.kill()
        except Exception:
            pass

    info = f"{snap_track.get('artist')} — {snap_track.get('title')}" if snap_track else "unknown"
    log.info("Music paused at chunk %d: %s", snap_idx, info)
    return {"status": "paused", "track": snap_track, "chunk_index": snap_idx}


@mcp.tool()
def resume_music() -> dict:
    """Resume playback from the point where it was paused."""
    global pause_state, playback_paused

    if not pause_state:
        return {"error": "No paused track to resume"}

    saved = pause_state
    pause_state = None

    with playback_lock:
        playback_paused = False

    idx           = saved.get("chunk_index", 0)
    hls_url       = saved.get("url", "")
    track_duration = float((saved.get("track") or {}).get("duration") or 0)

    # Re-fetch chunks for fresh URLs; also needed for accurate remaining-duration calc
    all_chunks = saved.get("chunks", [])
    if hls_url:
        try:
            all_chunks = _get_track_chunks({"url": hls_url})
            log.debug("Re-fetched %d chunks for resume", len(all_chunks))
        except Exception as e:
            log.warning("Failed to re-fetch chunks on resume, using cached: %s", e)

    remaining = all_chunks[idx:]
    if not remaining:
        return {"error": "No remaining audio data to resume"}

    # Exact remaining duration from #EXTINF tags; fall back to proportional estimate
    remaining_duration = sum(c.get("extinf", 0.0) for c in remaining)
    if not remaining_duration:
        total = len(all_chunks)
        remaining_duration = (track_duration * len(remaining) / total) if total else track_duration

    track = dict(saved.get("track") or {})
    track["_resume_chunks"] = remaining
    track["_duration"]      = remaining_duration
    track["url"]            = hls_url  # needed by _playback_loop for current_track_url

    _enqueue_tracks([track], play_now=True)

    info = f"{track.get('artist')} — {track.get('title')}"
    log.info("Resuming music from chunk %d/%d (%.0fs remaining): %s",
             idx, len(all_chunks), remaining_duration, info)
    return {
        "status":        "resumed",
        "track":         {"artist": track.get("artist"), "title": track.get("title")},
        "from_chunk":    idx,
        "remaining_sec": int(remaining_duration),
    }


@mcp.tool()
def volume_up() -> dict:
    """Increase the system audio volume by 10%."""
    env = os.environ.copy()
    if PULSE_SERVER:
        env["PULSE_SERVER"] = PULSE_SERVER
    try:
        result = subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"],
            env=env, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip() or "pactl failed"}
        log.info("Volume increased by 10%%")
        return {"status": "ok", "action": "volume_up"}
    except Exception as e:
        log.exception("volume_up error: %s", e)
        return {"error": str(e)}


@mcp.tool()
def volume_down() -> dict:
    """Decrease the system audio volume by 10%."""
    env = os.environ.copy()
    if PULSE_SERVER:
        env["PULSE_SERVER"] = PULSE_SERVER
    try:
        result = subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"],
            env=env, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip() or "pactl failed"}
        log.info("Volume decreased by 10%%")
        return {"status": "ok", "action": "volume_down"}
    except Exception as e:
        log.exception("volume_down error: %s", e)
        return {"error": str(e)}


@mcp.tool()
def get_queue() -> dict:
    """Return the current playback queue and the currently playing track."""
    with playback_lock:
        queue_copy = list(playback_queue)

    tracks = [
        {
            "artist": t.get("artist"),
            "title": t.get("title"),
            "duration": t.get("duration"),
        }
        for t in queue_copy
        if "_resume_chunks" not in t
    ]

    result: dict = {
        "status": "ok",
        "currently_playing": current_track_info,
        "paused": playback_paused,
        "queue_count": len(tracks),
        "queue": tracks,
    }
    if pause_state:
        result["paused_track"] = pause_state.get("track")
    return result


@mcp.tool()
def skip_tracks(count: int = 1) -> dict:
    """Skip one or more tracks.

    Use for prompts like:
      'next track' / 'skip' → count=1
      'skip two songs' → count=2
      'skip three tracks' → count=3  (and so on)

    Skips the currently playing track plus the next (count-1) queued tracks.

    Args:
        count: Total number of tracks to skip (including the current one). Default is 1.
    """
    if count < 1:
        return {"error": "count must be at least 1"}

    with playback_lock:
        to_remove = min(count - 1, len(playback_queue))
        for _ in range(to_remove):
            playback_queue.pop(0)

    with active_music_lock:
        procs = list(active_music_processes)
    for proc in procs:
        try:
            proc.kill()
        except Exception:
            pass

    log.info("Skipping %d track(s)", count)
    return {"status": "skipped", "count": count}


@mcp.tool()
def whoami() -> dict:
    """Return information about the VK account whose token is currently in use.

    Returns:
        Dict with id, first_name, last_name, and profile photo URL.
    """
    _reload_token_if_changed()
    if not VK_TOKEN:
        return {"error": "VK_TOKEN is not set"}
    user = _fetch_vk_user()
    if not user:
        return {"error": "Failed to retrieve user info from VK"}
    return {
        "status": "ok",
        "id": user.get("id"),
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "photo": user.get("photo_50"),
    }


@mcp.tool()
def find_user(query: str = "") -> dict:
    """Find a VK user by name/surname and return their ID and basic info.

    Friends of the token owner appear first in results, followed by global search results.
    If query is empty, returns info about the account whose token is in use.

    Args:
        query: Name or surname to search (e.g. "Ivan Petrov"). Leave empty to get own info.
    """
    q = (query or "").strip()
    if not q:
        user = _fetch_vk_user()
        if not user:
            return {"error": "VK_TOKEN not set or failed to fetch own user info"}
        return {
            "status": "ok",
            "users": [{
                "id": user.get("id"),
                "first_name": user.get("first_name"),
                "last_name": user.get("last_name"),
            }],
        }

    def _fmt(u: dict) -> dict:
        return {
            "id": u.get("id"),
            "first_name": u.get("first_name"),
            "last_name": u.get("last_name"),
        }

    try:
        results: list[dict] = []
        seen_ids: set = set()

        # Friends first
        try:
            fr_resp = _vk_call("friends.search", {"q": q, "count": 10, "fields": ""})
            for u in fr_resp.get("items", []):
                uid = u.get("id")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    results.append({**_fmt(u), "is_friend": True})
        except Exception as e:
            log.debug("friends.search failed (non-critical): %s", e)

        # Global search — append only those not already in friends results
        gl_resp = _vk_call("users.search", {"q": q, "count": 10})
        for u in gl_resp.get("items", []):
            uid = u.get("id")
            if uid and uid not in seen_ids:
                seen_ids.add(uid)
                results.append({**_fmt(u), "is_friend": False})

        return {"status": "ok", "users": results}
    except Exception as e:
        log.exception("find_user error: %s", e)
        return {"error": str(e)}


@mcp.tool()
def play_user_audio(owner_id: int = 0, shuffle: bool = False, count: int = 1000) -> dict:
    """Fetch and queue audio tracks from a VK user's profile.

    Tracks are fetched and queued directly — the model never sees individual track data,
    making this safe even for very large libraries.

    VK API has no server-side shuffle; when shuffle=True the order is randomised locally
    before queuing.

    Args:
        owner_id: VK user ID whose audio to play. Pass 0 (default) to play own music.
        shuffle: Randomise track order before queuing (default False).
        count: Maximum number of tracks to fetch (default 1000).
    """
    _reload_token_if_changed()
    if not VK_TOKEN:
        return {"error": "VK_TOKEN is not set"}

    params: dict = {"count": count, "extended": 0}
    display_owner = owner_id

    if owner_id:
        params["owner_id"] = owner_id
    else:
        user = _fetch_vk_user()
        if user:
            display_owner = user.get("id", 0)

    try:
        resp = _vk_call("audio.get", params)
        items = resp.get("items", [])

        track_objs = [
            {
                "url": item.get("url"),
                "artist": item.get("artist"),
                "title": item.get("title"),
                "duration": item.get("duration"),
            }
            for item in items
            if item.get("url")
        ]

        if not track_objs:
            return {"error": "No playable tracks found for this user"}

        if shuffle:
            import random
            random.shuffle(track_objs)

        _enqueue_tracks(track_objs)

        log.info("Queued %d tracks for user %s (shuffle=%s)", len(track_objs), display_owner, shuffle)
        return {
            "status": "queued",
            "owner_id": display_owner,
            "count": len(track_objs),
            "shuffled": shuffle,
        }
    except Exception as e:
        log.exception("play_user_audio error: %s", e)
        return {"error": str(e)}


@mcp.tool()
def play_album(owner_id: int, album_id: int, access_key: str = "", shuffle: bool = False) -> dict:
    """Fetch and queue all tracks from a VK album or playlist.

    Tracks are fetched and queued directly — safe for large albums.
    Use search_album first to obtain owner_id, album_id and access_key.

    VK API has no server-side shuffle; when shuffle=True the order is randomised locally.

    Args:
        owner_id: Album owner's VK user or community ID.
        album_id: Album (playlist) ID.
        access_key: Album access key returned by search_album (required for many albums).
        shuffle: Randomise track order before queuing (default False).
    """
    try:
        params: dict = {
            "owner_id": owner_id,
            "album_id": album_id,
            "count": 1000,
            "extended": 0,
        }
        if access_key:
            params["access_key"] = access_key

        resp = _vk_call("audio.get", params)
        items = resp.get("items", [])

        track_objs = [
            {
                "url": item.get("url"),
                "artist": item.get("artist"),
                "title": item.get("title"),
                "duration": item.get("duration"),
            }
            for item in items
            if item.get("url")
        ]

        if not track_objs:
            return {"error": "No playable tracks found in this album"}

        if shuffle:
            import random
            random.shuffle(track_objs)

        _enqueue_tracks(track_objs)

        log.info(
            "Queued %d tracks from album %s/%s (shuffle=%s)",
            len(track_objs), owner_id, album_id, shuffle,
        )
        return {
            "status": "queued",
            "owner_id": owner_id,
            "album_id": album_id,
            "count": len(track_objs),
            "shuffled": shuffle,
        }
    except Exception as e:
        log.exception("play_album error: %s", e)
        return {"error": str(e)}


if __name__ == "__main__":
    log.info("Starting Music MCP server (stdio), mode=%s", MODE)
    mcp.run(transport="stdio")
