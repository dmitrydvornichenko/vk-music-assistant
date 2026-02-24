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


if not VK_TOKEN:
    log.error("VK_TOKEN not set! Music service won't work.")

mcp = FastMCP("music-server")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def _get_track_chunks(track):
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
            chunks.append({"file_url": file_url, "key_method": key_method, "key_url": key_url, "file": file})

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

    log.debug("Found %d segments", len(chunks))
    return chunks


def _decrypt_chunk(chunk) -> bytes:
    if chunk["key"]:
        from urllib.request import urlopen
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad
        key = chunk["key"]
        file_in = urlopen(chunk["file_url"])
        iv = file_in.read(16)
        ciphered_data = file_in.read()
        file_in.close()
        cipher = AES.new(key, AES.MODE_CBC, iv=iv)
        return unpad(cipher.decrypt(ciphered_data), AES.block_size)
    import requests as req
    return req.get(chunk["file_url"]).content


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

    base = f"https://api.vk.com/method/{method}"
    # Ensure token and version
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
    global playback_running
    try:
        while True:
            with playback_lock:
                if not playback_queue:
                    playback_running = False
                    return
                track = playback_queue.pop(0)

            try:
                chunks = _get_track_chunks(track)
            except Exception as e:
                log.warning("Failed to get chunks for queued track: %s", e)
                continue

            if not chunks:
                log.warning("No segments for queued track, skipping")
                continue

            try:
                if MODE == "file":
                    _file_music_thread(chunks)
                else:
                    _stream_music_thread(chunks)
            except Exception as e:
                log.exception("Error while playing queued track: %s", e)
                continue
    finally:
        playback_running = False


def _file_music_thread(chunks: list):
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

        ffmpeg_dec.wait()
        paplay_proc.wait()
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


def _stream_music_thread(chunks: list):
    ffmpeg_proc = None
    paplay_proc = None
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
        ffmpeg_proc.wait()
        paplay_proc.wait()
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
    Play one or several music tracks (songs or albums). If user asks to play a track or tracks or album,
    you should search for the track or album, retrieve owner_id, track_id (and access_key if available)
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
    """Stop all currently playing music.

    Returns:
        Dict with status and the number of processes that were stopped.
    """
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

    log.info("Stopped %d music processes", stopped)
    return {"status": "stopped", "count": stopped}


@mcp.tool()
def music_health() -> dict:
    """Return health status of the music service.

    Returns:
        Dict with status, whether VK token is set, and current playback mode.
    """
    result = {
        "status": "ok",
        "vk_token_set": bool(VK_TOKEN),
        "mode": MODE,
    }
    if MODE == "file":
        result["temp_dir"] = TEMP_DIR
    return result


if __name__ == "__main__":
    log.info("Starting Music MCP server (stdio), mode=%s", MODE)
    mcp.run(transport="stdio")
