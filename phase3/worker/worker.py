#!/usr/bin/env python3
"""
Phase 3 worker — download Ballchasing .replay → parse → compact JSON → upload → delete local file.

Zadržava shared-hosting upload (User-Agent + retry) iz radne verzije repoa,
plus pouzdaniji import sprocket-boxcars-py i REPROCESS_EMPTY za mečeve bez heatmap-a.

Env (GitHub Secrets / local .env):
  BALLCHASING_TOKEN   required
  HOST_BASE_URL       e.g. https://tvoj-sajt.com  (no trailing slash)
  UPLOAD_TOKEN        shared secret for upload/pending API
  BATCH_SIZE          default 180
  GRID_SIZE           heatmap grid default 24
  REPROCESS_EMPTY     1/true = ponovo obradi JSON-ove bez frame heatmap-a
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ── boxcars / sprocket-boxcars-py ──────────────────────────────────────────
# Package name on PyPI: sprocket-boxcars-py
# Import module name:   boxcars_py  (parse_replay)
boxcars_parse = None
HAS_BOXCARS = False
_BOXCARS_ERR = None
try:
    from boxcars_py import parse_replay as boxcars_parse  # type: ignore
    HAS_BOXCARS = True
except Exception as e1:
    _BOXCARS_ERR = str(e1)
    try:
        import boxcars_py as _bp  # type: ignore

        boxcars_parse = getattr(_bp, "parse_replay", None) or getattr(_bp, "parse", None)
        HAS_BOXCARS = callable(boxcars_parse)
        if not HAS_BOXCARS:
            _BOXCARS_ERR = (
                f"{e1} | boxcars_py loaded but no parse_replay; "
                f"attrs={[x for x in dir(_bp) if not x.startswith('_')][:20]}"
            )
    except Exception as e2:
        _BOXCARS_ERR = f"{e1} | {e2}"

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    np = None  # type: ignore

API = "https://ballchasing.com/api"
TMP = Path(os.environ.get("TMPDIR", "/tmp")) / "rl_phase3"
TMP.mkdir(parents=True, exist_ok=True)

TOKEN = os.environ.get("BALLCHASING_TOKEN", "").strip()
HOST = os.environ.get("HOST_BASE_URL", "").strip().rstrip("/")
UPLOAD_TOKEN = os.environ.get("UPLOAD_TOKEN", "").strip()
BATCH_SIZE = max(1, min(200, int(os.environ.get("BATCH_SIZE", "180"))))
GRID = max(12, min(48, int(os.environ.get("GRID_SIZE", "24"))))
REPROCESS_EMPTY = os.environ.get("REPROCESS_EMPTY", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def bc_headers() -> dict:
    return {"Authorization": TOKEN}


def bc_get(path: str, timeout: int = 60) -> requests.Response:
    url = path if path.startswith("http") else API + path
    return requests.get(url, headers=bc_headers(), timeout=timeout)


# Shared-hosting friendly headers (mod_security / bot filters)
HOST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}


def host_get(path: str, extra_params: dict | None = None, retries: int = 3) -> Any:
    url = f"{HOST}{path}"
    params: dict[str, Any] = {"token": UPLOAD_TOKEN}
    if extra_params:
        params.update(extra_params)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=60, headers=HOST_HEADERS)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            last_err = e
            log(f"  host_get attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    raise last_err  # type: ignore[misc]


def host_post(path: str, payload: dict, retries: int = 3) -> Any:
    url = f"{HOST}{path}"
    headers = {**HOST_HEADERS, "Content-Type": "application/json"}
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                url,
                params={"token": UPLOAD_TOKEN},
                json=payload,
                timeout=120,
                headers=headers,
            )
            if r.status_code >= 400:
                raise RuntimeError(f"Host POST {path} → {r.status_code}: {r.text[:400]}")
            return r.json()
        except (requests.exceptions.RequestException, RuntimeError) as e:
            last_err = e
            log(f"  host_post attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    raise last_err  # type: ignore[misc]


def download_replay(replay_id: str) -> Path:
    """Download .replay binary. Respect Ballchasing file rate limits."""
    out = TMP / f"{replay_id}.replay"
    if out.exists() and out.stat().st_size > 1000:
        return out
    r = bc_get(f"/replays/{replay_id}/file", timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"download {replay_id}: HTTP {r.status_code} {r.text[:200]}")
    out.write_bytes(r.content)
    log(f"  downloaded {replay_id} ({len(r.content)} bytes)")
    time.sleep(1.05)  # free tier ~1 req/s on /file
    return out


def fetch_bc_details(replay_id: str) -> dict:
    r = bc_get(f"/replays/{replay_id}", timeout=60)
    if r.status_code != 200:
        log(f"  warn: details {replay_id} HTTP {r.status_code}")
        return {}
    time.sleep(0.4)
    return r.json()


def _team_players(details: dict) -> list[dict]:
    out = []
    for color in ("blue", "orange"):
        side = details.get(color) or {}
        for p in side.get("players") or []:
            name = p.get("name") or p.get("id", {}).get("id") or "unknown"
            out.append({"name": name, "team": color, "raw": p})
    return out


def build_heatmap_empty() -> dict:
    n = GRID * GRID
    return {"cols": GRID, "rows": GRID, "cells": [0] * n, "max": 0}


def extract_heatmap_from_boxcars(
    raw: bytes, player_names: list[str]
) -> tuple[dict[str, dict], int]:
    """Best-effort position sampling → per-player heatmap. Returns (heatmaps, frame_count)."""
    heatmaps = {n: build_heatmap_empty() for n in player_names}
    frame_count = 0
    if not HAS_BOXCARS or not HAS_NUMPY:
        return heatmaps, 0

    try:
        parsed = boxcars_parse(raw)
    except Exception as e:
        log(f"  boxcars parse failed: {e}")
        return heatmaps, 0

    log(f"  parse type={type(parsed).__name__}")

    frames = None
    if isinstance(parsed, dict):
        frames = parsed.get("network_frames") or parsed.get("frames")
        if isinstance(frames, dict):
            frames = frames.get("frames") or frames.get("network_frames")
    if frames is None:
        frames = getattr(parsed, "network_frames", None) or getattr(parsed, "frames", None)

    if not frames:
        keys = list(parsed.keys())[:30] if isinstance(parsed, dict) else dir(parsed)[:30]
        log(f"  no network frames — keys/attrs sample: {keys}")
        return heatmaps, 0

    grids = {n: np.zeros((GRID, GRID), dtype=np.int32) for n in player_names}
    name_lower = {n.lower(): n for n in player_names}

    def accumulate(name: str, x: float, y: float) -> None:
        nx = (float(x) + 4000) / 8000
        ny = (float(y) + 5000) / 10000
        nx = max(0.0, min(0.999, nx))
        ny = max(0.0, min(0.999, ny))
        c = int(nx * GRID)
        r = int(ny * GRID)
        grids[name][r, c] += 1

    try:
        for fr in frames:
            frame_count += 1
            if frame_count % 4 != 0:
                continue
            actors = None
            if isinstance(fr, dict):
                actors = (
                    fr.get("actors")
                    or fr.get("new_actors")
                    or fr.get("updated_actors")
                )
            if not actors:
                continue
            if isinstance(actors, dict):
                actors = actors.values()
            for act in actors:
                if not isinstance(act, dict):
                    continue
                aname = act.get("name") or act.get("player_name") or ""
                key = str(aname).lower()
                if key not in name_lower:
                    continue
                pos = act.get("position") or act.get("pos") or act.get("RigidBody")
                if isinstance(pos, dict):
                    x = pos.get("x") or pos.get("X")
                    y = pos.get("y") or pos.get("Y")
                elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    x, y = pos[0], pos[1]
                else:
                    x = act.get("x")
                    y = act.get("y")
                if x is None or y is None:
                    continue
                accumulate(name_lower[key], x, y)
    except Exception as e:
        log(f"  frame walk error: {e}")

    for n, g in grids.items():
        flat = g.flatten().tolist()
        mx = int(g.max()) if g.size else 0
        heatmaps[n] = {"cols": GRID, "rows": GRID, "cells": flat, "max": mx}

    if frame_count:
        log(f"  frames sampled path: {frame_count}, heat max={[heatmaps[n]['max'] for n in player_names]}")
    return heatmaps, frame_count


def build_advanced(
    replay_id: str,
    details: dict,
    heatmaps: dict,
    frame_count: int,
    frames_ok: bool,
) -> dict:
    players_out = []
    timeline = []
    mistakes = []

    duration = float(details.get("duration") or 0)
    map_name = details.get("map_name") or details.get("map_code") or ""
    playlist = details.get("playlist_id") or details.get("playlist") or ""

    for color in ("blue", "orange"):
        side = details.get(color) or {}
        for p in side.get("players") or []:
            name = p.get("name") or "unknown"
            stats = p.get("stats") or {}
            core = stats.get("core") or {}
            boost = stats.get("boost") or {}
            pos = stats.get("positioning") or {}

            hm = heatmaps.get(name) or build_heatmap_empty()

            goals_n = int(core.get("goals") or 0)
            saves_n = int(core.get("saves") or 0)
            events = {
                "goals": [{"t": None, "n": goals_n}] if goals_n else [],
                "assists": [],
                "saves": [{"t": None, "n": saves_n}] if saves_n else [],
                "demos_inflicted": [],
                "demos_taken": [],
            }

            demo = stats.get("demo") or {}
            if int(demo.get("inflicted") or 0):
                events["demos_inflicted"] = [{"t": None, "n": int(demo["inflicted"])}]
            if int(demo.get("taken") or 0):
                events["demos_taken"] = [{"t": None, "n": int(demo["taken"])}]

            behind = float(pos.get("percent_behind_ball") or 0)
            zero_b = float(boost.get("percent_zero_boost") or 0)
            gal = float(pos.get("goals_against_while_last_defender") or 0)

            flags = {
                "low_behind_ball": behind > 0 and behind < 60,
                "high_zero_boost": zero_b > 20,
                "goals_against_last": gal >= 1,
            }

            if flags["low_behind_ball"]:
                mistakes.append(
                    {
                        "severity": "major",
                        "type": "low_behind_ball",
                        "player": name,
                        "detail": f"Behind Ball {behind:.1f}%",
                        "t": None,
                    }
                )
            if flags["high_zero_boost"]:
                mistakes.append(
                    {
                        "severity": "major",
                        "type": "high_zero_boost",
                        "player": name,
                        "detail": f"Time at 0 boost {zero_b:.1f}%",
                        "t": None,
                    }
                )
            if flags["goals_against_last"]:
                mistakes.append(
                    {
                        "severity": "critical",
                        "type": "goals_against_last_defender",
                        "player": name,
                        "detail": f"GA while last defender: {gal}",
                        "t": None,
                    }
                )

            players_out.append(
                {
                    "name": name,
                    "team": color,
                    "heatmap": hm,
                    "stats_snapshot": {
                        "goals": goals_n,
                        "assists": int(core.get("assists") or 0),
                        "saves": saves_n,
                        "score": int(core.get("score") or 0),
                        "behind_ball": behind,
                        "zero_boost": zero_b,
                    },
                    "events": events,
                    "flags": flags,
                }
            )

    for g in details.get("goals") or []:
        timeline.append(
            {
                "t": g.get("frame_number") or g.get("time") or None,
                "type": "goal",
                "player": (
                    (g.get("player") or {}).get("name")
                    if isinstance(g.get("player"), dict)
                    else g.get("player")
                ),
                "team": None,
            }
        )

    return {
        "v": 1,
        "replay_id": replay_id,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "duration": duration,
        "map": map_name,
        "playlist": playlist,
        "players": players_out,
        "timeline": timeline,
        "mistakes": mistakes,
        "source": {
            "ballchasing": bool(details),
            "frames_parsed": frames_ok and frame_count > 0,
            "frame_count": frame_count,
            "boxcars": HAS_BOXCARS,
        },
    }


def process_one(replay_id: str) -> dict:
    log(f"Processing {replay_id}")
    details = fetch_bc_details(replay_id)
    names = [p["name"] for p in _team_players(details)]

    path = download_replay(replay_id)
    raw = path.read_bytes()
    heatmaps, frame_count = extract_heatmap_from_boxcars(raw, names)
    frames_ok = frame_count > 0

    advanced = build_advanced(replay_id, details, heatmaps, frame_count, frames_ok)

    try:
        path.unlink(missing_ok=True)
        log(f"  deleted local {path.name}")
    except Exception as e:
        log(f"  warn delete: {e}")

    return advanced


def main() -> int:
    if not TOKEN:
        log("ERROR: BALLCHASING_TOKEN missing")
        return 1
    if not HOST or not UPLOAD_TOKEN:
        log("ERROR: HOST_BASE_URL and UPLOAD_TOKEN required")
        return 1

    log(
        f"boxcars available: {HAS_BOXCARS}, numpy: {HAS_NUMPY}"
        + (f" (err: {_BOXCARS_ERR})" if not HAS_BOXCARS and _BOXCARS_ERR else "")
    )
    log(f"Host: {HOST}, batch: {BATCH_SIZE}, reprocess_empty: {REPROCESS_EMPTY}")

    extra = {}
    if REPROCESS_EMPTY:
        extra["reprocess_empty"] = "1"

    pending = host_get("/api/advanced_pending.php", extra_params=extra or None)
    if not pending.get("ok"):
        log(f"ERROR pending: {pending}")
        return 1

    ids = pending.get("replay_ids") or []
    log(f"Pending: {len(ids)} (processing up to {BATCH_SIZE})")
    ids = ids[:BATCH_SIZE]
    if not ids:
        log("Nothing to do.")
        return 0

    ok_n = fail_n = 0
    for rid in ids:
        try:
            adv = process_one(rid)
            resp = host_post("/api/upload_advanced.php", adv)
            if resp.get("ok"):
                ok_n += 1
                heat_ok = any(
                    (p.get("heatmap") or {}).get("max", 0) > 0
                    for p in (adv.get("players") or [])
                )
                log(f"  uploaded {rid} (heatmap={'yes' if heat_ok else 'no'})")
            else:
                fail_n += 1
                log(f"  upload failed {rid}: {resp}")
        except Exception as e:
            fail_n += 1
            log(f"  FAIL {rid}: {e}")
            traceback.print_exc()
            time.sleep(2)

    log(f"Done. ok={ok_n} fail={fail_n}")
    for f in TMP.glob("*.replay"):
        try:
            f.unlink()
        except Exception:
            pass
    return 0 if fail_n == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
