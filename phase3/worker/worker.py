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

import json
import unicodedata
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ── Parser backends ────────────────────────────────────────────────────────
# 1) rrrocket CLI (preferred) — tracks upstream boxcars, supports new RL attrs
#    e.g. TAGame.Car_TA:DodgesRefreshedCounter (v0.10.11+)
# 2) sprocket-boxcars-py fallback (often outdated vs live RL patches)
boxcars_parse = None
HAS_BOXCARS = False
HAS_RRROCKET = False
RRROCKET_BIN: str | None = None
_BOXCARS_ERR = None
_IMPORT_ERRORS: list[str] = []

def _find_rrrocket() -> str | None:
    env = os.environ.get("RRROCKET_BIN", "").strip()
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    which = shutil.which("rrrocket")
    if which:
        return which
    for cand in (
        Path("/usr/local/bin/rrrocket"),
        Path.cwd() / "rrrocket",
        Path(__file__).resolve().parent / "rrrocket",
        Path("/opt/rrrocket/rrrocket"),
    ):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None

RRROCKET_BIN = _find_rrrocket()
if RRROCKET_BIN:
    HAS_RRROCKET = True
    HAS_BOXCARS = True  # treat as available parser for heatmap path

for _mod_name in ("sprocket_boxcars_py", "boxcars_py", "boxcars"):
    try:
        _m = __import__(_mod_name)
        _fn = getattr(_m, "parse_replay", None) or getattr(_m, "parse", None)
        if callable(_fn):
            boxcars_parse = _fn
            HAS_BOXCARS = True
            break
        _pub = [x for x in dir(_m) if not x.startswith("_")]
        _IMPORT_ERRORS.append(f"{_mod_name} loaded but no parse_replay; attrs={_pub[:20]}")
    except Exception as _e:
        _IMPORT_ERRORS.append(f"{_mod_name}: {_e}")
if not HAS_BOXCARS:
    _BOXCARS_ERR = " | ".join(_IMPORT_ERRORS) if _IMPORT_ERRORS else "no candidate module"


def parse_replay_dict(raw: bytes, replay_path: Path | None = None) -> dict:
    """Parse .replay → dict. Prefer rrrocket (up-to-date), else in-process boxcars."""
    errors: list[str] = []

    if HAS_RRROCKET and RRROCKET_BIN:
        path: Path | None = Path(replay_path) if replay_path else None
        tmp_created = False
        out_json: Path | None = None
        try:
            if path is None or not path.exists():
                path = TMP / f"_parse_{os.getpid()}_{int(time.time() * 1000)}.replay"
                path.write_bytes(raw)
                tmp_created = True
            out_json = path.with_suffix(".rrrocket.json")
            # Prefer file redirect — more reliable than capturing multi-MB stdout
            with open(out_json, "wb") as fout:
                proc = subprocess.run(
                    [RRROCKET_BIN, "-n", str(path)],
                    stdout=fout,
                    stderr=subprocess.PIPE,
                    timeout=180,
                    check=False,
                )
            if proc.returncode != 0:
                err = (proc.stderr or b"").decode("utf-8", "replace")[:600]
                errors.append(f"rrrocket exit {proc.returncode}: {err}")
            elif not out_json.exists() or out_json.stat().st_size < 10:
                errors.append("rrrocket produced empty JSON")
            else:
                with open(out_json, "r", encoding="utf-8") as fin:
                    data = json.load(fin)
                if isinstance(data, dict):
                    log(f"  rrrocket OK ({out_json.stat().st_size} bytes JSON)")
                    return data
                errors.append("rrrocket returned non-object JSON")
        except Exception as e:
            errors.append(f"rrrocket: {e}")
        finally:
            if out_json is not None:
                try:
                    out_json.unlink(missing_ok=True)
                except Exception:
                    pass
            if tmp_created and path is not None:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

    if callable(boxcars_parse):
        try:
            parsed = boxcars_parse(raw)
            if isinstance(parsed, dict):
                log("  boxcars_py OK (fallback)")
                return parsed
            errors.append(f"boxcars returned non-dict: {type(parsed)}")
        except Exception as e:
            errors.append(f"boxcars: {e}")

    raise RuntimeError(
        "No usable replay parser — " + (" | ".join(errors) if errors else "none installed")
    )

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


def _as_dict(obj: Any) -> Any:
    """Normalize boxcars objects that may arrive as dicts (serde JSON) or plain values."""
    return obj


def _obj_id(val: Any) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, dict):
        for k in ("value", "ObjectId", "object_id", "id"):
            if k in val and isinstance(val[k], int):
                return val[k]
    return None


def _actor_id(val: Any) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, dict):
        for k in ("value", "ActorId", "actor_id", "id"):
            if k in val and isinstance(val[k], int):
                return val[k]
    return None


def _xy_from_location(loc: Any) -> tuple[float, float] | None:
    if loc is None:
        return None
    if isinstance(loc, dict):
        x = loc.get("x", loc.get("X"))
        y = loc.get("y", loc.get("Y"))
        if x is not None and y is not None:
            return float(x), float(y)
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return None


def _xy_from_attribute(attr: Any) -> tuple[float, float] | None:
    """Extract x,y from a boxcars Attribute (RigidBody / Location / nested)."""
    if attr is None:
        return None
    if not isinstance(attr, dict):
        return None
    # serde enum: {"RigidBody": {...}} or {"Location": {...}}
    if "RigidBody" in attr:
        rb = attr["RigidBody"]
        if isinstance(rb, dict):
            return _xy_from_location(rb.get("location") or rb.get("Location"))
    if "Location" in attr:
        return _xy_from_location(attr["Location"])
    # already unwrapped rigid body
    if "location" in attr or "Location" in attr:
        return _xy_from_location(attr.get("location") or attr.get("Location"))
    if "x" in attr or "X" in attr:
        return _xy_from_location(attr)
    return None


def _string_from_attribute(attr: Any) -> str | None:
    if attr is None:
        return None
    if isinstance(attr, str):
        return attr
    if isinstance(attr, dict):
        if "String" in attr:
            s = attr["String"]
            return str(s) if s is not None else None
        if "string" in attr:
            s = attr["string"]
            return str(s) if s is not None else None
    return None


def _active_actor_id(attr: Any) -> int | None:
    """Engine.Pawn:PlayerReplicationInfo style ActiveActor → target actor id."""
    if not isinstance(attr, dict):
        return None
    node = attr.get("ActiveActor") if "ActiveActor" in attr else attr
    if not isinstance(node, dict):
        return None
    # common shapes: {active: true, actor: 12} or {actor: {value: 12}}
    for k in ("actor", "Actor", "actor_id", "ActorId"):
        if k in node:
            return _actor_id(node[k])
    return None



def _fold_name(s: str) -> str:
    """Lowercase + strip diacritics so 'Činčila' matches 'Cincila' / mangled 'inila'."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def extract_heatmap_from_boxcars(
    raw: bytes, player_names: list[str], replay_path: Path | None = None
) -> tuple[dict[str, dict], int]:
    """
    Boxcars network body structure (serde JSON):
      network_frames.frames[] → {
        new_actors: [{actor_id, object_id, name_id, initial_trajectory}],
        updated_actors: [{actor_id, object_id, stream_id, attribute: {RigidBody|String|ActiveActor|...}}],
        deleted_actors: [actor_id]
      }
      objects[] → class/archetype names (index = object_id)
      names[]   → optional name table

    Cars don't carry player names. Linkage:
      PRI actor  ← PlayerName String attribute
      Car actor  ← PlayerReplicationInfo ActiveActor → PRI actor
      Car actor  ← RigidBody.location for heatmap samples
    """
    heatmaps = {n: build_heatmap_empty() for n in player_names}
    frame_count = 0
    if (not HAS_BOXCARS and not HAS_RRROCKET) or not HAS_NUMPY:
        return heatmaps, 0

    try:
        parsed = parse_replay_dict(raw, replay_path=replay_path)
    except Exception as e:
        log(f"  parse failed: {e}")
        return heatmaps, 0

    log(f"  parse type={type(parsed).__name__} keys={list(parsed.keys())[:12] if isinstance(parsed, dict) else 'n/a'}")
    if not isinstance(parsed, dict):
        log(f"  unexpected parse type, attrs={dir(parsed)[:20]}")
        return heatmaps, 0

    objects = parsed.get("objects") or []
    if not isinstance(objects, list):
        objects = []

    nf = parsed.get("network_frames")
    frames = None
    if isinstance(nf, dict):
        frames = nf.get("frames")
    elif isinstance(nf, list):
        frames = nf
    if frames is None:
        frames = parsed.get("frames")
    if not frames:
        log(f"  no network frames — top keys: {list(parsed.keys())[:30]}")
        return heatmaps, 0

    name_lower = {n.lower(): n for n in player_names}
    grids = {n: np.zeros((GRID, GRID), dtype=np.int32) for n in player_names}

    actor_is_car: dict[int, bool] = {}
    actor_is_pri: dict[int, bool] = {}
    pri_name: dict[int, str] = {}
    car_to_pri: dict[int, int] = {}
    car_pos: dict[int, tuple[float, float]] = {}
    # reverse: PRI → car (latest)
    pri_to_car: dict[int, int] = {}

    hits = 0
    rb_seen = 0
    names_seen = 0
    links_seen = 0
    raw_names_found: set[str] = set()

    def accumulate(name: str, x: float, y: float) -> None:
        nonlocal hits
        if name not in grids:
            return
        nx = (float(x) + 4096.0) / 8192.0
        ny = (float(y) + 5120.0) / 10240.0
        nx = max(0.0, min(0.999, nx))
        ny = max(0.0, min(0.999, ny))
        c = int(nx * GRID)
        r = int(ny * GRID)
        grids[name][r, c] += 1
        hits += 1

    def object_name(oid: int | None) -> str:
        if oid is None or oid < 0 or oid >= len(objects):
            return ""
        o = objects[oid]
        return str(o) if o is not None else ""

    def classify_object(oname: str) -> str:
        low = oname.lower()
        # Prefer strict car archetype match
        if "archetypes.car." in low or low.endswith("car_default") or "archetypes.car" in low:
            return "car"
        if "playerreplicationinfo" in low or "default__pri" in low or low.endswith("pri_ta"):
            return "pri"
        return ""

    def match_player(raw_name: str) -> str | None:
        if not raw_name:
            return None
        key = _fold_name(raw_name)
        if not key:
            return None
        # exact folded match
        for cand, orig in name_lower.items():
            if _fold_name(cand) == key:
                return orig
        key2 = key.split("#")[0].strip()
        for cand, orig in name_lower.items():
            fc = _fold_name(cand)
            if fc == key2:
                return orig
        # substring folded (handles missing first letters from bad sanitizers)
        for cand, orig in name_lower.items():
            fc = _fold_name(cand)
            if len(fc) >= 3 and len(key) >= 3 and (fc in key or key in fc or fc in key2 or key2 in fc):
                return orig
        # last resort: compare alnum-only
        def alnum(x: str) -> str:
            return "".join(ch for ch in _fold_name(x) if ch.isalnum())
        ka = alnum(raw_name)
        if len(ka) >= 3:
            for cand, orig in name_lower.items():
                ca = alnum(cand)
                if ca == ka or (len(ca) >= 3 and (ca in ka or ka in ca)):
                    return orig
        return None

    # Seed names from header PlayerStats when present
    props = parsed.get("properties")
    if isinstance(props, list):
        # boxcars sometimes uses list of [key, value] pairs
        prop_map = {}
        for item in props:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                prop_map[str(item[0])] = item[1]
            elif isinstance(item, dict) and "name" in item:
                prop_map[str(item.get("name"))] = item.get("value")
        props = prop_map
    if isinstance(props, dict):
        pstats = props.get("PlayerStats") or props.get("PlayerStats") 
        if isinstance(pstats, list):
            for ps in pstats:
                if not isinstance(ps, dict):
                    continue
                nm = ps.get("Name") or ps.get("PlayerName") or ps.get("name")
                if isinstance(nm, str):
                    raw_names_found.add(nm)
                    match_player(nm)  # warm soft-match paths

    try:
        for fr in frames:
            frame_count += 1
            if not isinstance(fr, dict):
                continue

            for na in fr.get("new_actors") or []:
                if not isinstance(na, dict):
                    continue
                aid = _actor_id(na.get("actor_id"))
                oid = _obj_id(na.get("object_id") if "object_id" in na else na.get("object_ind"))
                if aid is None:
                    continue
                kind = classify_object(object_name(oid))
                if kind == "car":
                    actor_is_car[aid] = True
                    traj = na.get("initial_trajectory") or {}
                    if isinstance(traj, dict):
                        xy = _xy_from_location(traj.get("location"))
                        if xy:
                            car_pos[aid] = xy
                elif kind == "pri":
                    actor_is_pri[aid] = True

            for ua in fr.get("updated_actors") or []:
                if not isinstance(ua, dict):
                    continue
                aid = _actor_id(ua.get("actor_id"))
                if aid is None:
                    continue
                oid = _obj_id(ua.get("object_id"))
                oname = object_name(oid)
                oname_l = oname.lower()
                attr = ua.get("attribute")

                # Any String that looks like a player name
                s = _string_from_attribute(attr)
                if s and len(s) < 64:
                    raw_names_found.add(s)
                    matched = match_player(s)
                    # Accept as PRI name if: already PRI, attribute is PlayerName, or matched a known player
                    if matched and (
                        actor_is_pri.get(aid)
                        or "playername" in oname_l
                        or "playerreplicationinfo" in oname_l
                        or matched is not None
                    ):
                        # Prefer PlayerName attribute; still allow matched strings on PRI actors
                        if actor_is_pri.get(aid) or "playername" in oname_l or "playerreplicationinfo" in oname_l or matched:
                            if "playername" in oname_l or actor_is_pri.get(aid) or matched:
                                if matched:
                                    pri_name[aid] = matched
                                    actor_is_pri[aid] = True
                                    names_seen += 1

                # Car → PRI ONLY via Engine.Pawn:PlayerReplicationInfo (not other ActiveActors)
                link = _active_actor_id(attr)
                if link is not None and "playerreplicationinfo" in oname_l:
                    car_to_pri[aid] = link
                    pri_to_car[link] = aid
                    actor_is_car[aid] = True
                    links_seen += 1

                # RigidBody → car position
                xy = _xy_from_attribute(attr)
                if xy is not None:
                    rb_seen += 1
                    if (
                        actor_is_car.get(aid)
                        or "replicatedrbstate" in oname_l
                        or "rbactor" in oname_l
                    ):
                        actor_is_car[aid] = True
                        car_pos[aid] = xy

            for da in fr.get("deleted_actors") or []:
                did = _actor_id(da) if not isinstance(da, int) else da
                if did is not None:
                    actor_is_car.pop(did, None)
                    car_to_pri.pop(did, None)
                    car_pos.pop(did, None)
                    actor_is_pri.pop(did, None)
                    # keep pri_name[did] — recycled IDs still help late links
                    for k, v in list(pri_to_car.items()):
                        if v == did:
                            pri_to_car.pop(k, None)

            # sample every 2nd frame (denser heatmaps)
            if frame_count % 2 != 0:
                continue
            for caid, xy in list(car_pos.items()):
                pri = car_to_pri.get(caid)
                pname = pri_name.get(pri) if pri is not None else None
                if not pname:
                    # reverse: this actor might itself be a known PRI with a car pos (rare)
                    pname = pri_name.get(caid)
                if pname:
                    accumulate(pname, xy[0], xy[1])

    except Exception as e:
        log(f"  frame walk error: {e}")
        traceback.print_exc()

    # Fallback: if no hits but we have car positions + player names, dump debug
    if hits == 0:
        sample_pri = list(pri_name.items())[:6]
        sample_links = list(car_to_pri.items())[:6]
        log(
            f"  debug names_raw={list(raw_names_found)[:12]} "
            f"pri_map={sample_pri} links={sample_links} "
            f"want={player_names}"
        )

    for n, g in grids.items():
        flat = g.flatten().tolist()
        mx = int(g.max()) if g.size else 0
        heatmaps[n] = {"cols": GRID, "rows": GRID, "cells": flat, "max": mx}

    log(
        f"  frames={frame_count} rb={rb_seen} names={names_seen} links={links_seen} "
        f"hits={hits} cars={len(actor_is_car)} pris={len(pri_name)} "
        f"heat_max={[heatmaps[n]['max'] for n in player_names]}"
    )
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
    heatmaps, frame_count = extract_heatmap_from_boxcars(raw, names, replay_path=path)
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
        f"parser: rrrocket={HAS_RRROCKET} ({RRROCKET_BIN}), boxcars_py={bool(boxcars_parse)}, numpy={HAS_NUMPY}"
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
