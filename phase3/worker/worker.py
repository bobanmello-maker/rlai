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
import difflib
import re
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
    rb = _rigid_body_from_attribute(attr)
    if rb and rb.get("location"):
        return _xy_from_location(rb["location"])
    if attr is None or not isinstance(attr, dict):
        return None
    if "Location" in attr:
        return _xy_from_location(attr["Location"])
    if "location" in attr or "Location" in attr:
        return _xy_from_location(attr.get("location") or attr.get("Location"))
    if "x" in attr or "X" in attr:
        return _xy_from_location(attr)
    return None


def _vec3(obj: Any) -> tuple[float, float, float] | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        x = obj.get("x", obj.get("X"))
        y = obj.get("y", obj.get("Y"))
        z = obj.get("z", obj.get("Z"))
        if x is not None and y is not None:
            return float(x), float(y), float(z or 0.0)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return float(obj[0]), float(obj[1]), float(obj[2] if len(obj) > 2 else 0.0)
    return None


def _rigid_body_from_attribute(attr: Any) -> dict | None:
    """Return {location:(x,y,z), velocity:(x,y,z)} from RigidBody attribute if present."""
    if attr is None or not isinstance(attr, dict):
        return None
    rb = attr.get("RigidBody") if "RigidBody" in attr else None
    if rb is None and ("location" in attr or "Location" in attr):
        rb = attr
    if not isinstance(rb, dict):
        return None
    loc = _vec3(rb.get("location") or rb.get("Location"))
    vel = _vec3(
        rb.get("linear_velocity")
        or rb.get("LinearVelocity")
        or rb.get("linearVelocity")
        or rb.get("velocity")
        or rb.get("Velocity")
    )
    if loc is None:
        return None
    return {"location": loc, "velocity": vel or (0.0, 0.0, 0.0)}


def _speed_uu(vel: tuple[float, float, float] | None) -> float:
    if not vel:
        return 0.0
    return (vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2) ** 0.5


# Unreal units/s → km/h (community approx used by RL tools)
UU_TO_KMH = 0.036


def _speed_kmh(speed_uu: float) -> float:
    return round(float(speed_uu) * UU_TO_KMH, 1)


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
) -> tuple[dict[str, dict], int, dict]:
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
    intel: dict[str, Any] = {
        "ball_speed_by_frame": {},  # frame_idx -> speed_uu (sparse)
        "ball_z_by_frame": {},
        "car_dist_to_ball": [],  # optional samples
    }
    if (not HAS_BOXCARS and not HAS_RRROCKET) or not HAS_NUMPY:
        return heatmaps, 0, intel

    try:
        parsed = parse_replay_dict(raw, replay_path=replay_path)
    except Exception as e:
        log(f"  parse failed: {e}")
        return heatmaps, 0, intel

    log(f"  parse type={type(parsed).__name__} keys={list(parsed.keys())[:12] if isinstance(parsed, dict) else 'n/a'}")
    if not isinstance(parsed, dict):
        log(f"  unexpected parse type, attrs={dir(parsed)[:20]}")
        return heatmaps, 0, intel

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
        return heatmaps, 0, intel

    name_lower = {n.lower(): n for n in player_names}
    grids = {n: np.zeros((GRID, GRID), dtype=np.int32) for n in player_names}

    actor_is_car: dict[int, bool] = {}
    actor_is_pri: dict[int, bool] = {}
    pri_name: dict[int, str] = {}
    car_to_pri: dict[int, int] = {}
    car_pos: dict[int, tuple[float, float]] = {}
    car_pos3: dict[int, tuple[float, float, float]] = {}
    # reverse: PRI → car (latest)
    pri_to_car: dict[int, int] = {}
    actor_is_ball: dict[int, bool] = {}
    # actor_id → archetype name from new_actors (needed: updated_actors object_id is the *attribute*)
    actor_object_name: dict[int, str] = {}
    ball_pos: tuple[float, float, float] | None = None
    ball_vel: tuple[float, float, float] | None = None
    ball_speed_by_frame: dict[int, float] = {}
    ball_z_by_frame: dict[int, float] = {}
    ball_pos_by_frame: dict[int, tuple[float, float, float]] = {}
    # kickoff: track early ball motion
    kickoff_ball_moved_frame: int | None = None
    kickoff_first_car: str | None = None
    kickoff_first_dist: float | None = None

    hits = 0
    rb_seen = 0
    names_seen = 0
    links_seen = 0
    ball_rb_seen = 0
    ball_actors_seen: set[int] = set()
    raw_names_found: set[str] = set()
    debug_ball_obj_names: set[str] = set()

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
        # Prefer strict car archetype match (exclude car components)
        if "carcomponent" in low or "specialpickup" in low:
            return ""
        if (
            "archetypes.car." in low
            or low.endswith("car_default")
            or low == "tagame.car_ta"
            or "archetypes.car" in low
            or low.endswith(".car_ta")
        ):
            return "car"
        if (
            "playerreplicationinfo" in low
            or "default__pri" in low
            or low.endswith("pri_ta")
            or "tagame.pri_ta" in low
        ):
            return "pri"
        # Ball archetypes: Archetypes.Ball.Ball_Default, Ball_Basketball, CubeBall, Ball_Breakout, …
        # Also class names: TAGame.Ball_TA, TAGame.Ball_Breakout_TA, …
        if (
            "archetypes.ball" in low
            or "ball_ta" in low
            or low.endswith(".ball")
            or low.startswith("tagame.ball")
            or "cubeball" in low
            or (".ball_" in low and "car" not in low and "component" not in low)
            or (low.endswith("ball") and "car" not in low and "boost" not in low and "pickup" not in low)
        ):
            return "ball"
        return ""

    def match_player(raw_name: str) -> str | None:
        if not raw_name:
            return None
        # ignore non-player strings
        if raw_name.strip().lower() in {"offline match", "online match", "none", ""}:
            return None

        def alnum(x: str) -> str:
            return "".join(ch for ch in _fold_name(x) if ch.isalnum())

        def split_suffix(x: str) -> tuple[str, str]:
            # "Name(2)" → ("Name", "(2)")
            m = re.match(r"^(.*?)(\(\d+\))\s*$", x.strip())
            if m:
                return m.group(1), m.group(2)
            return x.strip(), ""

        key = _fold_name(raw_name)
        if not key:
            return None
        raw_base, raw_suf = split_suffix(raw_name)
        key_base = alnum(raw_base)
        key_all = alnum(raw_name)

        # 1) exact folded
        for cand, orig in name_lower.items():
            if _fold_name(cand) == key:
                return orig

        # 2) same numeric suffix + fuzzy base (handles Činčila vs inila)
        best: tuple[float, str] | None = None
        for cand, orig in name_lower.items():
            c_base, c_suf = split_suffix(cand)
            # prefer same (1)/(2)/(3) suffix when both have it
            if raw_suf and c_suf and raw_suf != c_suf:
                continue
            ca = alnum(c_base) if raw_suf else alnum(cand)
            ka = key_base if raw_suf else key_all
            if not ca or not ka:
                continue
            if ca == ka:
                return orig
            ratio = difflib.SequenceMatcher(None, ca, ka).ratio()
            # also try full alnum
            ratio2 = difflib.SequenceMatcher(None, alnum(cand), key_all).ratio()
            ratio = max(ratio, ratio2)
            if ratio >= 0.78 and (best is None or ratio > best[0]):
                best = (ratio, orig)
        if best:
            return best[1]

        # 3) substring fallback
        for cand, orig in name_lower.items():
            fc = alnum(cand)
            if len(fc) >= 4 and len(key_all) >= 4 and (fc in key_all or key_all in fc):
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
                oname_spawn = object_name(oid)
                if oname_spawn:
                    actor_object_name[aid] = oname_spawn
                kind = classify_object(oname_spawn)
                traj = na.get("initial_trajectory") or {}
                traj_loc = None
                if isinstance(traj, dict):
                    traj_loc = _vec3(traj.get("location") or traj.get("Location"))
                if kind == "car":
                    actor_is_car[aid] = True
                    if traj_loc:
                        car_pos[aid] = (traj_loc[0], traj_loc[1])
                        car_pos3[aid] = traj_loc
                elif kind == "pri":
                    actor_is_pri[aid] = True
                elif kind == "ball":
                    actor_is_ball[aid] = True
                    ball_actors_seen.add(aid)
                    if oname_spawn:
                        debug_ball_obj_names.add(oname_spawn)
                    if traj_loc:
                        ball_pos = traj_loc
                        ball_vel = (0.0, 0.0, 0.0)

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

                # RigidBody → car / ball physics
                # NOTE: ua.object_id is the *attribute* stream (e.g. TAGame.RBActor_TA:ReplicatedRBState),
                # NOT the actor archetype. Actor type must come from new_actors → actor_object_name.
                rb = _rigid_body_from_attribute(attr)
                if rb is not None:
                    rb_seen += 1
                    loc = rb["location"]
                    vel = rb["velocity"]
                    arch = (actor_object_name.get(aid) or "").lower()
                    arch_kind = classify_object(actor_object_name.get(aid) or "")

                    is_ball = (
                        actor_is_ball.get(aid)
                        or arch_kind == "ball"
                        or "archetypes.ball" in arch
                        or arch.startswith("tagame.ball")
                        or ("ball" in arch and "car" not in arch and "component" not in arch and "pickup" not in arch)
                    )
                    is_car = (
                        actor_is_car.get(aid)
                        or arch_kind == "car"
                        or "archetypes.car" in arch
                        or arch.endswith("car_ta")
                        or ("car" in arch and "component" not in arch and "ball" not in arch)
                    )
                    # Attribute-name fallback only when we still have no archetype mapping
                    if not is_ball and not is_car and not arch:
                        if "replicatedrbstate" in oname_l or "rbactor" in oname_l:
                            # Unknown RB actor: prefer ball if near field center and no car link
                            if aid not in car_to_pri and abs(loc[0]) < 4500 and abs(loc[1]) < 6000:
                                is_ball = True
                            else:
                                is_car = True

                    if is_ball and not is_car:
                        actor_is_ball[aid] = True
                        ball_actors_seen.add(aid)
                        if arch:
                            debug_ball_obj_names.add(actor_object_name.get(aid) or arch)
                        ball_pos = loc
                        ball_vel = vel
                        ball_rb_seen += 1
                    elif is_car or actor_is_car.get(aid):
                        actor_is_car[aid] = True
                        car_pos[aid] = (loc[0], loc[1])
                        car_pos3[aid] = loc

            for da in fr.get("deleted_actors") or []:
                did = _actor_id(da) if not isinstance(da, int) else da
                if did is not None:
                    actor_is_car.pop(did, None)
                    actor_is_ball.pop(did, None)
                    actor_object_name.pop(did, None)
                    car_to_pri.pop(did, None)
                    car_pos.pop(did, None)
                    car_pos3.pop(did, None)
                    actor_is_pri.pop(did, None)
                    for k, v in list(pri_to_car.items()):
                        if v == did:
                            pri_to_car.pop(k, None)

            # Record ball speed / position every frame (for goal lookup + kickoff)
            if ball_pos is not None:
                sp = _speed_uu(ball_vel) if ball_vel is not None else 0.0
                ball_speed_by_frame[frame_count] = sp
                ball_z_by_frame[frame_count] = ball_pos[2]
                ball_pos_by_frame[frame_count] = ball_pos
                # Kickoff: first time ball leaves center with meaningful speed
                if (
                    kickoff_ball_moved_frame is None
                    and frame_count < 400
                    and sp > 350
                ):
                    # near midfield at kickoff (slightly wider window)
                    if abs(ball_pos[0]) < 800 and abs(ball_pos[1]) < 800:
                        kickoff_ball_moved_frame = frame_count
                        # closest car to ball = first touch proxy
                        best_d = None
                        best_name = None
                        for caid, cxy in list(car_pos.items()):
                            pri = car_to_pri.get(caid)
                            pname = pri_name.get(pri) if pri is not None else pri_name.get(caid)
                            if not pname:
                                continue
                            d = ((cxy[0] - ball_pos[0]) ** 2 + (cxy[1] - ball_pos[1]) ** 2) ** 0.5
                            if best_d is None or d < best_d:
                                best_d = d
                                best_name = pname
                        kickoff_first_car = best_name
                        kickoff_first_dist = best_d

            # sample every 2nd frame (denser heatmaps)
            if frame_count % 2 != 0:
                continue
            for caid, xy in list(car_pos.items()):
                pri = car_to_pri.get(caid)
                pname = pri_name.get(pri) if pri is not None else None
                if not pname:
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

    intel["ball_speed_by_frame"] = ball_speed_by_frame
    intel["ball_z_by_frame"] = ball_z_by_frame
    intel["ball_pos_by_frame"] = ball_pos_by_frame
    intel["kickoff"] = {
        "ball_moved_frame": kickoff_ball_moved_frame,
        "first_touch_player": kickoff_first_car,
        "first_touch_dist": round(kickoff_first_dist, 1) if kickoff_first_dist is not None else None,
    }
    intel["ball_rb_samples"] = ball_rb_seen
    intel["ball_actor_ids"] = sorted(ball_actors_seen)

    log(
        f"  frames={frame_count} rb={rb_seen} ball_rb={ball_rb_seen} names={names_seen} links={links_seen} "
        f"hits={hits} cars={len(actor_is_car)} pris={len(pri_name)} ball_speeds={len(ball_speed_by_frame)} "
        f"ball_actors={len(ball_actors_seen)} heat_max={[heatmaps[n]['max'] for n in player_names]}"
    )
    if ball_rb_seen == 0:
        # Help diagnose missing ball: sample object names that look related
        sample_objs = [str(o) for o in objects if o and "ball" in str(o).lower()][:12]
        log(
            f"  warn: no ball RB — ball_obj_names={list(debug_ball_obj_names)[:8]} "
            f"objects_with_ball={sample_objs}"
        )
    return heatmaps, frame_count, intel


def _lookup_ball_speed(intel: dict, frame_hint: Any) -> tuple[float | None, float | None, bool]:
    """Nearest ball speed sample around a goal frame. Returns (speed_uu, speed_kmh, aerial).

    frame_hint may be network frame index or (rarely) time in seconds — we try both.
    Prefer the *maximum* speed in a small window before the goal (ball is fastest just
    before crossing the line, then may slow on post/ground contact).
    """
    speeds = intel.get("ball_speed_by_frame") or {}
    zs = intel.get("ball_z_by_frame") or {}
    if not speeds:
        return None, None, False
    try:
        target = float(frame_hint)
    except (TypeError, ValueError):
        return None, None, False

    candidates: list[int] = []
    # Direct frame index (Ballchasing / header Goals usually give frame numbers)
    t_frame = int(target)
    # Also try ~30 fps conversion if hint looks like seconds (small number)
    t_from_sec = int(round(target * 30.0)) if target < 900 else None

    for f in speeds.keys():
        fi = int(f)
        if abs(fi - t_frame) <= 60:
            candidates.append(fi)
        elif t_from_sec is not None and abs(fi - t_from_sec) <= 60:
            candidates.append(fi)

    if not candidates:
        # fallback: nearest any frame within ±90
        best_f = None
        best_d = 9999
        for f in speeds.keys():
            d = abs(int(f) - t_frame)
            if d < best_d:
                best_d = d
                best_f = int(f)
        if best_f is None or best_d > 90:
            return None, None, False
        candidates = [best_f]

    # Prefer peak speed in the window (shot speed), ignore near-zero
    best_su = -1.0
    best_z = 0.0
    for fi in candidates:
        su = float(speeds.get(fi) or 0)
        if su > best_su:
            best_su = su
            best_z = float(zs.get(fi) or 0)
    if best_su < 0:
        return None, None, False
    aerial = best_z > 120  # ball clearly off ground (~ ball radius + margin)
    return round(best_su, 1), _speed_kmh(best_su), aerial


def build_advanced(
    replay_id: str,
    details: dict,
    heatmaps: dict,
    frame_count: int,
    frames_ok: bool,
    intel: dict | None = None,
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

    intel = intel or {}
    goal_speeds_all: list[dict] = []

    for g in details.get("goals") or []:
        player = (
            (g.get("player") or {}).get("name")
            if isinstance(g.get("player"), dict)
            else g.get("player")
        )
        # ballchasing variants: frame / frame_number / time
        frame_hint = (
            g.get("frame_number")
            or g.get("frame")
            or g.get("time")
            or g.get("time_seconds")
        )
        su, sk, aerial = _lookup_ball_speed(intel, frame_hint)
        entry = {
            "t": frame_hint,
            "type": "goal",
            "player": player,
            "team": None,
            "speed_uu": su,
            "speed_kmh": sk,
            "aerial": aerial,
        }
        timeline.append(entry)
        if su is not None:
            goal_speeds_all.append(
                {
                    "player": player,
                    "frame": frame_hint,
                    "speed_uu": su,
                    "speed_kmh": sk,
                    "aerial": aerial,
                }
            )

    # Attach per-player goal events with speeds (one entry per scored goal when possible)
    by_player: dict[str, list] = {}
    for gs in goal_speeds_all:
        p = gs.get("player") or ""
        by_player.setdefault(p, []).append(gs)

    for po in players_out:
        name = po.get("name")
        glist = by_player.get(name) or []
        goals_n = int((po.get("stats_snapshot") or {}).get("goals") or 0)
        if glist:
            po["events"]["goals"] = [
                {
                    "t": g.get("frame"),
                    "n": 1,
                    "speed_uu": g.get("speed_uu"),
                    "speed_kmh": g.get("speed_kmh"),
                    "aerial": g.get("aerial"),
                }
                for g in glist
            ]
        elif goals_n:
            po["events"]["goals"] = [{"t": None, "n": goals_n}]

        # per-player goal speed summary
        speeds = [g["speed_kmh"] for g in glist if g.get("speed_kmh") is not None]
        if speeds:
            po["stats_snapshot"]["goal_speed_max_kmh"] = max(speeds)
            po["stats_snapshot"]["goal_speed_min_kmh"] = min(speeds)
            po["stats_snapshot"]["goal_speed_avg_kmh"] = round(sum(speeds) / len(speeds), 1)
            po["stats_snapshot"]["goals_aerial"] = sum(1 for g in glist if g.get("aerial"))
            po["stats_snapshot"]["goals_ground"] = sum(1 for g in glist if not g.get("aerial"))

    # Match-level goal speed aggregate
    all_kmh = [g["speed_kmh"] for g in goal_speeds_all if g.get("speed_kmh") is not None]
    goal_speed_summary = None
    if all_kmh:
        fastest = max(goal_speeds_all, key=lambda x: x.get("speed_kmh") or 0)
        slowest = min(goal_speeds_all, key=lambda x: x.get("speed_kmh") or 9999)
        goal_speed_summary = {
            "count": len(all_kmh),
            "max_kmh": max(all_kmh),
            "min_kmh": min(all_kmh),
            "avg_kmh": round(sum(all_kmh) / len(all_kmh), 1),
            "fastest_player": fastest.get("player"),
            "slowest_player": slowest.get("player"),
            "aerial_pct": round(
                100.0 * sum(1 for g in goal_speeds_all if g.get("aerial")) / len(goal_speeds_all), 1
            ),
        }

    kickoff = intel.get("kickoff") or {}

    # Match-level ball motion summary (independent of goals)
    ball_speeds_all = list((intel.get("ball_speed_by_frame") or {}).values())
    ball_zs = list((intel.get("ball_z_by_frame") or {}).values())
    ball_motion = None
    if ball_speeds_all:
        max_sp = max(ball_speeds_all)
        ball_motion = {
            "samples": len(ball_speeds_all),
            "max_speed_uu": round(max_sp, 1),
            "max_speed_kmh": _speed_kmh(max_sp),
            "avg_speed_kmh": _speed_kmh(sum(ball_speeds_all) / len(ball_speeds_all)),
            "max_height_uu": round(max(ball_zs), 1) if ball_zs else None,
            "air_pct": round(
                100.0 * sum(1 for z in ball_zs if z > 120) / max(1, len(ball_zs)), 1
            )
            if ball_zs
            else None,
        }

    return {
        "v": 2,
        "replay_id": replay_id,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "duration": duration,
        "map": map_name,
        "playlist": playlist,
        "players": players_out,
        "timeline": timeline,
        "mistakes": mistakes,
        "goal_speed": goal_speed_summary,
        "ball_motion": ball_motion,
        "kickoff": {
            "first_touch_player": kickoff.get("first_touch_player"),
            "ball_moved_frame": kickoff.get("ball_moved_frame"),
            "first_touch_dist": kickoff.get("first_touch_dist"),
        },
        "source": {
            "ballchasing": bool(details),
            "frames_parsed": frames_ok and frame_count > 0,
            "frame_count": frame_count,
            "boxcars": HAS_BOXCARS,
            "ball_speed_samples": len(intel.get("ball_speed_by_frame") or {}),
            "ball_rb_samples": int(intel.get("ball_rb_samples") or 0),
            "metrics": [
                "goal_speed",
                "kickoff_first_touch",
                "aerial_goal",
                "ball_motion",
            ],
        },
    }


def process_one(replay_id: str) -> dict:
    log(f"Processing {replay_id}")
    details = fetch_bc_details(replay_id)
    names = [p["name"] for p in _team_players(details)]

    path = download_replay(replay_id)
    raw = path.read_bytes()
    heatmaps, frame_count, intel = extract_heatmap_from_boxcars(raw, names, replay_path=path)
    frames_ok = frame_count > 0

    advanced = build_advanced(
        replay_id, details, heatmaps, frame_count, frames_ok, intel=intel
    )

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
