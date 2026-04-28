import json
import logging
import os
import time
from typing import Dict, List

import boto3

logger = logging.getLogger(__name__)

# Amazon Nova Lite — no approval form required, just enable in Bedrock Model Access.
# Override with env var BEDROCK_MODEL_ID to use any other model.
_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

_client = boto3.client("bedrock-runtime", region_name=_REGION)

_AUDIO_SYSTEM_PROMPT = (
    "You are a music analysis expert. Given a list of songs (track name + artist), "
    "estimate their Spotify-style audio features based on your knowledge of the music.\n\n"
    "Return a JSON array. Each element must have exactly these fields:\n"
    "- track_id: string (the id provided)\n"
    "- energy: float 0.0-1.0 (intensity and activity; 0=calm, 1=intense)\n"
    "- valence: float 0.0-1.0 (musical positiveness; 0=sad/dark, 1=happy/euphoric)\n"
    "- danceability: float 0.0-1.0 (rhythm regularity and beat strength)\n"
    "- acousticness: float 0.0-1.0 (acoustic vs electronic; 1=fully acoustic)\n"
    "- speechiness: float 0.0-1.0 (spoken word; >0.66=speech, 0.33-0.66=mixed, <0.33=music)\n"
    "- instrumentalness: float 0.0-1.0 (0=vocals present, 1=purely instrumental)\n"
    "- liveness: float 0.0-1.0 (live audience probability; >0.8=likely live recording)\n"
    "- loudness: float -60.0 to 0.0 (overall loudness in dB; typical songs: -10 to -5)\n"
    "- tempo: float 40-220 (estimated BPM)\n"
    "- key: integer 0-11 (C=0, C#=1, D=2, Eb=3, E=4, F=5, F#=6, G=7, Ab=8, A=9, Bb=10, B=11)\n"
    "- mode: integer 0 or 1 (0=minor, 1=major)\n"
    "- time_signature: integer 3-7 (beats per bar; 4 is most common)\n"
    "- estimated: true\n\n"
    "Return ONLY a valid JSON array with no explanation, markdown fences, or extra text."
)

_GENRE_SYSTEM_PROMPT = (
    "You are a music genre expert. Given a list of artists and some of their track titles, "
    "assign Spotify-style genre tags based on your knowledge of the music.\n\n"
    "Spotify genre tags are lowercase, hyphenated strings. Be specific — prefer tags like "
    "'congolese rumba', 'afro gospel', 'lingala gospel', 'afropop', 'worship', "
    "'christian hip-hop', 'highlife' over generic ones like 'pop' or 'world'.\n\n"
    "Return a JSON array. Each element must have exactly these fields:\n"
    "- artist_id: string (the id provided)\n"
    "- genres: array of 1-5 genre strings (lowercase, Spotify-style)\n\n"
    "Return ONLY a valid JSON array with no explanation, markdown fences, or extra text."
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Bedrock helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_body(system_prompt: str, user_message: str) -> str:
    """Builds the Bedrock request body for the active model family."""
    if _MODEL_ID.startswith("anthropic."):
        return json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        })
    elif _MODEL_ID.startswith("amazon.nova"):
        return json.dumps({
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_message}]}],
            "inferenceConfig": {"maxTokens": 4096, "temperature": 0.1},
        })
    else:
        raise ValueError(
            f"Unsupported model family: {_MODEL_ID}. "
            "Set BEDROCK_MODEL_ID to an anthropic.* or amazon.nova* model."
        )


def _extract_text(response_body: dict) -> str:
    """Pulls the generated text out of a Bedrock response regardless of model family."""
    if _MODEL_ID.startswith("anthropic."):
        return response_body["content"][0]["text"].strip()
    elif _MODEL_ID.startswith("amazon.nova"):
        return response_body["output"]["message"]["content"][0]["text"].strip()
    return ""


def _invoke(system_prompt: str, user_message: str) -> str:
    """Makes a single Bedrock call and returns the raw text response."""
    response = _client.invoke_model(
        modelId=_MODEL_ID,
        body=_build_body(system_prompt, user_message),
        contentType="application/json",
        accept="application/json",
    )
    raw = _extract_text(json.loads(response["body"].read()))
    if raw.startswith("```"):
        raw = raw[raw.index("\n") + 1:]
        raw = raw[:raw.rfind("```")].strip()
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Audio feature estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_audio_features(tracks: List[Dict]) -> Dict[str, Dict]:
    """
    Estimates Spotify-style audio features for a list of tracks using
    an LLM via Amazon Bedrock.

    Args:
        tracks: raw track dicts from Spotify (must have 'id', 'name', 'artists')

    Returns:
        dict keyed by track_id with estimated feature values.
    """
    seen: Dict[str, Dict] = {}
    for t in tracks:
        tid = t.get("id")
        if tid and tid not in seen:
            seen[tid] = t
    unique_tracks = list(seen.values())

    features_map: Dict[str, Dict] = {}
    batch_size = 20

    for i in range(0, len(unique_tracks), batch_size):
        batch = unique_tracks[i: i + batch_size]
        features_map.update(_estimate_audio_batch(batch))
        if i + batch_size < len(unique_tracks):
            time.sleep(0.3)

    logger.info(f"Bedrock estimated audio features for {len(features_map)} tracks")
    return features_map


def _estimate_audio_batch(tracks: List[Dict]) -> Dict[str, Dict]:
    track_lines = "\n".join(
        f'{i+1}. id="{t["id"]}" | "{t["name"]}" by '
        f'{t["artists"][0]["name"] if t.get("artists") else "Unknown"}'
        for i, t in enumerate(tracks)
    )
    try:
        raw = _invoke(_AUDIO_SYSTEM_PROMPT,
                      f"Estimate audio features for these tracks:\n\n{track_lines}")
        results = json.loads(raw)
        return {r["track_id"]: r for r in results if r.get("track_id")}
    except Exception as e:
        logger.error(f"Bedrock audio feature estimation failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Artist genre estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_artist_genres(artists: List[Dict]) -> Dict[str, List[str]]:
    """
    Estimates Spotify-style genre tags for artists whose genre list is empty.

    Args:
        artists: list of dicts with:
                   'id'          — Spotify artist ID
                   'name'        — artist name
                   'track_names' — (optional) list of sample track titles for context

    Returns:
        dict keyed by artist_id with a list of estimated genre strings.
    """
    if not artists:
        return {}

    unique: Dict[str, Dict] = {}
    for a in artists:
        aid = a.get("id")
        if aid and aid not in unique:
            unique[aid] = a

    genres_map: Dict[str, List[str]] = {}
    batch_size = 20

    for i in range(0, len(unique), batch_size):
        batch = list(unique.values())[i: i + batch_size]
        genres_map.update(_estimate_genres_batch(batch))
        if i + batch_size < len(unique):
            time.sleep(0.3)

    logger.info(f"Bedrock estimated genres for {len(genres_map)} artists")
    return genres_map


def _estimate_genres_batch(artists: List[Dict]) -> Dict[str, List[str]]:
    artist_lines = "\n".join(
        '{i}. id="{aid}" | Artist: "{name}" | Sample tracks: {tracks}'.format(
            i=i + 1,
            aid=a["id"],
            name=a["name"],
            tracks=", ".join(a.get("track_names", [])[:3]) or "unknown",
        )
        for i, a in enumerate(artists)
    )
    try:
        raw = _invoke(_GENRE_SYSTEM_PROMPT,
                      f"Estimate genre tags for these artists:\n\n{artist_lines}")
        results = json.loads(raw)
        return {
            r["artist_id"]: r["genres"]
            for r in results
            if r.get("artist_id") and isinstance(r.get("genres"), list)
        }
    except Exception as e:
        logger.error(f"Bedrock genre estimation failed: {e}")
        return {}
