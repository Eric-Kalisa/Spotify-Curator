"""
discoverer.py
=============
Stage 4: Discovery  (Spotify deprecated endpoints replaced)

Finds new candidate tracks for each cluster using three sources:
  A) Spotify Related Artists  — two-hop: seed artists → related artists → their top tracks.
                                 Uses only non-deprecated Spotify endpoints.
  B) Deezer Artist Radio       — artist-seeded radio from Deezer's free API.
                                 Also extracts BPM when Deezer has it.
  C) New Releases              — weekly new music in matching genres (Spotify, unchanged).

sp.recommendations() and sp.audio_features() were deprecated for new Spotify apps
in November 2024. Audio features are now estimated via llm_features.estimate_audio_features()
as a fallback when Spotify returns nothing.
"""

import time
import logging
import numpy as np
import pandas as pd
import requests
from typing import List, Dict, Optional

from auth_spotify import get_spotify_client
from taste_model import ClusterProfile

try:
    from llm_features import estimate_audio_features as _llm_estimate_audio
    _LLM_FALLBACK_AVAILABLE = True
except ImportError:
    _LLM_FALLBACK_AVAILABLE = False

logger = logging.getLogger(__name__)
sp = get_spotify_client()

_DEEZER_BASE = "https://api.deezer.com"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_track(track: Dict, source: str) -> Dict:
    """
    Normalizes a raw Spotify track object into the flat dict format
    that track_to_vector() in scorer.py expects.

    Handles both track objects from recommendations and new releases,
    which have slightly different shapes.
    """
    artist_id = None
    artist_name = None
    if track.get("artists"):
        artist_id   = track["artists"][0].get("id")
        artist_name = track["artists"][0].get("name")

    release_date = track.get("album", {}).get("release_date", "")
    release_year = None
    if release_date:
        try:
            release_year = int(release_date[:4])
        except ValueError:
            pass

    return {
        "track_id":    track.get("id"),
        "title":       track.get("name"),
        "artist":      artist_name,
        "artist_id":   artist_id,
        "popularity":  track.get("popularity"),
        "explicit":    track.get("explicit"),
        "duration_ms": track.get("duration_ms"),
        "release_year": release_year,
        "source":      source,
        # audio features filled in by _enrich_with_audio_features()
        "energy":           None,
        "valence":          None,
        "danceability":     None,
        "acousticness":     None,
        "tempo_bpm":        None,
        "loudness":         None,
        "speechiness":      None,
        "instrumentalness": None,
        "liveness":         None,
        "key":              None,
        "mode":             None,
        "time_signature":   None,
        # artist genres filled by _enrich_with_artist_genres()
        "artist_genres":    [],
    }


def _enrich_with_audio_features(tracks: List[Dict]) -> List[Dict]:
    """
    Fills audio features for a list of normalized track dicts.

    Strategy (in order):
      1. Spotify audio_features API — works for apps created before Nov 2024.
      2. LLM estimation via llm_features.py — fallback for any tracks Spotify
         could not cover (deprecated endpoint or missing data).
      3. Values already set on the track (e.g. BPM from Deezer) — kept as a
         last resort when both Spotify and LLM have nothing.
    """
    track_ids = [t["track_id"] for t in tracks if t.get("track_id")]
    if not track_ids:
        return tracks

    features_map: Dict[str, Dict] = {}

    # ── Step 1: try Spotify ───────────────────────────────────────────────────
    try:
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i + 100]
            results = sp.audio_features(batch)
            for f in (results or []):
                if f:
                    features_map[f["id"]] = f
            time.sleep(0.1)
    except Exception as e:
        logger.warning(
            f"sp.audio_features() unavailable ({e}). "
            "Falling back to LLM estimation for all tracks."
        )

    # ── Step 2: LLM fallback for anything Spotify missed ─────────────────────
    missing = [
        t for t in tracks
        if t.get("track_id") not in features_map
        and t.get("title") and t.get("artist")
    ]
    if missing and _LLM_FALLBACK_AVAILABLE:
        llm_input = [
            {
                "id":      t["track_id"],
                "name":    t["title"],
                "artists": [{"name": t["artist"]}],
            }
            for t in missing
        ]
        logger.info(f"LLM audio feature estimation for {len(llm_input)} tracks")
        try:
            estimated = _llm_estimate_audio(llm_input)
            for track_id, feat in estimated.items():
                feat["id"] = track_id
                features_map[track_id] = feat
        except Exception as e:
            logger.error(f"LLM audio feature estimation failed: {e}")
    elif missing:
        logger.warning(
            f"{len(missing)} tracks have no audio features and "
            "llm_features is not importable — they may be filtered by scorer.py"
        )

    # ── Step 3: merge into track dicts ───────────────────────────────────────
    for track in tracks:
        f = features_map.get(track.get("track_id"), {})

        if f:
            # Spotify uses "tempo"; LLM uses "tempo" too
            track["energy"]           = f.get("energy")
            track["valence"]          = f.get("valence")
            track["danceability"]     = f.get("danceability")
            track["acousticness"]     = f.get("acousticness")
            track["tempo_bpm"]        = f.get("tempo") or f.get("tempo_bpm")
            track["loudness"]         = f.get("loudness")
            track["speechiness"]      = f.get("speechiness")
            track["instrumentalness"] = f.get("instrumentalness")
            track["liveness"]         = f.get("liveness")
            track["key"]              = f.get("key")
            track["mode"]             = f.get("mode")
            track["time_signature"]   = f.get("time_signature")
        # else: keep whatever the track already has (e.g. BPM pre-filled from Deezer)

        # Derive normalized features from whatever we ended up with
        if track["tempo_bpm"] is not None:
            track["tempo_norm"] = float(np.clip((track["tempo_bpm"] - 60) / 140, 0, 1))
        if track["loudness"] is not None:
            track["loudness_norm"] = float(np.clip((track["loudness"] + 60) / 60, 0, 1))
        if track["release_year"] is not None:
            track["era_score"] = float(np.clip((track["release_year"] - 1950) / 75, 0, 1))
        if track["popularity"] is not None:
            track["popularity_norm"] = track["popularity"] / 100
        track["hour_sin"] = 0.0
        track["hour_cos"] = 1.0
        if track["key"] is not None and track["mode"] is not None:
            track["harmonic_sig"] = (track["key"] / 11) * track["mode"]
        else:
            track["harmonic_sig"] = 0.0
        if track["time_signature"] is not None:
            track["time_sig_norm"] = float(np.clip((track["time_signature"] - 3) / (7 - 3), 0, 1))

    return tracks


def _enrich_with_artist_genres(tracks: List[Dict]) -> List[Dict]:
    """
    Fetches genre tags for all unique artists across the track list
    and merges them into each track dict.
    """
    artist_ids = list({
        t["artist_id"] for t in tracks
        if t.get("artist_id")
    })
    if not artist_ids:
        return tracks

    artist_genre_map = {}
    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i:i+50]
        try:
            results = sp.artists(batch)
            for a in results["artists"]:
                if a:
                    artist_genre_map[a["id"]] = a.get("genres", [])
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error fetching artist genres: {e}")

    for track in tracks:
        track["artist_genres"] = artist_genre_map.get(track.get("artist_id"), [])

    return tracks


def _deduplicate(
    candidates: List[Dict],
    existing_ids: set
) -> List[Dict]:
    """
    Removes tracks already in the user's library or the existing playlist.
    """
    seen = set()
    deduped = []
    for t in candidates:
        tid = t.get("track_id")
        if tid and tid not in existing_ids and tid not in seen:
            seen.add(tid)
            deduped.append(t)
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Source A: Spotify Related Artists  (replaces deprecated Recommendations API)
# ─────────────────────────────────────────────────────────────────────────────

def discover_via_related_artists(
    profile: ClusterProfile,
    existing_ids: set,
    max_related_artists: int = 20,
    tracks_per_artist: int = 5,
) -> List[Dict]:
    """
    Two-hop discovery using only non-deprecated Spotify endpoints:
      1. Take the cluster's representative track artists as seeds.
      2. Fetch sp.artist_related_artists() for each seed (not deprecated).
      3. Fetch sp.artist_top_tracks() for each related artist (not deprecated).

    This replaces sp.recommendations(), which was deprecated for new apps
    created after November 2024.

    Parameters
    ----------
    profile              : ClusterProfile for this cluster
    existing_ids         : track IDs already in the user's library (deduplication)
    max_related_artists  : cap on how many related artists to expand into
    tracks_per_artist    : top-N tracks to take per related artist
    """
    seed_artist_ids = list({
        t["artist_id"]
        for t in profile.representative_tracks[:5]
        if t.get("artist_id")
    })

    if not seed_artist_ids:
        logger.warning(
            f"Cluster {profile.cluster_id}: no seed artist IDs — skipping related artists"
        )
        return []

    # artists already represented in the cluster — don't re-surface their tracks
    cluster_artist_ids = {
        t.get("artist_id") for t in profile.representative_tracks
    }

    related_artist_ids: set = set()
    for artist_id in seed_artist_ids:
        try:
            result = sp.artist_related_artists(artist_id)
            for a in result.get("artists", [])[:10]:
                aid = a.get("id")
                if aid and aid not in cluster_artist_ids:
                    related_artist_ids.add(aid)
            time.sleep(0.1)
        except Exception as e:
            logger.warning(f"artist_related_artists failed for {artist_id}: {e}")

    if not related_artist_ids:
        logger.warning(f"Cluster {profile.cluster_id}: no related artists found")
        return []

    # cap to keep API call count reasonable
    related_list = list(related_artist_ids)[:max_related_artists]

    raw_tracks = []
    for artist_id in related_list:
        try:
            result = sp.artist_top_tracks(artist_id, country="US")
            raw_tracks.extend(result.get("tracks", [])[:tracks_per_artist])
            time.sleep(0.05)
        except Exception as e:
            logger.warning(f"artist_top_tracks failed for {artist_id}: {e}")

    normalized = [_normalize_track(t, "related_artists") for t in raw_tracks]
    enriched   = _enrich_with_audio_features(normalized)
    enriched   = _enrich_with_artist_genres(enriched)
    deduped    = _deduplicate(enriched, existing_ids)

    logger.info(
        f"Cluster {profile.cluster_id} related artists: "
        f"{len(seed_artist_ids)} seeds → {len(related_list)} related artists → "
        f"{len(raw_tracks)} raw tracks → {len(deduped)} after dedup"
    )
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Source B: New Releases
# ─────────────────────────────────────────────────────────────────────────────

def discover_via_new_releases(
    profile: ClusterProfile,
    existing_ids: set,
    max_albums: int = 20,
) -> List[Dict]:
    """
    Polls Spotify's New Releases endpoint for recent albums/EPs,
    then fetches their tracks. Filters by genre tag overlap with
    the cluster's top genres before enriching with audio features.

    This surfaces genuinely new music rather than Spotify's
    recommendation model recycling familiar artists.
    """
    try:
        results = sp.new_releases(limit=max_albums)
        albums  = results.get("albums", {}).get("items", [])

        if not albums:
            logger.warning("new_releases returned no albums")
            return []

        # gather all tracks from these albums
        raw_tracks = []
        for album in albums:
            try:
                album_tracks = sp.album_tracks(album["id"], limit=50)
                for t in album_tracks.get("items", []):
                    # inject album metadata that track objects don't carry
                    t["album"] = {"release_date": album.get("release_date", "")}
                    raw_tracks.append(t)
                time.sleep(0.05)
            except Exception as e:
                logger.warning(f"Could not fetch tracks for album {album['id']}: {e}")

        if not raw_tracks:
            return []

        normalized = [_normalize_track(t, "new_releases") for t in raw_tracks]
        enriched   = _enrich_with_audio_features(normalized)
        enriched   = _enrich_with_artist_genres(enriched)

        # genre filter: only keep tracks whose artist shares ≥1 genre tag
        # with this cluster. Without this filter, new releases would
        # include completely unrelated genres.
        cluster_genres = set(profile.top_genres)
        if cluster_genres:
            genre_filtered = [
                t for t in enriched
                if cluster_genres & set(t.get("artist_genres", []))
            ]
            logger.info(
                f"Cluster {profile.cluster_id} new releases: "
                f"{len(enriched)} tracks → {len(genre_filtered)} after genre filter"
            )
        else:
            genre_filtered = enriched
            logger.warning(
                f"Cluster {profile.cluster_id} has no genre tags — skipping genre filter"
            )

        deduped = _deduplicate(genre_filtered, existing_ids)
        return deduped

    except Exception as e:
        logger.error(f"new_releases discovery failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Source B: Deezer Artist Radio
# ─────────────────────────────────────────────────────────────────────────────

def _deezer_artist_id(artist_name: str) -> Optional[int]:
    """
    Looks up a Deezer artist ID by name.
    Returns None if the artist genuinely has no match.
    Raises on network errors or non-2xx HTTP responses so the caller can log them.
    """
    r = requests.get(
        f"{_DEEZER_BASE}/search/artist",
        params={"q": artist_name, "limit": 1},
        timeout=5,
    )
    r.raise_for_status()
    items = r.json().get("data", [])
    return items[0]["id"] if items else None


def discover_via_deezer(
    profile: ClusterProfile,
    existing_ids: set,
    radio_limit: int = 40,
) -> List[Dict]:
    """
    Deezer artist radio discovery:
      1. Map each cluster seed's artist name → Deezer artist ID (free, no key needed).
      2. Pull Deezer artist radio — similar track suggestions from Deezer's graph.
      3. Search Spotify for each suggestion to get a canonical Spotify track ID.
      4. Inject Deezer BPM into tempo_bpm when available, as a head-start before
         the LLM audio feature estimation step in _enrich_with_audio_features.

    Parameters
    ----------
    profile      : ClusterProfile for this cluster
    existing_ids : track IDs already in the user's library (deduplication)
    radio_limit  : number of radio tracks to request per seed artist
    """
    deezer_suggestions: Dict[str, Dict] = {}  # keyed by "title::artist" to deduplicate

    for seed in profile.representative_tracks[:3]:
        artist_name = seed.get("artist")
        if not artist_name:
            continue

        try:
            deezer_id = _deezer_artist_id(artist_name)
        except Exception as e:
            logger.warning(f"Deezer artist lookup failed for '{artist_name}': {e}")
            time.sleep(0.1)
            continue

        if not deezer_id:
            logger.debug(f"Deezer: no match for artist '{artist_name}'")
            continue

        try:
            r = requests.get(
                f"{_DEEZER_BASE}/artist/{deezer_id}/radio",
                params={"limit": radio_limit},
                timeout=5,
            )
            r.raise_for_status()
            for dt in r.json().get("data", []):
                title  = dt.get("title", "").strip()
                artist = dt.get("artist", {}).get("name", "").strip()
                if not title or not artist:
                    continue
                key = f"{title.lower()}::{artist.lower()}"
                if key not in deezer_suggestions:
                    deezer_suggestions[key] = {
                        "title":  title,
                        "artist": artist,
                        "bpm":    dt.get("bpm") or 0,  # 0 = Deezer doesn't know
                    }
            time.sleep(0.2)
        except Exception as e:
            logger.warning(f"Deezer radio failed for artist '{artist_name}' (id={deezer_id}): {e}")

    if not deezer_suggestions:
        return []

    # Search Spotify for each Deezer suggestion to get a Spotify track object
    candidates = []
    for info in deezer_suggestions.values():
        try:
            results = sp.search(
                q=f'track:"{info["title"]}" artist:"{info["artist"]}"',
                type="track",
                limit=1,
            )
            items = results.get("tracks", {}).get("items", [])
            if not items:
                continue

            normalized = _normalize_track(items[0], "deezer_radio")

            # Pre-fill BPM from Deezer; _enrich_with_audio_features will
            # overwrite with Spotify/LLM data if available, or keep this as fallback
            if info["bpm"] > 0:
                normalized["tempo_bpm"]  = float(info["bpm"])
                normalized["tempo_norm"] = float(np.clip((info["bpm"] - 60) / 140, 0, 1))

            candidates.append(normalized)
            time.sleep(0.05)
        except Exception as e:
            logger.debug(f"Spotify search failed for '{info['title']}': {e}")

    enriched = _enrich_with_audio_features(candidates)
    enriched = _enrich_with_artist_genres(enriched)
    deduped  = _deduplicate(enriched, existing_ids)

    logger.info(
        f"Cluster {profile.cluster_id} Deezer radio: "
        f"{len(deezer_suggestions)} suggestions → {len(deduped)} after Spotify match + dedup"
    )
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Main discovery pipeline
# ─────────────────────────────────────────────────────────────────────────────

def get_existing_track_ids(df: pd.DataFrame) -> set:
    """Returns the set of track IDs already in the user's library."""
    return set(df["track_id"].dropna().tolist())


def run_discovery(
    profiles: List[ClusterProfile],
    df: pd.DataFrame,
) -> Dict[int, List[Dict]]:
    """
    Runs all three discovery sources for every cluster profile.

    Sources
    -------
    A) related_artists : Spotify non-deprecated two-hop (replaces recommendations)
    B) deezer_radio    : Deezer artist radio → matched back to Spotify
    C) new_releases    : Spotify new releases filtered by cluster genre tags

    Returns
    -------
    {cluster_id: [candidate track dicts]}
    Each dict has audio features and artist genres populated,
    ready to be passed to scorer.py.
    """
    logger.info("=== Starting discovery pipeline ===")

    existing_ids   = get_existing_track_ids(df)
    all_candidates: Dict[int, List[Dict]] = {}

    for profile in profiles:
        logger.info(
            f"Discovering for cluster {profile.cluster_id} '{profile.label}' "
            f"({profile.size} tracks)"
        )

        # Source A: Spotify related artists (replaces deprecated recommendations)
        related_candidates = discover_via_related_artists(profile, existing_ids)

        # Source B: Deezer artist radio
        deezer_candidates = discover_via_deezer(profile, existing_ids)

        # Source C: New releases
        new_candidates = discover_via_new_releases(profile, existing_ids)

        # Merge and deduplicate across all three sources
        combined = _deduplicate(
            related_candidates + deezer_candidates + new_candidates,
            existing_ids,
        )

        all_candidates[profile.cluster_id] = combined

        logger.info(
            f"Cluster {profile.cluster_id}: "
            f"{len(related_candidates)} related_artists + "
            f"{len(deezer_candidates)} deezer + "
            f"{len(new_candidates)} new_releases = "
            f"{len(combined)} unique candidates"
        )

        time.sleep(0.2)  # gentle on rate limits between clusters

    total = sum(len(c) for c in all_candidates.values())
    logger.info(
        f"=== Discovery complete: {total} total candidates "
        f"across {len(all_candidates)} clusters ==="
    )
    return all_candidates
