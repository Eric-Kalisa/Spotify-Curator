import time
import logging
from typing import List, Dict
from datetime import datetime
from auth_spotify import get_spotify_client
from llm_features import estimate_audio_features, estimate_artist_genres

"""
Ingest pipeline: Spotify listening data → flat records ready for feature engineering.

Pipeline stages:
1. Raw Track Collectors — fetch tracks from recently_played, liked, and top endpoints
2. Assembly             — merge tracks + LLM audio features + LLM genre tags into flat dicts
3. Main Pipeline        — orchestrate collection, LLM enrichment, and assembly
"""

logging.basicConfig(
    filename="logs/ingest.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

sp = get_spotify_client()


# ─────────────────────────────────────────────
# SECTION 1: RAW TRACK COLLECTORS
# Each function returns a list of raw dicts
# with a 'source' tag so we know where it came from
# ─────────────────────────────────────────────

def fetch_recently_played(limit: int = 50) -> List[Dict]:
    """
    Pulls your last 50 played tracks with timestamps.
    50 is the API hard limit per call — we paginate using
    the 'before' cursor to go further back.
    """
    tracks = []
    try:
        results = sp.current_user_recently_played(limit=50)
        while results:
            for item in results["items"]:
                track = item["track"]
                track["played_at"] = item["played_at"]
                track["source"] = "recently_played"
                tracks.append(track)

            if results["next"] and len(tracks) < limit:
                before = results["cursors"]["before"]
                results = sp.current_user_recently_played(limit=50, before=before)
            else:
                break

        logging.info(f"Fetched {len(tracks)} recently played tracks")
    except Exception as e:
        logging.error(f"Error fetching recently played: {e}")

    return tracks


def fetch_liked_songs(limit: int = 500) -> List[Dict]:
    """
    Pulls your saved/liked tracks library.
    API returns max 50 per page so we paginate with offset.
    Liked songs are a strong explicit positive signal.
    """
    tracks = []
    offset = 0
    try:
        while len(tracks) < limit:
            results = sp.current_user_saved_tracks(limit=50, offset=offset)
            if not results["items"]:
                break
            for item in results["items"]:
                track = item["track"]
                track["saved_at"] = item["added_at"]
                track["source"] = "liked"
                tracks.append(track)

            offset += 50
            time.sleep(0.1)

        logging.info(f"Fetched {len(tracks)} liked songs")
    except Exception as e:
        logging.error(f"Error fetching liked songs: {e}")

    return tracks


def fetch_top_tracks() -> List[Dict]:
    """
    Fetches your top tracks across 3 time windows.
    Comparing short vs long term lets us detect taste drift later.
    """
    tracks = []
    time_ranges = ["short_term", "medium_term", "long_term"]
    # short  = last 4 weeks
    # medium = last 6 months
    # long   = all time

    try:
        for time_range in time_ranges:
            results = sp.current_user_top_tracks(limit=50, time_range=time_range)
            for item in results["items"]:
                item["source"] = f"top_{time_range}"
                tracks.append(item)
            time.sleep(0.1)

        logging.info(f"Fetched {len(tracks)} top tracks across all time ranges")
    except Exception as e:
        logging.error(f"Error fetching top tracks: {e}")

    return tracks


# ─────────────────────────────────────────────
# SECTION 2: ASSEMBLY
# Merges everything into one clean record per track
# ─────────────────────────────────────────────

def assemble_track_records(
    raw_tracks: List[Dict],
    audio_features_map: Dict[str, Dict],
    genres_map: Dict[str, List],
) -> List[Dict]:
    """
    Joins raw track data + LLM audio features + LLM genre tags
    into one flat dict per track. This is your training record.
    """
    records = []
    seen_ids = set()

    for track in raw_tracks:
        track_id = track.get("id")
        if not track_id or track_id in seen_ids:
            continue
        seen_ids.add(track_id)

        features  = audio_features_map.get(track_id, {})
        artist_id = track["artists"][0]["id"] if track.get("artists") else None

        played_at    = track.get("played_at") or track.get("saved_at")
        hour_of_day  = None
        release_year = None

        if played_at:
            try:
                dt = datetime.fromisoformat(played_at.replace("Z", "+00:00"))
                hour_of_day = dt.hour
            except Exception:
                pass

        release_date = track.get("album", {}).get("release_date", "")
        if release_date:
            try:
                release_year = int(release_date[:4])
            except Exception:
                pass

        record = {
            # identifiers
            "track_id":         track_id,
            "title":            track.get("name"),
            "artist":           track["artists"][0]["name"] if track.get("artists") else None,
            "artist_id":        artist_id,
            "source":           track.get("source"),

            # audio features — LLM estimated (layer 1)
            "energy":           features.get("energy"),
            "valence":          features.get("valence"),
            "danceability":     features.get("danceability"),
            "acousticness":     features.get("acousticness"),
            "tempo_bpm":        features.get("tempo"),
            "loudness":         features.get("loudness"),
            "speechiness":      features.get("speechiness"),
            "instrumentalness": features.get("instrumentalness"),
            "liveness":         features.get("liveness"),
            "key":              features.get("key"),
            "mode":             features.get("mode"),
            "time_signature":   features.get("time_signature"),

            # track metadata (layer 2)
            "popularity":       track.get("popularity"),
            "explicit":         track.get("explicit"),
            "duration_ms":      track.get("duration_ms"),
            "release_year":     release_year,

            # genre tags — LLM estimated (layer 3)
            "artist_genres":    genres_map.get(artist_id, []),

            # behavioral signals (layer 4)
            "hour_of_day":      hour_of_day,
            "played_at":        played_at,
        }
        records.append(record)

    logging.info(f"Assembled {len(records)} unique track records")
    return records


# ─────────────────────────────────────────────
# SECTION 3: MAIN INGEST PIPELINE
# ─────────────────────────────────────────────

def run_ingest() -> List[Dict]:
    """
    Runs the full ingestion pipeline and returns
    assembled records ready for feature engineering.
    """
    logging.info("=== Starting ingest pipeline ===")

    # 1. collect raw tracks from all sources
    raw_tracks = (
        fetch_recently_played(limit=200) +
        fetch_liked_songs(limit=500) +
        fetch_top_tracks()
    )
    logging.info(f"Total raw tracks collected: {len(raw_tracks)}")

    # 2. build unique artist list with sample track titles for LLM context
    seen_artists: Dict[str, Dict] = {}
    for t in raw_tracks:
        if not t.get("artists"):
            continue
        a   = t["artists"][0]
        aid = a.get("id")
        if not aid:
            continue
        if aid not in seen_artists:
            seen_artists[aid] = {"id": aid, "name": a.get("name", ""), "track_names": []}
        seen_artists[aid]["track_names"].append(t["name"])

    for a in seen_artists.values():
        a["track_names"] = a["track_names"][:3]

    # 3. estimate features via Bedrock
    audio_features_map = estimate_audio_features(raw_tracks)
    genres_map         = estimate_artist_genres(list(seen_artists.values()))

    # 4. assemble into flat records
    records = assemble_track_records(raw_tracks, audio_features_map, genres_map)

    logging.info("=== Ingest pipeline complete ===")
    return records
