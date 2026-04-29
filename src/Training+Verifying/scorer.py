"""
scorer.py
=========
Replaces the old hand-weighted _score_song() with cosine similarity
scoring against cluster centroids.

Why cosine similarity over the old approach:
- Old approach: hardcoded weights (genre=2.0, mood=1.0) that aren't
  learned from your data and ignore 8 of the 12 audio features.
- New approach: every feature contributes proportionally. Weights are
  implicit in your actual listening history via the centroid. A song
  scores high if its feature vector points in the same direction as
  your cluster's centroid — capturing the whole shape of your taste.

Confidence score (0–1):
  Raw cosine similarity ranges from -1 to 1. We map it to 0–1 and
  treat it as a confidence score. The guardrail stage uses this to
  decide whether a candidate track is good enough to add to a playlist.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass

from taste_model import ClusterProfile

logger = logging.getLogger(__name__)

# Tracks scoring below this threshold are rejected before the LLM guardrail.
# The LLM then reviews everything above this threshold.
CONFIDENCE_THRESHOLD = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredTrack:
    """A candidate track with its confidence score and explanation."""
    track_id:       str
    title:          str
    artist:         str
    cluster_id:     int
    confidence:     float          # 0–1, higher = better match
    raw_similarity: float          # raw cosine similarity (-1 to 1)
    explanation:    str
    audio_features: Dict           # raw feature values for the guardrail prompt
    passed_threshold: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector builder
# ─────────────────────────────────────────────────────────────────────────────

# These are the continuous audio features we use for cosine similarity.
# Must match the order used when building the feature matrix in features.py
AUDIO_FEATURE_COLS = [
    "energy", "valence", "danceability", "acousticness",
    "tempo_norm", "loudness_norm", "speechiness",
    "instrumentalness", "liveness", "time_sig_norm",
    "harmonic_sig", "era_score", "popularity_norm",
    "hour_sin", "hour_cos",
]


def track_to_vector(track: Dict, genre_cols: List[str]) -> np.ndarray:
    """
    Converts a track dict (from ingest or Spotify discovery) into
    the same feature vector format used to build X in features.py.

    Genre columns are weighted by 0.3 to match features.py scaling.
    Returns None if critical audio features are missing.
    """
    # validate minimum required features
    for col in ["energy", "valence", "danceability", "acousticness"]:
        if track.get(col) is None:
            logger.debug(f"Track {track.get('title')} missing '{col}' — skipping")
            return None

    # continuous audio features
    base_vector = []
    for col in AUDIO_FEATURE_COLS:
        val = track.get(col)
        if val is None:
            # use column-specific defaults for missing values
            defaults = {
                "tempo_norm": 0.5, "loudness_norm": 0.5,
                "speechiness": 0.05, "instrumentalness": 0.0,
                "liveness": 0.15, "time_sig_norm": 0.25,
                "harmonic_sig": 0.5, "era_score": 0.5,
                "popularity_norm": 0.5, "hour_sin": 0.0, "hour_cos": 1.0,
            }
            val = defaults.get(col, 0.0)
        base_vector.append(float(val))

    # genre multi-hot (weighted down by 0.3 to match features.py)
    if genre_cols:
        track_genres = set(track.get("artist_genres", []))
        genre_vector = [
            0.3 if col.replace("genre_", "").replace("_", " ") in track_genres else 0.0
            for col in genre_cols
        ]
        return np.array(base_vector + genre_vector)

    return np.array(base_vector)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Returns 0 if either vector is zero (avoids division by zero).
    Range: -1 to 1. For music features, almost always 0 to 1.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def similarity_to_confidence(raw_similarity: float) -> float:
    """
    Maps raw cosine similarity (-1 to 1) to a 0–1 confidence score.
    We clip at 0 since negative similarity means opposite taste.
    """
    return float(np.clip(raw_similarity, 0, 1))


def _build_explanation(
    track: Dict,
    profile: ClusterProfile,
    confidence: float
) -> str:
    """
    Builds a human-readable explanation for why a track scored the way it did.
    Used in logging and passed to the LLM guardrail for context.
    """
    reasons = []

    # audio feature comparisons against cluster centroid
    energy_diff = abs(track.get("energy", 0.5) - profile.avg_energy)
    valence_diff = abs(track.get("valence", 0.5) - profile.avg_valence)

    if energy_diff < 0.15:
        reasons.append(f"energy matches cluster avg ({track.get('energy', 0):.2f} ≈ {profile.avg_energy:.2f})")
    elif track.get("energy", 0.5) > profile.avg_energy:
        reasons.append(f"more energetic than cluster avg ({track.get('energy', 0):.2f} vs {profile.avg_energy:.2f})")
    else:
        reasons.append(f"less energetic than cluster avg ({track.get('energy', 0):.2f} vs {profile.avg_energy:.2f})")

    if valence_diff < 0.15:
        reasons.append(f"mood matches cluster avg ({track.get('valence', 0):.2f} ≈ {profile.avg_valence:.2f})")

    # genre overlap
    track_genres = set(track.get("artist_genres", []))
    cluster_genres = set(profile.top_genres)
    overlap = track_genres & cluster_genres
    if overlap:
        reasons.append(f"shares genre tags: {', '.join(list(overlap)[:2])}")

    confidence_pct = int(confidence * 100)
    reason_str = "; ".join(reasons) if reasons else "overall feature similarity"
    return (
        f"Confidence {confidence_pct}% — matched cluster '{profile.label}' because: {reason_str}."
    )


def score_candidates(
    candidates: List[Dict],
    profile: ClusterProfile,
    genre_cols: List[str],
) -> List[ScoredTrack]:
    """
    Scores a list of candidate tracks against a single ClusterProfile
    using cosine similarity.

    Returns a sorted list of ScoredTrack, best first.
    Tracks that fail to build a feature vector are skipped with a warning.
    """
    scored = []

    for track in candidates:
        vec = track_to_vector(track, genre_cols)
        if vec is None:
            logger.warning(
                f"Skipping '{track.get('title')}' — could not build feature vector"
            )
            continue

        raw_sim     = cosine_similarity(vec, profile.centroid)
        confidence  = similarity_to_confidence(raw_sim)
        explanation = _build_explanation(track, profile, confidence)
        passed      = confidence >= CONFIDENCE_THRESHOLD

        scored.append(ScoredTrack(
            track_id=       track.get("track_id", track.get("id", "")),
            title=          track.get("title",   track.get("name", "")),
            artist=         track.get("artist",  ""),
            cluster_id=     profile.cluster_id,
            confidence=     confidence,
            raw_similarity= raw_sim,
            explanation=    explanation,
            passed_threshold= passed,
            audio_features= {
                "energy":        track.get("energy"),
                "valence":       track.get("valence"),
                "danceability":  track.get("danceability"),
                "acousticness":  track.get("acousticness"),
                "tempo_bpm":     track.get("tempo_bpm"),
                "genres":        track.get("artist_genres", []),
            }
        ))

    scored.sort(key=lambda s: s.confidence, reverse=True)

    passed_count = sum(1 for s in scored if s.passed_threshold)
    logger.info(
        f"Cluster {profile.cluster_id} '{profile.label}': "
        f"scored {len(scored)} candidates | "
        f"{passed_count} above threshold ({CONFIDENCE_THRESHOLD})"
    )
    return scored


def score_all_clusters(
    candidates_by_cluster: Dict[int, List[Dict]],
    profiles: List[ClusterProfile],
    genre_cols: List[str],
) -> Dict[int, List[ScoredTrack]]:
    """
    Scores candidates for every cluster profile.

    Parameters
    ----------
    candidates_by_cluster : {cluster_id: [track dicts]} from discoverer.py
    profiles              : list of ClusterProfile from taste_model.py
    genre_cols            : genre column names from features.py

    Returns
    -------
    {cluster_id: [ScoredTrack sorted by confidence desc]}
    """
    profile_map = {p.cluster_id: p for p in profiles}
    results = {}

    for cluster_id, candidates in candidates_by_cluster.items():
        if cluster_id not in profile_map:
            logger.warning(f"No profile found for cluster {cluster_id} — skipping")
            continue

        profile = profile_map[cluster_id]
        scored  = score_candidates(candidates, profile, genre_cols)
        results[cluster_id] = scored

    total_passed = sum(
        sum(1 for s in tracks if s.passed_threshold)
        for tracks in results.values()
    )
    logger.info(
        f"Scoring complete: {total_passed} tracks passed threshold "
        f"across {len(results)} clusters"
    )
    return results
