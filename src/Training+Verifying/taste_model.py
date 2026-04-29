"""
taste_model.py
==============
Stage 2: Taste Modeling

Takes your processed DataFrame + feature matrix and computes a
weighted feature centroid representing your overall taste profile.

This replaces the old manual UserProfile(mood="sad", energy=0.9)
with one that is inferred entirely from your real listening data.

The centroid is also used as a fallback profile for clusters that
are too small to compute their own centroid reliably.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterProfile:
    """
    Replaces the old UserProfile dataclass.
    Instead of hand-specified fields (mood, genre, likes_acoustic),
    the profile is a dense feature vector learned from your data.

    Fields
    ------
    cluster_id      : int  — HDBSCAN cluster label (-1 = noise/outlier)
    centroid        : np.ndarray — mean feature vector for this cluster
    size            : int  — number of tracks in this cluster
    signal_weight   : float — average signal weight of member tracks
                              (higher = cluster made of liked/top songs)
    representative_tracks : list of dicts — top 5 tracks closest to centroid,
                            used as seed tracks for Spotify discovery
    label           : str  — auto-generated vibe label based on audio features
                             e.g. "High Energy / Low Acoustic"
    feature_names   : list[str] — names matching centroid vector positions
    """
    cluster_id:            int
    centroid:              np.ndarray
    size:                  int
    signal_weight:         float
    representative_tracks: List[dict] = field(default_factory=list)
    label:                 str        = ""
    feature_names:         List[str]  = field(default_factory=list)

    # audio summary fields — populated by _summarize_audio()
    avg_energy:        float = 0.0
    avg_valence:       float = 0.0
    avg_danceability:  float = 0.0
    avg_acousticness:  float = 0.0
    avg_tempo:         float = 0.0
    top_genres:        List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Core taste modeling functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_centroid(
    X: np.ndarray,
    df: pd.DataFrame
) -> np.ndarray:
    """
    Computes a single weighted centroid across ALL your tracks.
    Used as your global taste profile and as a fallback for
    clusters that are too small.

    Weighting: signal_weight column from feature engineering
    (liked songs and recent top tracks count more).
    """
    weights = df["signal_weight"].values.astype(float)

    # guard: if weights sum to zero (no signal data), use uniform weights
    if weights.sum() == 0:
        logger.warning("All signal weights are zero — using uniform weights for centroid")
        weights = np.ones(len(weights))

    # weighted mean across all tracks
    weighted_centroid = np.average(X, axis=0, weights=weights)
    logger.info(f"Global centroid computed from {len(X)} tracks")
    return weighted_centroid


def _auto_label(profile: ClusterProfile) -> str:
    """
    Generates a human-readable vibe label from a cluster's
    average audio features. Used for logging and the guardrail prompt.
    """
    parts = []

    # energy axis
    if profile.avg_energy >= 0.7:
        parts.append("High Energy")
    elif profile.avg_energy <= 0.35:
        parts.append("Low Energy")
    else:
        parts.append("Mid Energy")

    # mood axis (valence)
    if profile.avg_valence >= 0.65:
        parts.append("Upbeat")
    elif profile.avg_valence <= 0.35:
        parts.append("Melancholic")
    else:
        parts.append("Neutral Mood")

    # texture axis
    if profile.avg_acousticness >= 0.6:
        parts.append("Acoustic")
    elif profile.avg_danceability >= 0.7:
        parts.append("Dance")
    else:
        parts.append("Electronic / Produced")

    return " · ".join(parts)


def _summarize_audio(
    profile: ClusterProfile,
    df: pd.DataFrame,
    member_indices: np.ndarray,
    feature_names: List[str]
) -> ClusterProfile:
    """
    Populates the human-readable audio summary fields on a ClusterProfile
    by averaging the raw DataFrame columns for cluster members.
    Also extracts the top 3 genre tags across the cluster.
    """
    members = df.iloc[member_indices]

    profile.avg_energy       = float(members["energy"].mean())
    profile.avg_valence      = float(members["valence"].mean())
    profile.avg_danceability = float(members["danceability"].mean())
    profile.avg_acousticness = float(members["acousticness"].mean())
    profile.avg_tempo        = float(members["tempo_bpm"].fillna(120).mean())

    # top genre tags: explode the list column and take most frequent 3
    if "artist_genres" in members.columns:
        all_genres = []
        for g in members["artist_genres"]:
            if isinstance(g, list):
                all_genres.extend(g)
        if all_genres:
            from collections import Counter
            profile.top_genres = [g for g, _ in Counter(all_genres).most_common(3)]

    profile.label = _auto_label(profile)
    return profile


def _find_representative_tracks(
    centroid: np.ndarray,
    X: np.ndarray,
    df: pd.DataFrame,
    member_indices: np.ndarray,
    n: int = 5
) -> List[dict]:
    """
    Finds the n tracks in a cluster that are closest to the centroid
    using Euclidean distance. These become seed tracks for Spotify
    Recommendations API calls in the discovery stage.
    """
    member_X = X[member_indices]
    distances = np.linalg.norm(member_X - centroid, axis=1)
    closest_local_indices = np.argsort(distances)[:n]
    closest_global_indices = member_indices[closest_local_indices]

    tracks = []
    for idx in closest_global_indices:
        row = df.iloc[idx]
        tracks.append({
            "track_id":  row.get("track_id", ""),
            "title":     row.get("title", ""),
            "artist":    row.get("artist", ""),
            "energy":    row.get("energy"),
            "valence":   row.get("valence"),
            "distance_to_centroid": float(distances[closest_local_indices[len(tracks)]])
            if len(tracks) < len(closest_local_indices) else 0.0,
        })
    return tracks


def build_cluster_profiles(
    X: np.ndarray,
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_names: List[str],
    min_cluster_size: int = 5
) -> List[ClusterProfile]:
    """
    Main function: given HDBSCAN cluster labels, builds a ClusterProfile
    for each valid cluster (label >= 0).

    Clusters with fewer than min_cluster_size tracks are skipped —
    too small to represent a real vibe. Noise points (label == -1)
    are also excluded from profiles but logged.

    Returns a list of ClusterProfile objects sorted by size descending.
    """
    unique_labels = set(cluster_labels)
    noise_count = int(np.sum(cluster_labels == -1))
    logger.info(
        f"Building profiles for {len(unique_labels) - (1 if -1 in unique_labels else 0)} "
        f"clusters. Noise points: {noise_count}"
    )

    profiles = []

    for label in sorted(unique_labels):
        if label == -1:
            continue   # skip noise

        member_indices = np.where(cluster_labels == label)[0]

        if len(member_indices) < min_cluster_size:
            logger.warning(
                f"Cluster {label} has only {len(member_indices)} tracks "
                f"(min={min_cluster_size}) — skipping"
            )
            continue

        member_X = X[member_indices]
        member_df = df.iloc[member_indices]

        # weighted centroid for this cluster
        weights = member_df["signal_weight"].values.astype(float)
        if weights.sum() == 0:
            weights = np.ones(len(weights))
        centroid = np.average(member_X, axis=0, weights=weights)

        avg_signal = float(member_df["signal_weight"].mean())

        profile = ClusterProfile(
            cluster_id=int(label),
            centroid=centroid,
            size=len(member_indices),
            signal_weight=avg_signal,
            feature_names=feature_names,
        )

        # enrich with audio summaries and representative tracks
        profile = _summarize_audio(profile, df, member_indices, feature_names)
        profile.representative_tracks = _find_representative_tracks(
            centroid, X, df, member_indices
        )

        profiles.append(profile)
        logger.info(
            f"Cluster {label}: {len(member_indices)} tracks | "
            f"label='{profile.label}' | "
            f"energy={profile.avg_energy:.2f} valence={profile.avg_valence:.2f}"
        )

    # sort by cluster size so the largest/most representative comes first
    profiles.sort(key=lambda p: p.size, reverse=True)
    return profiles


def run_taste_model(
    X: np.ndarray,
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_names: List[str],
) -> tuple[np.ndarray, List[ClusterProfile]]:
    """
    Full taste modeling pipeline.

    Returns
    -------
    global_centroid : np.ndarray  — your overall taste vector
    profiles        : list[ClusterProfile] — one profile per vibe cluster
    """
    logger.info("=== Starting taste modeling ===")

    global_centroid = compute_global_centroid(X, df)
    profiles = build_cluster_profiles(X, df, cluster_labels, feature_names)

    logger.info(
        f"Taste model complete: {len(profiles)} cluster profiles built"
    )
    return global_centroid, profiles
