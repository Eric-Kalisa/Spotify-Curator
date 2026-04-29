import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

"""
This module transforms raw Spotify song records into a feature matrix suitable for UMAP dimensionality reduction.
It includes:
- Recency weighting: More recent plays get higher weight.
- Source weighting: Differentiate between liked songs, top tracks, and recently played. 
- Audio feature normalization: Scale features like tempo and loudness to 0-1.
- Categorical encoding: Convert genres into multi-hot vectors.
The goal is to create a rich, balanced feature set that captures both the audio characteristics and the implicit signals of user preference.
"""
# Source signals reflect how strong an implicit positive signal each source is
SOURCE_WEIGHTS = {
    "liked":              1.0,   # strongest — you explicitly saved it
    "top_short_term":     0.9,   # what you're listening to right now
    "top_medium_term":    0.7,   # last 6 months
    "top_long_term":      0.5,   # all time but may be outdated
    "recently_played":    0.4,   # could be background listening
}

def build_dataframe(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    # drop rows with no audio features — can't use them
    audio_cols = ["energy", "valence", "danceability", "acousticness"]
    df = df.dropna(subset=audio_cols)
    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw fields into model-ready features.
    """

    # ── Recency weight ──────────────────────────────────────
    # More recently played = higher weight
    # We normalize played_at to a 0-1 recency score
    df["played_at_dt"] = pd.to_datetime(df["played_at"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    max_age_days = 365

    df["recency_weight"] = df["played_at_dt"].apply(
        lambda dt: max(0, 1 - (now - dt).days / max_age_days)
        if pd.notnull(dt) else 0.3   # default if no timestamp
    )

    # ── Source weight ────────────────────────────────────────
    df["source_weight"] = df["source"].map(SOURCE_WEIGHTS).fillna(0.3)

    # ── Combined signal weight ───────────────────────────────
    df["signal_weight"] = (df["recency_weight"] + df["source_weight"]) / 2

    # ── Normalize tempo ──────────────────────────────────────
    # tempo_bpm can be 60-200, we scale to 0-1 to match other features
    df["tempo_norm"] = (df["tempo_bpm"].fillna(120) - 60) / (200 - 60)
    df["tempo_norm"] = df["tempo_norm"].clip(0, 1)

    # ── Normalize loudness ───────────────────────────────────
    # loudness is -60 to 0 dB, scale to 0-1
    df["loudness_norm"] = (df["loudness"].fillna(-30) + 60) / 60
    df["loudness_norm"] = df["loudness_norm"].clip(0, 1)

    # ── Normalize release year ───────────────────────────────
    # gives a 0-1 era score (older = 0, newer = 1)
    min_year, max_year = 1950, 2025
    df["era_score"] = (df["release_year"].fillna(2000) - min_year) / (max_year - min_year)
    df["era_score"] = df["era_score"].clip(0, 1)

    # ── Normalize popularity ─────────────────────────────────
    df["popularity_norm"] = df["popularity"].fillna(50) / 100

    # ── Time-of-day encoding ─────────────────────────────────
    # encode hour as a cyclical feature so 23 and 0 are close
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"].fillna(12) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"].fillna(12) / 24)

    # ── Key + mode combined ──────────────────────────────────
    # encode as a single harmonic signature (0-1)
    df["harmonic_sig"] = (df["key"].fillna(0) / 11) * df["mode"].fillna(0.5)

    # ── Normalize time signature ─────────────────────────────
    # Spotify's time_signature ranges from 3 to 7 (3/4 to 7/4)
    df["time_sig_norm"] = (df["time_signature"].fillna(4) - 3) / (7 - 3)
    df["time_sig_norm"] = df["time_sig_norm"].clip(0, 1)

    # ── fill remaining nulls ─────────────────────────────────
    fill_defaults = {
        "speechiness": 0.0,
        "instrumentalness": 0.0,
        "liveness": 0.0,
    }
    df = df.fillna(fill_defaults)

    return df


def encode_genre_tags(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Multi-hot encodes artist_genres list into binary columns.
    e.g. ["indie pop", "dream pop"] → [0,1,0,1,0...]
    """
    mlb = MultiLabelBinarizer()

    # fill missing genre lists with empty list
    genre_lists = df["artist_genres"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    genre_encoded = mlb.fit_transform(genre_lists)
    genre_df = pd.DataFrame(
        genre_encoded,
        columns=[f"genre_{g.replace(' ', '_')}" for g in mlb.classes_]
    )
    genre_cols = list(genre_df.columns)

    df = pd.concat([df.reset_index(drop=True), genre_df], axis=1)
    return df, genre_cols


def build_feature_matrix(df: pd.DataFrame, genre_cols: list) -> np.ndarray:
    """
    Assembles the final feature matrix for UMAP input.
    Returns a numpy array of shape (n_songs, n_features).
    """
    # core continuous audio features
    audio_features = [
        "energy", "valence", "danceability", "acousticness",
        "tempo_norm", "loudness_norm", "speechiness",
        "instrumentalness", "liveness", "time_sig_norm",
    ]

    # engineered features
    engineered = [
        "harmonic_sig", "era_score", "popularity_norm",
        "hour_sin", "hour_cos",
    ]

    # genre multi-hot (weighted down so they don't dominate)
    # we scale genre columns by 0.3 to balance them against
    # continuous features — genres are sparse so raw they'd overwhelm
    base_cols = audio_features + engineered
    feature_matrix = df[base_cols].values.astype(float)

    if genre_cols:
        genre_matrix = df[genre_cols].values.astype(float) * 0.3
        feature_matrix = np.hstack([feature_matrix, genre_matrix])

    return feature_matrix


def run_feature_engineering(records: List[Dict]):
    """
    Full feature engineering pipeline.
    Returns df and feature matrix ready for UMAP.
    """
    df = build_dataframe(records)
    df = engineer_features(df)
    df, genre_cols = encode_genre_tags(df)
    X = build_feature_matrix(df, genre_cols)

    return df, X, genre_cols