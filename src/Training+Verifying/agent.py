"""
agent.py
========
Orchestrator: runs the full pipeline in order.

  Stage 1 → ingest.py        : pull Spotify data
  Stage 2 → features.py      : engineer feature matrix X
  Stage 3 → segmenter.py     : UMAP + HDBSCAN → cluster labels
  Stage 4 → taste_model.py   : build ClusterProfiles from clusters
  Stage 5 → discoverer.py    : find candidate tracks per cluster
  Stage 6 → scorer.py        : cosine similarity scoring
  Stage 7 → guardrail.py     : LLM self-critique review
  Stage 8 → (playlist_writer): write approved tracks to Spotify [next step]

Run with:
    python agent.py
    python agent.py --skip-ingest   # reuse cached data/raw_records.json
"""

import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# pipeline modules
from ingest import run_ingest
from features import run_feature_engineering
from segmenter import run_segmentation
from taste_model import run_taste_model
from discoverer import run_discovery
from scorer import score_all_clusters
from guardrail import run_guardrail

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(f"logs/run_{run_id}.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("agent")


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers (avoid re-fetching during development)
# ─────────────────────────────────────────────────────────────────────────────

def save_cache(records, df, X, cluster_labels, genre_cols):
    np.save("data/X.npy", X)
    np.save("data/cluster_labels.npy", cluster_labels)
    df.to_csv("data/processed_tracks.csv", index=False)
    with open("data/raw_records.json", "w") as f:
        json.dump(records, f, default=str)
    with open("data/genre_cols.json", "w") as f:
        json.dump(genre_cols, f)
    logger.info("Pipeline data cached to data/")


def load_cache():
    X             = np.load("data/X.npy")
    cluster_labels = np.load("data/cluster_labels.npy")
    df            = pd.read_csv("data/processed_tracks.csv")
    with open("data/raw_records.json") as f:
        records = json.load(f)
    with open("data/genre_cols.json") as f:
        genre_cols = json.load(f)

    # re-parse artist_genres from string back to list
    if "artist_genres" in df.columns:
        import ast
        def _parse_genres(x):
            if not isinstance(x, str):
                return []
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return []
        df["artist_genres"] = df["artist_genres"].apply(_parse_genres)

    logger.info(f"Loaded cached data: {len(records)} records, X={X.shape}")
    return records, df, X, cluster_labels, genre_cols


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages (each wrapped with error handling + timing)
# ─────────────────────────────────────────────────────────────────────────────

def stage(name: str):
    """Decorator-like context for logging stage start/end/duration."""
    class _Stage:
        def __enter__(self):
            self.start = datetime.now()
            print(f"\n{'─'*60}")
            print(f"  Stage: {name}")
            print(f"{'─'*60}")
            logger.info(f"START: {name}")
            return self

        def __exit__(self, exc_type, exc_val, _):
            elapsed = (datetime.now() - self.start).total_seconds()
            if exc_type:
                logger.error(f"FAILED: {name} after {elapsed:.1f}s — {exc_val}")
                print(f"\n  ✗ {name} FAILED: {exc_val}")
                return False  # re-raise
            logger.info(f"DONE: {name} ({elapsed:.1f}s)")
            print(f"\n  ✓ {name} complete ({elapsed:.1f}s)")
    return _Stage()


# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(skip_ingest: bool = False, min_cluster_size: int = 10):
    print("\n" + "=" * 60)
    print("  SPOTIFY CURATOR AGENT")
    print(f"  Run ID: {run_id}")
    print("=" * 60)

    # ── Stage 1+2: Ingest + Feature Engineering ──────────────────────────────
    if skip_ingest and Path("data/X.npy").exists():
        with stage("Load cached data"):
            records, df, X, _, genre_cols = load_cache()
    else:
        with stage("Stage 1: Ingest from Spotify"):
            records = run_ingest()
            print(f"  Collected {len(records)} tracks across all sources")

        with stage("Stage 2: Feature Engineering"):
            df, X, genre_cols = run_feature_engineering(records)
            print(f"  Feature matrix: {X.shape[0]} tracks × {X.shape[1]} features")

    # ── Stage 3: Segmentation ─────────────────────────────────────────────────
    with stage("Stage 3: UMAP + HDBSCAN Segmentation"):
        _, cluster_labels, _ = run_segmentation(
            X, df, min_cluster_size=min_cluster_size
        )

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        if n_clusters == 0:
            logger.error("No clusters found — cannot continue. Try a smaller min_cluster_size.")
            sys.exit(1)

    # ── Stage 4: Taste Model ──────────────────────────────────────────────────
    # feature_names for taste_model — the base continuous features
    feature_names = [
        "energy", "valence", "danceability", "acousticness",
        "tempo_norm", "loudness_norm", "speechiness",
        "instrumentalness", "liveness", "time_sig_norm",
        "harmonic_sig", "era_score", "popularity_norm", "hour_sin", "hour_cos",
    ] + genre_cols

    with stage("Stage 4: Taste Modeling"):
        _, profiles = run_taste_model(
            X, df, cluster_labels, feature_names
        )
        print(f"  Built {len(profiles)} cluster profiles:")
        for p in profiles:
            print(f"    Cluster {p.cluster_id}: '{p.label}' ({p.size} tracks)")

    # save cache after stages 1-4 complete
    save_cache(records, df, X, cluster_labels, genre_cols)

    # ── Stage 5: Discovery ────────────────────────────────────────────────────
    with stage("Stage 5: Discovery (Recommendations + New Releases)"):
        candidates_by_cluster = run_discovery(profiles, df)
        total_candidates = sum(len(c) for c in candidates_by_cluster.values())
        print(f"  Found {total_candidates} total candidates across {len(candidates_by_cluster)} clusters")

    # ── Stage 6: Scoring ──────────────────────────────────────────────────────
    with stage("Stage 6: Cosine Similarity Scoring"):
        scored_by_cluster = score_all_clusters(
            candidates_by_cluster, profiles, genre_cols
        )
        for cluster_id, scored in scored_by_cluster.items():
            passed = sum(1 for s in scored if s.passed_threshold)
            print(f"  Cluster {cluster_id}: {passed}/{len(scored)} candidates passed threshold")

    # ── Stage 7: LLM Guardrail ────────────────────────────────────────────────
    with stage("Stage 7: LLM Self-Critique Guardrail"):
        guardrail_reports = run_guardrail(profiles, scored_by_cluster)
        total_approved = sum(len(r.approved) for r in guardrail_reports.values())
        total_rejected = sum(len(r.rejected) for r in guardrail_reports.values())
        print(f"\n  Final: {total_approved} tracks approved, {total_rejected} rejected")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Clusters:          {len(profiles)}")
    print(f"  Candidates found:  {total_candidates}")
    print(f"  Passed scoring:    {sum(sum(1 for s in sc if s.passed_threshold) for sc in scored_by_cluster.values())}")
    print(f"  Approved by LLM:   {total_approved}")
    print(f"  Rejected by LLM:   {total_rejected}")
    print(f"  Log:               logs/run_{run_id}.log")
    print("=" * 60)
    print("\n  Ready for Stage 8: playlist_writer.py\n")

    return guardrail_reports, profiles


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spotify Curator Agent")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip Spotify API calls and reuse cached data/raw_records.json",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Minimum tracks to form a cluster (default: 10)",
    )
    args = parser.parse_args()

    run_agent(
        skip_ingest=args.skip_ingest,
        min_cluster_size=args.min_cluster_size,
    )
