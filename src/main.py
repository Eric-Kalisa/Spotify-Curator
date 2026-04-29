"""
main.py
=======
Demo runner for the Spotify Curator pipeline.

Runs every stage — ingest, feature engineering, segmentation, taste modeling,
discovery, scoring, and guardrail — then prints a formatted summary of each
stage's output to the terminal.  Nothing is written to Spotify.

Run with:
    python src/main.py                   # full run (Spotify API + Bedrock)
    python src/main.py --skip-ingest     # reuse cached data/raw_records.json
    python src/main.py --skip-guardrail  # skip the Bedrock guardrail call
"""

import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # src/
_ROOT = _HERE.parent                              # project root
_DATA = _ROOT / "data"

sys.path.insert(0, str(_HERE / "Preprocessing"))
sys.path.insert(0, str(_HERE / "Training+Verifying"))
sys.path.insert(0, str(_ROOT / "Service_authentications"))

# silence pipeline-internal logs so only our formatted output reaches the terminal
logging.disable(logging.CRITICAL)

# ── console layout ────────────────────────────────────────────────────────────
W = 74


def _rule(char="─"):
    print("  " + char * (W - 2))


def _header(title):
    print()
    print("  " + "═" * (W - 2))
    pad = (W - 2 - len(title)) // 2
    print(" " * (pad + 2) + title)
    print("  " + "═" * (W - 2))


def _section(label):
    print(f"\n  ▸  {label}")
    _rule()


def _bar(val, width=16):
    if val is None:
        return "·" * width + "   —"
    v = max(0.0, min(1.0, float(val)))
    filled = round(v * width)
    return "█" * filled + "░" * (width - filled) + f"  {v:.2f}"


def _trunc(s, n):
    if s is None:
        return "—".ljust(n)
    s = str(s)
    return (s[: n - 1] + "…") if len(s) > n else s.ljust(n)


def _pct_bar(val, width=10):
    filled = round(val * width)
    return "█" * filled + "░" * (width - filled)


# ── display: stage 1 ─────────────────────────────────────────────────────────

def show_spotify_sample(records, n=8):
    _header("STAGE 1  —  SPOTIFY DATA SAMPLE")
    print(f"\n  Pulled {len(records)} unique tracks across all sources.\n")

    _section("Sample tracks")
    print(f"  {'#':<3}  {'Title':<28}  {'Artist':<22}  {'Source':<18}  Pop")
    _rule()
    for i, r in enumerate(records[:n], 1):
        pop = r.get("popularity")
        pop_str = f"{int(pop):>3}" if pop is not None else "  —"
        print(
            f"  {i:<3}  {_trunc(r.get('title'), 28)}  "
            f"{_trunc(r.get('artist'), 22)}  "
            f"{_trunc(r.get('source'), 18)}  {pop_str}"
        )
    if len(records) > n:
        print(f"\n  … and {len(records) - n} more tracks not shown.")

    _section("Source breakdown")
    sources = Counter(r.get("source", "unknown") for r in records)
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        bar = _pct_bar(count / len(records), width=20)
        print(f"  {_trunc(src, 20)}  {bar}  {count:>4} tracks")


# ── display: stage 2 ─────────────────────────────────────────────────────────

def show_llm_features(records, n=6):
    _header("STAGE 2  —  LLM FEATURE ESTIMATES  (Amazon Nova Lite)")
    print("\n  Audio features estimated by the LLM for tracks Spotify no")
    print("  longer serves via its audio_features endpoint.\n")

    for r in records[:n]:
        title  = _trunc(r.get("title"),  30)
        artist = _trunc(r.get("artist"), 24)
        print(f"  {title}  —  {artist}")
        print(f"    Energy       {_bar(r.get('energy'))}")
        print(f"    Valence      {_bar(r.get('valence'))}")
        print(f"    Danceability {_bar(r.get('danceability'))}")
        print(f"    Acousticness {_bar(r.get('acousticness'))}")

        tempo = r.get("tempo_bpm")
        ts    = r.get("time_signature")
        key   = r.get("key")
        mode  = r.get("mode")
        mode_str = "major" if mode == 1 else ("minor" if mode == 0 else "—")
        key_names = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
        key_str = key_names[int(key)] if key is not None else "—"

        print(f"    Tempo        {f'{tempo:.0f} BPM' if tempo else '—':<10}  "
              f"Key: {key_str} {mode_str}   Time sig: {ts if ts else '—'}/4")

        genres = r.get("artist_genres", [])
        print(f"    Genres       {', '.join(genres[:4]) if genres else '—'}")
        print()


# ── display: stage 3 ─────────────────────────────────────────────────────────

def show_clusters(profiles):
    _header("STAGE 3  —  TASTE CLUSTERS  (UMAP + HDBSCAN)")
    print(f"\n  Found {len(profiles)} vibe clusters in your listening history.")
    print("  Each cluster is a natural pocket of similar-sounding music.\n")

    for p in profiles:
        print(f"  ┌──  Cluster {p.cluster_id}:  \"{p.label}\"")
        print(f"  │    {p.size} tracks  |  avg signal weight: {p.signal_weight:.2f}")
        print(f"  │")
        print(f"  │    Energy       {_bar(p.avg_energy)}")
        print(f"  │    Valence      {_bar(p.avg_valence)}")
        print(f"  │    Danceability {_bar(p.avg_danceability)}")
        print(f"  │    Acousticness {_bar(p.avg_acousticness)}")
        print(f"  │    Avg tempo    {p.avg_tempo:.0f} BPM")
        if p.top_genres:
            print(f"  │    Genres       {', '.join(p.top_genres)}")
        print(f"  │")
        print(f"  │    Representative tracks (closest to centroid):")
        for t in p.representative_tracks[:5]:
            dist = t.get("distance_to_centroid")
            dist_str = f"  dist={dist:.3f}" if dist else ""
            print(f"  │      • {_trunc(t.get('title', '?'), 32)}  "
                  f"— {_trunc(t.get('artist', '?'), 22)}{dist_str}")
        print(f"  └")
        print()


# ── display: stage 4 ─────────────────────────────────────────────────────────

def show_discovery(candidates_by_cluster, scored_by_cluster, profiles):
    _header("STAGE 4  —  DISCOVERY + SCORING")

    profile_map = {p.cluster_id: p for p in profiles}
    total_cands  = sum(len(c) for c in candidates_by_cluster.values())
    total_passed = sum(
        sum(1 for t in scored if t.passed_threshold)
        for scored in scored_by_cluster.values()
    )

    print(f"\n  Discovered {total_cands} candidate tracks total.")
    print(f"  {total_passed} passed the cosine similarity threshold.\n")

    from scorer import CONFIDENCE_THRESHOLD

    for cluster_id, candidates in candidates_by_cluster.items():
        label  = profile_map[cluster_id].label if cluster_id in profile_map else "?"
        scored = scored_by_cluster.get(cluster_id, [])
        passed = [s for s in scored if s.passed_threshold]

        print(f"  Cluster {cluster_id}: \"{label}\"")

        # source breakdown
        sources = Counter(c.get("source", "?") for c in candidates)
        src_parts = "  |  ".join(
            f"{src.replace('_', ' ')}: {n}"
            for src, n in sorted(sources.items(), key=lambda x: -x[1])
        )
        print(f"    Sources  {src_parts}")
        print(f"    Scored   {len(scored)} candidates  |  "
              f"{len(passed)} passed threshold (≥{CONFIDENCE_THRESHOLD:.0%})\n")

        # top 5 by confidence
        print(f"    {'':>2}  {'Conf':>5}  {'Title':<30}  Artist")
        _rule()
        for s in scored[:5]:
            status   = "✓" if s.passed_threshold else "✗"
            conf_bar = _pct_bar(s.confidence, width=8)
            print(f"    {status}  {conf_bar} {s.confidence:.0%}  "
                  f"{_trunc(s.title, 30)}  {_trunc(s.artist, 20)}")
        _rule()
        print()


# ── display: stage 5 ─────────────────────────────────────────────────────────

def show_guardrail(reports, profiles):
    _header("STAGE 5  —  LLM GUARDRAIL  (Amazon Nova Pro)")
    print("\n  A second LLM pass reviews every track that passed scoring,")
    print("  catching live recordings, genre mismatches, and interludes.\n")

    profile_map = {p.cluster_id: p for p in profiles}
    total_approved = sum(len(r.approved) for r in reports.values())
    total_rejected = sum(len(r.rejected) for r in reports.values())

    _section("Overall result")
    total = total_approved + total_rejected
    if total > 0:
        a_bar = _pct_bar(total_approved / total, width=30)
        r_bar = _pct_bar(total_rejected / total, width=30)
        print(f"  Approved  {a_bar}  {total_approved} tracks")
        print(f"  Rejected  {r_bar}  {total_rejected} tracks")

    for cluster_id, report in reports.items():
        label = profile_map[cluster_id].label if cluster_id in profile_map else "?"
        print(f"\n  Cluster {cluster_id}: \"{label}\"")
        print(f"    Approved: {len(report.approved)}   Rejected: {len(report.rejected)}")

        if report.llm_summary:
            wrapped = _trunc(report.llm_summary, W - 8)
            print(f"    LLM:  {wrapped}")

        if report.approved:
            print(f"\n    Approved tracks:")
            for v in report.approved[:5]:
                print(f"      ✓  {_trunc(v.title, 30)}  —  {_trunc(v.artist, 22)}")
            if len(report.approved) > 5:
                print(f"         … and {len(report.approved) - 5} more")

        if report.rejected:
            print(f"\n    Rejected tracks:")
            for v in report.rejected:
                print(f"      ✗  {_trunc(v.title, 30)}  —  {_trunc(v.artist, 22)}")
                print(f"         Reason: {_trunc(v.reason, W - 18)}")


# ── cache helpers ─────────────────────────────────────────────────────────────

def _save_cache(records, df, X, cluster_labels, genre_cols):
    _DATA.mkdir(exist_ok=True)
    np.save(_DATA / "X.npy", X)
    np.save(_DATA / "cluster_labels.npy", cluster_labels)
    df.to_csv(_DATA / "processed_tracks.csv", index=False)
    with open(_DATA / "raw_records.json", "w") as f:
        json.dump(records, f, default=str)
    with open(_DATA / "genre_cols.json", "w") as f:
        json.dump(genre_cols, f)


def _load_cache():
    X              = np.load(_DATA / "X.npy")
    cluster_labels = np.load(_DATA / "cluster_labels.npy")
    df             = pd.read_csv(_DATA / "processed_tracks.csv")
    with open(_DATA / "raw_records.json") as f:
        records = json.load(f)
    with open(_DATA / "genre_cols.json") as f:
        genre_cols = json.load(f)
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
    return records, df, X, cluster_labels, genre_cols


# ── main ──────────────────────────────────────────────────────────────────────

def run_demo(skip_ingest: bool, skip_guardrail: bool, min_cluster_size: int):
    print()
    print("  " + "█" * (W - 2))
    title = "SPOTIFY CURATOR  —  DEMO RUN"
    pad   = (W - 2 - len(title)) // 2
    print(" " * (pad + 2) + title)
    print("  " + "█" * (W - 2))
    print(f"\n  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  Nothing will be written to Spotify during this run.\n")

    # ── Stage 1 + 2: Ingest ──────────────────────────────────────────────────
    if skip_ingest and (_DATA / "raw_records.json").exists():
        print("  [cache]  Loading records from data/raw_records.json …")
        records, df, X, cluster_labels, genre_cols = _load_cache()
        print(f"  [cache]  {len(records)} records loaded.\n")
        need_feature_engineering = False
        need_segmentation        = False
    else:
        print("  [stage 1]  Fetching tracks from Spotify …")
        from ingest import run_ingest
        records = run_ingest()
        print(f"  [stage 1]  {len(records)} unique tracks collected.\n")

        print("  [stage 2]  Running feature engineering …")
        from features import run_feature_engineering
        df, X, genre_cols = run_feature_engineering(records)
        print(f"  [stage 2]  Feature matrix: {X.shape[0]} tracks × {X.shape[1]} features.\n")

        need_feature_engineering = False
        need_segmentation        = True

    show_spotify_sample(records)
    show_llm_features(records)

    # ── Stage 3: Segmentation ─────────────────────────────────────────────────
    if need_segmentation:
        print("\n  [stage 3]  Running UMAP + HDBSCAN segmentation …")
        from segmenter import run_segmentation
        X_reduced, cluster_labels, clusterer = run_segmentation(
            X, df, min_cluster_size=min_cluster_size
        )
    else:
        print("\n  [stage 3]  Re-running segmentation on cached matrix …")
        from segmenter import run_segmentation
        X_reduced, cluster_labels, clusterer = run_segmentation(
            X, df, min_cluster_size=min_cluster_size
        )

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    if n_clusters == 0:
        print("\n  No clusters found — library may be too small or too uniform.")
        print("  Try re-running with --min-cluster-size 5")
        return

    # ── Stage 4: Taste model ──────────────────────────────────────────────────
    print("  [stage 4]  Building cluster profiles …")
    from taste_model import run_taste_model

    feature_names = [
        "energy", "valence", "danceability", "acousticness",
        "tempo_norm", "loudness_norm", "speechiness",
        "instrumentalness", "liveness", "time_sig_norm",
        "harmonic_sig", "era_score", "popularity_norm", "hour_sin", "hour_cos",
    ] + genre_cols

    global_centroid, profiles = run_taste_model(X, df, cluster_labels, feature_names)
    print(f"  [stage 4]  {len(profiles)} cluster profiles built.\n")

    _save_cache(records, df, X, cluster_labels, genre_cols)

    show_clusters(profiles)

    # ── Stage 5: Discovery ────────────────────────────────────────────────────
    print("  [stage 5]  Running discovery (Spotify related artists + Deezer + new releases) …")
    from discoverer import run_discovery
    candidates_by_cluster = run_discovery(profiles, df)
    total = sum(len(c) for c in candidates_by_cluster.values())
    print(f"  [stage 5]  {total} candidate tracks found.\n")

    # ── Stage 6: Scoring ──────────────────────────────────────────────────────
    print("  [stage 6]  Scoring candidates by cosine similarity …")
    from scorer import score_all_clusters
    scored_by_cluster = score_all_clusters(candidates_by_cluster, profiles, genre_cols)
    passed = sum(
        sum(1 for s in sc if s.passed_threshold) for sc in scored_by_cluster.values()
    )
    print(f"  [stage 6]  {passed} candidates passed the confidence threshold.\n")

    show_discovery(candidates_by_cluster, scored_by_cluster, profiles)

    # ── Stage 7: Guardrail ────────────────────────────────────────────────────
    if skip_guardrail:
        print("  [stage 7]  Guardrail skipped (--skip-guardrail).")
        show_guardrail({}, profiles)
    else:
        print("  [stage 7]  Running LLM guardrail review (Amazon Nova Pro) …")
        from guardrail import run_guardrail
        reports = run_guardrail(profiles, scored_by_cluster)
        total_approved = sum(len(r.approved) for r in reports.values())
        total_rejected = sum(len(r.rejected) for r in reports.values())
        print(f"  [stage 7]  {total_approved} approved / {total_rejected} rejected.\n")
        show_guardrail(reports, profiles)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("  " + "═" * (W - 2))
    print("  DEMO COMPLETE  —  ready for playlist_writer.py")
    print("  " + "═" * (W - 2))
    print(f"\n  Clusters found:      {len(profiles)}")
    print(f"  Candidates sourced:  {total}")
    print(f"  Passed scoring:      {passed}")
    if not skip_guardrail:
        total_approved = sum(len(r.approved) for r in reports.values())
        print(f"  Approved by LLM:     {total_approved}")
    print(f"\n  Cache written to:    data/")
    print()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spotify Curator — demo run")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Reuse cached data/raw_records.json instead of fetching from Spotify",
    )
    parser.add_argument(
        "--skip-guardrail",
        action="store_true",
        help="Skip the Bedrock guardrail LLM call (faster demo)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Minimum tracks to form a cluster (default: 10)",
    )
    args = parser.parse_args()

    run_demo(
        skip_ingest=args.skip_ingest,
        skip_guardrail=args.skip_guardrail,
        min_cluster_size=args.min_cluster_size,
    )
