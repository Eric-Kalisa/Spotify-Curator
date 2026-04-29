"""
segmenter.py
============
Stage 3: Segmentation

Reduces your high-dimensional feature matrix with UMAP, then
clusters the reduced space with HDBSCAN to find natural vibe
buckets in your listening history.

Why UMAP + HDBSCAN over K-Means:
- UMAP preserves local and global structure better than PCA for music
- HDBSCAN finds clusters of arbitrary shape (music taste isn't spherical)
- HDBSCAN auto-determines the number of clusters (no need to pick K)
- HDBSCAN marks genuine outliers as noise (-1) instead of forcing them in
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency check — give a clear error if libraries are missing
# ─────────────────────────────────────────────────────────────────────────────
def _check_dependencies():
    try:
        import umap
    except ImportError:
        raise ImportError(
            "Missing required package: umap-learn\n"
            "Run: pip install umap-learn"
        )


# ─────────────────────────────────────────────────────────────────────────────
# UMAP dimensionality reduction
# ─────────────────────────────────────────────────────────────────────────────

def reduce_dimensions(
    X: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduces the feature matrix to a lower-dimensional space using UMAP.

    Parameters
    ----------
    X             : feature matrix (n_songs, n_features)
    n_components  : number of output dimensions.
                    10 is a good balance — low enough for HDBSCAN to work
                    well, high enough to preserve structure.
                    Use 2 for visualization purposes.
    n_neighbors   : controls how UMAP balances local vs global structure.
                    15 works well for datasets up to ~5000 songs.
                    Increase to 30 if you have 5000+ tracks.
    min_dist      : how tightly UMAP packs points together.
                    0.1 is good for clustering (tighter clusters).
                    Use 0.5 for visualization (more spread out).
    random_state  : seed for reproducibility

    Returns
    -------
    X_reduced : np.ndarray of shape (n_songs, n_components)
    """
    import umap as umap_lib

    logger.info(
        f"Running UMAP: {X.shape[1]} → {n_components} dimensions | "
        f"n_neighbors={n_neighbors} min_dist={min_dist}"
    )

    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        verbose=False,
    )

    X_reduced = reducer.fit_transform(X)
    logger.info(f"UMAP complete: output shape = {X_reduced.shape}")
    return X_reduced


def reduce_for_visualization(X: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper: reduces to 2D purely for plotting.
    Do NOT use this output for clustering — use reduce_dimensions() for that.
    """
    return reduce_dimensions(X, n_components=2, min_dist=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# HDBSCAN clustering
# ─────────────────────────────────────────────────────────────────────────────

def cluster_tracks(
    X_reduced: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.0,
) -> Tuple[np.ndarray, object]:
    """
    Clusters the UMAP-reduced feature matrix using HDBSCAN.

    Parameters
    ----------
    X_reduced               : UMAP output (n_songs, n_components)
    min_cluster_size        : minimum tracks to form a cluster.
                              10 is a reasonable floor — any smaller
                              and clusters won't represent a real vibe.
                              Increase if you have very large libraries.
    min_samples             : controls how conservative clustering is.
                              Higher = more noise points, fewer but
                              more confident clusters.
    cluster_selection_epsilon : merges clusters closer than this distance.
                              0.0 = let HDBSCAN decide naturally.
                              Increase (e.g. 0.3) if you're getting
                              too many tiny clusters.

    Returns
    -------
    labels    : np.ndarray of shape (n_songs,)
                -1 = noise/outlier, 0..N = cluster id
    clusterer : fitted HDBSCAN object (for soft clustering membership
                probabilities via clusterer.probabilities_)
    """
    from sklearn.cluster import HDBSCAN

    logger.info(
        f"Running HDBSCAN: min_cluster_size={min_cluster_size} "
        f"min_samples={min_samples}"
    )

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method="eom",   # "eom" finds bigger clusters,
                                          # use "leaf" for more granular ones
        metric="euclidean",
    )

    labels = clusterer.fit_predict(X_reduced)

    # report
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(labels == -1))
    noise_pct  = 100 * n_noise / len(labels)

    logger.info(
        f"HDBSCAN complete: {n_clusters} clusters found | "
        f"{n_noise} noise points ({noise_pct:.1f}%)"
    )
    print(f"\n  Clusters found: {n_clusters}")
    print(f"  Noise points:   {n_noise} ({noise_pct:.1f}% of library)")

    # warn if clustering quality seems poor
    if n_clusters == 0:
        logger.warning(
            "HDBSCAN found 0 clusters — your library may be too small or too uniform. "
            "Try reducing min_cluster_size."
        )
    elif noise_pct > 40:
        logger.warning(
            f"High noise ratio ({noise_pct:.1f}%) — consider reducing min_samples "
            "or increasing n_neighbors in UMAP."
        )

    return labels, clusterer


def get_soft_memberships(
    clusterer,
    X_reduced: np.ndarray
) -> np.ndarray:
    """
    Returns per-track cluster membership confidence scores.
    Shape: (n_songs,) — confidence that each track belongs to its assigned cluster.

    sklearn's HDBSCAN stores this in clusterer.probabilities_. Values near 1.0
    mean the track sits firmly in its cluster; near 0.0 means it's on the edge.
    """
    try:
        probs = clusterer.probabilities_
        logger.info(f"Soft memberships computed: shape = {probs.shape}")
        return probs
    except Exception as e:
        logger.warning(f"Could not compute soft memberships: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Cluster stability validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_cluster_stability(
    X_reduced: np.ndarray,
    n_runs: int = 5,
    min_cluster_size: int = 10,
) -> dict:
    """
    Runs HDBSCAN n_runs times with different random seeds and checks
    whether the number of clusters is stable across runs.

    A stable result means the vibe structure in your data is real,
    not an artifact of randomness in UMAP.

    Returns a dict with: cluster_counts, is_stable, recommended_n
    """
    logger.info(f"Running cluster stability check ({n_runs} runs)")
    from sklearn.cluster import HDBSCAN

    counts = []
    for seed in range(n_runs):
        # re-run UMAP with a different seed each time
        X_rerun = reduce_dimensions(X_reduced, n_components=10, random_state=seed)
        c = HDBSCAN(min_cluster_size=min_cluster_size)
        labels = c.fit_predict(X_rerun)
        n = len(set(labels)) - (1 if -1 in labels else 0)
        counts.append(n)

    variance    = np.var(counts)
    is_stable   = variance <= 2.0   # allow ±1 cluster variation
    recommended = int(np.median(counts))

    logger.info(
        f"Stability check: counts={counts} variance={variance:.2f} "
        f"stable={is_stable} recommended_n={recommended}"
    )
    return {
        "cluster_counts": counts,
        "variance":       variance,
        "is_stable":      is_stable,
        "recommended_n":  recommended,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full segmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(
    X: np.ndarray,
    df: pd.DataFrame,
    min_cluster_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Full segmentation pipeline:
      1. Reduce dimensions with UMAP (10D for clustering)
      2. Cluster with HDBSCAN
      3. Attach cluster labels back to df

    Returns
    -------
    X_reduced      : UMAP-reduced matrix (for taste_model centroid computation)
    cluster_labels : per-track cluster assignments
    clusterer      : fitted HDBSCAN (for soft memberships)
    """
    _check_dependencies()

    logger.info("=== Starting segmentation pipeline ===")

    # Step 1: reduce to 10D for clustering
    X_reduced = reduce_dimensions(X, n_components=10)

    # Step 2: cluster
    cluster_labels, clusterer = cluster_tracks(
        X_reduced,
        min_cluster_size=min_cluster_size
    )

    # Step 3: attach labels to dataframe for downstream use
    df["cluster_id"] = cluster_labels

    logger.info("=== Segmentation pipeline complete ===")
    return X_reduced, cluster_labels, clusterer
