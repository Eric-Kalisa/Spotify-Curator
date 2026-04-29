"""
training_tests.py
=================
Unit and smoke tests for the Training+Verifying pipeline.
No external API calls required — all tests use synthetic data.

Suites:
  1 - Feature Engineering : column contracts, time_sig_norm, scorer alignment
  2 - Scorer              : cosine math, track_to_vector, confidence range
  3 - Segmenter           : UMAP+HDBSCAN output shapes (requires umap-learn + hdbscan)
  4 - Taste Model         : centroids, cluster profiles, sorting
  5 - Guardrail Parser    : _parse_llm_response JSON handling and fallback

Run with:
    python tests/training_tests.py
    python tests/training_tests.py -v
"""

import sys
import json
import unittest
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "Preprocessing"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "Training+Verifying"))

logging.basicConfig(
    filename="training_tests.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_raw_df(n=20, seed=42):
    """
    Builds a minimal raw DataFrame with all fields engineer_features needs.
    Values stay within valid Spotify ranges.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "energy":           rng.uniform(0, 1, n).tolist(),
        "valence":          rng.uniform(0, 1, n).tolist(),
        "danceability":     rng.uniform(0, 1, n).tolist(),
        "acousticness":     rng.uniform(0, 1, n).tolist(),
        "tempo_bpm":        rng.uniform(60, 200, n).tolist(),
        "loudness":         rng.uniform(-60, 0, n).tolist(),
        "speechiness":      rng.uniform(0, 0.3, n).tolist(),
        "instrumentalness": rng.uniform(0, 0.5, n).tolist(),
        "liveness":         rng.uniform(0, 0.5, n).tolist(),
        "key":              rng.integers(0, 12, n).tolist(),
        "mode":             rng.choice([0.0, 1.0], n).tolist(),
        "time_signature":   rng.choice([3, 4, 5], n).tolist(),
        "popularity":       rng.uniform(0, 100, n).tolist(),
        "release_year":     rng.integers(1970, 2025, n).tolist(),
        "hour_of_day":      rng.integers(0, 24, n).tolist(),
        "played_at":        ["2024-01-15T10:00:00Z"] * n,
        "source":           ["liked"] * n,
        "artist_genres":    [["indie pop", "dream pop"]] * n,
    })


def _make_engineered_data(n=30):
    """Runs the full feature engineering pipeline on synthetic data."""
    from features import engineer_features, encode_genre_tags, build_feature_matrix
    df = _make_raw_df(n)
    df = engineer_features(df)
    df, genre_cols = encode_genre_tags(df)
    X = build_feature_matrix(df, genre_cols)
    return df, X, genre_cols


def _make_cluster_profile(centroid, cluster_id=0):
    from taste_model import ClusterProfile
    p = ClusterProfile(
        cluster_id=cluster_id,
        centroid=centroid,
        size=15,
        signal_weight=0.7,
    )
    p.avg_energy = 0.6
    p.avg_valence = 0.5
    p.avg_danceability = 0.65
    p.avg_acousticness = 0.3
    p.avg_tempo = 120.0
    p.top_genres = ["indie pop"]
    p.label = "High Energy · Upbeat · Electronic / Produced"
    p.representative_tracks = [
        {"title": "Test Track", "artist": "Test Artist", "artist_id": "abc", "track_id": "t0"}
    ]
    return p


# =============================================================================
# SUITE 1 — FEATURE ENGINEERING
# =============================================================================

class TestFeatureEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from features import engineer_features, encode_genre_tags, build_feature_matrix
        cls.df_raw = _make_raw_df(20)
        cls.df = engineer_features(cls.df_raw.copy())
        cls.df_enc, cls.genre_cols = encode_genre_tags(cls.df.copy())
        cls.X = build_feature_matrix(cls.df_enc, cls.genre_cols)

    def test_1_1_all_required_columns_produced(self):
        """
        engineer_features must produce every column that build_feature_matrix
        selects. A missing column causes a KeyError when building the matrix.
        """
        expected = [
            "energy", "valence", "danceability", "acousticness",
            "tempo_norm", "loudness_norm", "speechiness",
            "instrumentalness", "liveness", "time_sig_norm",
            "harmonic_sig", "era_score", "popularity_norm",
            "hour_sin", "hour_cos",
        ]
        for col in expected:
            self.assertIn(col, self.df.columns,
                f"engineer_features did not produce column '{col}'")

    def test_1_2_time_sig_norm_in_range(self):
        """time_sig_norm must be in [0, 1] for all rows."""
        col = self.df["time_sig_norm"]
        self.assertTrue(
            (col >= 0).all() and (col <= 1).all(),
            f"time_sig_norm out of [0, 1]: min={col.min():.3f} max={col.max():.3f}"
        )

    def test_1_3_all_bounded_columns_in_range(self):
        """All normalized and score columns must be clipped to [0, 1]."""
        for col in ["tempo_norm", "loudness_norm", "era_score", "popularity_norm", "time_sig_norm"]:
            series = self.df[col]
            self.assertTrue(
                (series >= 0).all() and (series <= 1).all(),
                f"'{col}' has values outside [0, 1]: min={series.min():.3f} max={series.max():.3f}"
            )

    def test_1_4_feature_matrix_shape(self):
        """Feature matrix must have one row per track and at least 15 columns."""
        self.assertEqual(self.X.shape[0], len(self.df_enc),
            "Feature matrix row count doesn't match DataFrame length")
        self.assertGreaterEqual(self.X.shape[1], 15,
            f"Feature matrix has only {self.X.shape[1]} columns, expected >= 15")

    def test_1_5_feature_matrix_no_nan_or_inf(self):
        """Feature matrix must contain no NaN or Inf values."""
        self.assertFalse(np.isnan(self.X).any(), "Feature matrix contains NaN")
        self.assertFalse(np.isinf(self.X).any(), "Feature matrix contains Inf")

    def test_1_6_scorer_column_alignment(self):
        """
        AUDIO_FEATURE_COLS in scorer.py must exactly match the base continuous
        columns assembled in build_feature_matrix. A mismatch means candidate
        vectors are a different length than cluster centroids.
        """
        from scorer import AUDIO_FEATURE_COLS
        expected = [
            "energy", "valence", "danceability", "acousticness",
            "tempo_norm", "loudness_norm", "speechiness",
            "instrumentalness", "liveness", "time_sig_norm",
            "harmonic_sig", "era_score", "popularity_norm",
            "hour_sin", "hour_cos",
        ]
        self.assertEqual(list(AUDIO_FEATURE_COLS), expected,
            f"AUDIO_FEATURE_COLS is out of sync with build_feature_matrix.\n"
            f"  Got:      {list(AUDIO_FEATURE_COLS)}\n"
            f"  Expected: {expected}"
        )


# =============================================================================
# SUITE 2 — SCORER
# =============================================================================

class TestScorer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df, cls.X, cls.genre_cols = _make_engineered_data(30)
        cls.centroid = cls.X.mean(axis=0)
        cls.profile = _make_cluster_profile(cls.centroid)

    def test_2_1_cosine_identical_vectors(self):
        """Cosine similarity of a vector with itself must be 1.0."""
        from scorer import cosine_similarity
        v = np.array([0.5, 0.3, 0.8, 0.1])
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=6)

    def test_2_2_cosine_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors must be 0.0."""
        from scorer import cosine_similarity
        self.assertAlmostEqual(
            cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            0.0, places=6
        )

    def test_2_3_cosine_zero_vector(self):
        """Cosine similarity must return 0.0 when either input is the zero vector."""
        from scorer import cosine_similarity
        v = np.array([0.5, 0.3])
        z = np.zeros(2)
        self.assertEqual(cosine_similarity(v, z), 0.0)
        self.assertEqual(cosine_similarity(z, v), 0.0)

    def test_2_4_track_to_vector_length_matches_centroid(self):
        """
        track_to_vector must return a vector with the same length as the cluster
        centroid. A mismatch causes incorrect cosine similarity results.
        """
        from scorer import track_to_vector
        vec = track_to_vector(dict(self.df.iloc[0]), self.genre_cols)
        self.assertIsNotNone(vec, "track_to_vector returned None for a valid track row")
        self.assertEqual(len(vec), len(self.centroid),
            f"track_to_vector length {len(vec)} != centroid length {len(self.centroid)}")

    def test_2_5_track_to_vector_none_for_incomplete_track(self):
        """track_to_vector must return None if any of the 4 required audio features is absent."""
        from scorer import track_to_vector
        self.assertIsNone(
            track_to_vector(
                {"energy": 0.7, "valence": 0.6, "danceability": 0.8},  # missing acousticness
                self.genre_cols
            ),
            "track_to_vector should return None when acousticness is missing"
        )

    def test_2_6_score_candidates_sorted_by_confidence(self):
        """score_candidates must return tracks sorted by confidence descending."""
        from scorer import score_candidates
        candidates = [dict(row) for _, row in self.df.iterrows()]
        scored = score_candidates(candidates, self.profile, self.genre_cols)
        if len(scored) < 2:
            self.skipTest("Not enough scoreable candidates in synthetic data")
        confidences = [s.confidence for s in scored]
        self.assertEqual(confidences, sorted(confidences, reverse=True),
            "score_candidates result is not sorted by confidence descending")

    def test_2_7_confidence_scores_in_range(self):
        """All confidence scores must be in [0, 1]."""
        from scorer import score_candidates
        candidates = [dict(row) for _, row in self.df.iterrows()]
        for s in score_candidates(candidates, self.profile, self.genre_cols):
            self.assertGreaterEqual(s.confidence, 0.0)
            self.assertLessEqual(s.confidence, 1.0)

    def test_2_8_passed_threshold_consistent(self):
        """passed_threshold must be True iff confidence >= CONFIDENCE_THRESHOLD."""
        from scorer import score_candidates, CONFIDENCE_THRESHOLD
        candidates = [dict(row) for _, row in self.df.iterrows()]
        for s in score_candidates(candidates, self.profile, self.genre_cols):
            self.assertEqual(
                s.passed_threshold, s.confidence >= CONFIDENCE_THRESHOLD,
                f"'{s.title}': confidence={s.confidence:.3f} but "
                f"passed_threshold={s.passed_threshold} (threshold={CONFIDENCE_THRESHOLD})"
            )


# =============================================================================
# SUITE 3 — SEGMENTER  (requires umap-learn + hdbscan)
# =============================================================================

class TestSegmenter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import umap
            from sklearn.cluster import HDBSCAN  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(
                f"Missing dependency: {e} — run: pip install umap-learn"
            )
        rng = np.random.default_rng(0)
        cls.X = rng.random((80, 15)).astype(np.float32)
        cls.df = pd.DataFrame({"track_id": [f"t{i}" for i in range(80)]})

    def test_3_1_reduce_dimensions_output_shape(self):
        """reduce_dimensions must return (n_tracks, n_components)."""
        from segmenter import reduce_dimensions
        out = reduce_dimensions(self.X, n_components=5)
        self.assertEqual(out.shape, (80, 5),
            f"Expected shape (80, 5), got {out.shape}")

    def test_3_2_cluster_tracks_label_count(self):
        """cluster_tracks must return exactly one label per input track."""
        from segmenter import reduce_dimensions, cluster_tracks
        X_r = reduce_dimensions(self.X, n_components=5)
        labels, _ = cluster_tracks(X_r, min_cluster_size=5)
        self.assertEqual(len(labels), len(self.X),
            f"Expected {len(self.X)} labels, got {len(labels)}")

    def test_3_3_cluster_labels_valid(self):
        """All cluster labels must be integers >= -1 (where -1 = noise)."""
        from segmenter import reduce_dimensions, cluster_tracks
        X_r = reduce_dimensions(self.X, n_components=5)
        labels, _ = cluster_tracks(X_r, min_cluster_size=5)
        self.assertTrue(all(int(l) >= -1 for l in labels),
            "cluster_tracks returned a label < -1")

    def test_3_4_run_segmentation_attaches_cluster_id(self):
        """run_segmentation must write a cluster_id column into the DataFrame."""
        from segmenter import run_segmentation
        df = self.df.copy()
        _, labels, _ = run_segmentation(self.X, df, min_cluster_size=5)
        self.assertIn("cluster_id", df.columns,
            "run_segmentation did not attach cluster_id to the DataFrame")
        self.assertEqual(len(df["cluster_id"]), len(self.X))


# =============================================================================
# SUITE 4 — TASTE MODEL
# =============================================================================

class TestTasteModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        n, f = 40, 15
        cls.X = rng.random((n, f))
        cls.df = pd.DataFrame({
            "track_id":      [f"t{i}" for i in range(n)],
            "signal_weight": rng.uniform(0.1, 1.0, n),
            "energy":        rng.uniform(0, 1, n),
            "valence":       rng.uniform(0, 1, n),
            "danceability":  rng.uniform(0, 1, n),
            "acousticness":  rng.uniform(0, 1, n),
            "tempo_bpm":     rng.uniform(60, 200, n),
            "artist_genres": [["indie pop"]] * n,
        })
        # two clean clusters: tracks 0–19 → cluster 0, 20–39 → cluster 1
        cls.labels = np.array([0] * 20 + [1] * 20)
        cls.feature_names = [f"f{i}" for i in range(f)]

    def test_4_1_global_centroid_shape(self):
        """compute_global_centroid must return a vector matching X's feature count."""
        from taste_model import compute_global_centroid
        centroid = compute_global_centroid(self.X, self.df)
        self.assertEqual(centroid.shape, (self.X.shape[1],))

    def test_4_2_global_centroid_is_weighted(self):
        """
        Weighted centroid must differ from a simple mean when signal weights
        are non-uniform. If they're equal, signal_weight isn't being applied.
        """
        from taste_model import compute_global_centroid
        weighted = compute_global_centroid(self.X, self.df)
        unweighted = self.X.mean(axis=0)
        self.assertFalse(np.allclose(weighted, unweighted),
            "Weighted centroid is identical to unweighted mean — "
            "signal_weight may not be applied correctly")

    def test_4_3_profile_count(self):
        """build_cluster_profiles must return one profile per non-noise cluster."""
        from taste_model import build_cluster_profiles
        profiles = build_cluster_profiles(
            self.X, self.df, self.labels, self.feature_names, min_cluster_size=5
        )
        self.assertEqual(len(profiles), 2,
            f"Expected 2 profiles (one per cluster), got {len(profiles)}")

    def test_4_4_centroid_shape(self):
        """Every ClusterProfile centroid must have the same length as X's columns."""
        from taste_model import build_cluster_profiles
        profiles = build_cluster_profiles(
            self.X, self.df, self.labels, self.feature_names, min_cluster_size=5
        )
        for p in profiles:
            self.assertEqual(len(p.centroid), self.X.shape[1],
                f"Cluster {p.cluster_id} centroid length {len(p.centroid)} != {self.X.shape[1]}")

    def test_4_5_profiles_sorted_by_size_descending(self):
        """build_cluster_profiles must return profiles sorted by size descending."""
        from taste_model import build_cluster_profiles
        profiles = build_cluster_profiles(
            self.X, self.df, self.labels, self.feature_names, min_cluster_size=5
        )
        sizes = [p.size for p in profiles]
        self.assertEqual(sizes, sorted(sizes, reverse=True),
            "Profiles are not sorted by size descending")

    def test_4_6_representative_tracks_populated(self):
        """Each ClusterProfile must have at least one representative track."""
        from taste_model import build_cluster_profiles
        profiles = build_cluster_profiles(
            self.X, self.df, self.labels, self.feature_names, min_cluster_size=5
        )
        for p in profiles:
            self.assertGreater(len(p.representative_tracks), 0,
                f"Cluster {p.cluster_id} has no representative tracks")


# =============================================================================
# SUITE 5 — GUARDRAIL PARSER  (no AWS required)
# =============================================================================

class TestGuardrailParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from scorer import ScoredTrack
        cls.candidates = [
            ScoredTrack(
                track_id="t1", title="Track One", artist="Artist A",
                cluster_id=0, confidence=0.92, raw_similarity=0.92,
                explanation="energy matches", passed_threshold=True,
                audio_features={"energy": 0.8, "valence": 0.7, "genres": ["indie pop"]},
            ),
            ScoredTrack(
                track_id="t2", title="Track Two", artist="Artist B",
                cluster_id=0, confidence=0.81, raw_similarity=0.81,
                explanation="mood matches", passed_threshold=True,
                audio_features={"energy": 0.6, "valence": 0.6, "genres": []},
            ),
        ]

    def test_5_1_parses_valid_json_correctly(self):
        """_parse_llm_response must correctly split approved vs rejected tracks."""
        from guardrail import _parse_llm_response
        raw = json.dumps({
            "summary": "One strong match, one weak.",
            "verdicts": [
                {"title": "Track One", "artist": "Artist A", "approved": True,  "reason": "fits well"},
                {"title": "Track Two", "artist": "Artist B", "approved": False, "reason": "too mellow"},
            ]
        })
        approved, rejected, summary = _parse_llm_response(raw, self.candidates)
        self.assertEqual(len(approved), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(approved[0].track_id, "t1")
        self.assertEqual(rejected[0].track_id, "t2")
        self.assertIn("match", summary)

    def test_5_2_fallback_approves_all_on_invalid_json(self):
        """
        If the LLM returns malformed JSON, _parse_llm_response must approve all
        candidates rather than silently discard them.
        """
        from guardrail import _parse_llm_response
        approved, rejected, _ = _parse_llm_response("not json {{{", self.candidates)
        self.assertEqual(len(approved), len(self.candidates),
            "Fallback should approve all candidates when JSON is unparseable")
        self.assertEqual(len(rejected), 0)

    def test_5_3_handles_too_few_verdicts(self):
        """
        If the LLM returns fewer verdicts than candidates, unreviewed tracks must
        be approved by default rather than silently dropped.
        """
        from guardrail import _parse_llm_response
        raw = json.dumps({
            "summary": "Only reviewed one.",
            "verdicts": [
                {"title": "Track One", "artist": "Artist A", "approved": True, "reason": "fits"},
            ]
        })
        approved, rejected, _ = _parse_llm_response(raw, self.candidates)
        self.assertEqual(len(approved) + len(rejected), len(self.candidates),
            f"Expected {len(self.candidates)} total verdicts, "
            f"got {len(approved) + len(rejected)}")

    def test_5_4_review_prompt_contains_profile_fields(self):
        """_build_review_prompt must include the cluster label and key audio stats."""
        from guardrail import _build_review_prompt
        profile = _make_cluster_profile(np.zeros(15))
        prompt = _build_review_prompt(profile, self.candidates)
        self.assertIn(profile.label, prompt, "Prompt is missing the cluster label")
        self.assertIn("Energy", prompt,       "Prompt is missing the Energy field")
        self.assertIn("Track One", prompt,    "Prompt is missing a candidate track title")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    print("\n" + "=" * 65)
    print("  TRAINING + VERIFYING TEST SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None
    runner = unittest.TextTestRunner(
        verbosity=2 if "-v" in sys.argv else 1,
        stream=sys.stdout,
    )

    suites = [
        ("Suite 1: Feature Engineering", TestFeatureEngineering),
        ("Suite 2: Scorer",              TestScorer),
        ("Suite 3: Segmenter",           TestSegmenter),
        ("Suite 4: Taste Model",         TestTasteModel),
        ("Suite 5: Guardrail Parser",    TestGuardrailParser),
    ]

    results = []
    for label, cls in suites:
        print(f"\n-- {label} " + "-" * (63 - len(label)))
        results.append(runner.run(loader.loadTestsFromTestCase(cls)))

    total_run  = sum(r.testsRun for r in results)
    total_fail = sum(len(r.failures) + len(r.errors) for r in results)
    total_skip = sum(len(r.skipped) for r in results)

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Tests run : {total_run}")
    print(f"  Passed    : {total_run - total_fail - total_skip}")
    print(f"  Skipped   : {total_skip}")
    print(f"  Failed    : {total_fail}")
    print(f"  Result    : {'ALL PASSED' if total_fail == 0 else 'SOME FAILED'}")
    print("=" * 65)
    print(f"\n  Full log: training_tests.log\n")
    return total_fail


if __name__ == "__main__":
    sys.exit(run_tests())
