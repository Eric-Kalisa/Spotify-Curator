"""
ingest_tests.py
===============
Integration tests for the full ingest pipeline.

Runs a small slice of real data through assembly and confirms the output
is correctly shaped for features.py. Assumes api_tests and AWS_API_tests
have already passed — this file does not re-check Spotify connectivity,
LLM field names, or audio feature ranges (those are covered upstream).

  Suite 1 - Raw Data        : Raw tracks have the fields assembly needs
  Suite 2 - Pipeline Output : Assembled records are feature-engineering-ready

Run with:
    python tests/ingest_tests.py
    python tests/ingest_tests.py -v
"""

import os
import sys
import unittest
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Service_authentications"))

load_dotenv(PROJECT_ROOT / ".env")

# ingest.py logs to logs/ingest.log — ensure the directory exists
os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)

logging.basicConfig(
    filename="ingest_tests.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Run a small pipeline once at module load — all tests share this data.
# We slice to 10 tracks to keep Bedrock calls fast.
# ─────────────────────────────────────────────────────────────────────────────
from ingest import fetch_recently_played, assemble_track_records
from llm_features import estimate_audio_features, estimate_artist_genres
from features import run_feature_engineering

print("\n  [setup] Fetching tracks from Spotify...")
_RAW_TRACKS = fetch_recently_played()[:10]

print(f"  [setup] Estimating audio features via Bedrock ({len(_RAW_TRACKS)} tracks)...")
_AUDIO_FEATURES = estimate_audio_features(_RAW_TRACKS) if _RAW_TRACKS else {}

print("  [setup] Estimating genre tags via Bedrock...")
_seen: dict = {}
for _t in _RAW_TRACKS:
    if _t.get("artists"):
        _a = _t["artists"][0]
        _aid = _a.get("id")
        if _aid and _aid not in _seen:
            _seen[_aid] = {"id": _aid, "name": _a.get("name", ""), "track_names": []}
        if _aid:
            _seen[_aid]["track_names"].append(_t["name"])
for _a in _seen.values():
    _a["track_names"] = _a["track_names"][:3]
_GENRES_MAP = estimate_artist_genres(list(_seen.values())) if _seen else {}

_RECORDS = assemble_track_records(_RAW_TRACKS, _AUDIO_FEATURES, _GENRES_MAP)
print(f"  [setup] Assembled {len(_RECORDS)} records. Starting tests...\n")

logger.info(f"Pipeline setup: {len(_RAW_TRACKS)} raw tracks -> {len(_RECORDS)} assembled records")


# =============================================================================
# SUITE 1 — RAW DATA
# Checks that source functions return data in the shape assemble_track_records
# expects before we commit to the expensive Bedrock call.
# =============================================================================
class TestRawData(unittest.TestCase):

    def test_1_1_raw_tracks_have_assembly_fields(self):
        """
        Every raw track must have id, name, artists, and source.
        These are the four fields assemble_track_records reads directly.
        Missing any one means that track gets silently dropped.
        """
        logger.info("TEST 1.1: Raw track field validation")
        self.assertGreater(len(_RAW_TRACKS), 0,
            "fetch_recently_played returned no tracks — play something on Spotify first")

        for i, track in enumerate(_RAW_TRACKS):
            for field in ["id", "name", "artists"]:
                self.assertIn(field, track,
                    f"Track {i} missing required field '{field}'")
            self.assertGreater(len(track["artists"]), 0,
                f"Track {i} has empty artists list")
            self.assertIsNotNone(track["artists"][0].get("id"),
                f"Track {i} first artist has no id — genres_map lookup will silently return []")

        logger.info("TEST 1.1 PASSED")
        print(f"\n  All {len(_RAW_TRACKS)} raw tracks have required assembly fields")


# =============================================================================
# SUITE 2 — ASSEMBLED RECORDS
# Verifies the output of assemble_track_records is shaped correctly for
# features.py — specifically for build_dataframe, engineer_features,
# encode_genre_tags, and build_feature_matrix.
# =============================================================================
class TestAssembledRecords(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not _RECORDS:
            raise unittest.SkipTest(
                "No assembled records — check Spotify auth and Bedrock access"
            )

    def test_2_1_assembly_produces_records(self):
        """assemble_track_records must return at least one record."""
        logger.info("TEST 2.1: Assembly output count")
        self.assertGreater(len(_RECORDS), 0,
            "assemble_track_records returned no records")
        logger.info(f"TEST 2.1 PASSED: {len(_RECORDS)} records")
        print(f"\n  Assembled records: {len(_RECORDS)}")

    def test_2_2_no_duplicate_track_ids(self):
        """
        Each assembled record must have a unique track_id.
        Duplicates would inflate signal weights in the feature matrix.
        """
        logger.info("TEST 2.2: Checking for duplicate track_ids")
        ids = [r["track_id"] for r in _RECORDS]
        self.assertEqual(len(ids), len(set(ids)),
            f"Found {len(ids) - len(set(ids))} duplicate track_ids in assembled records")
        logger.info("TEST 2.2 PASSED: no duplicates")
        print(f"  No duplicate track_ids across {len(_RECORDS)} records")

    def test_2_3_identifier_fields_populated(self):
        """
        track_id, title, artist, and source must be non-null on every record.
        These are used for deduplication and source weighting in features.py.
        """
        logger.info("TEST 2.3: Identifier fields")
        required = ["track_id", "title", "artist", "source"]

        for i, record in enumerate(_RECORDS):
            for field in required:
                self.assertIn(field, record,
                    f"Record {i} missing field '{field}'")
                self.assertIsNotNone(record[field],
                    f"Record {i} has null '{field}'")

        logger.info("TEST 2.3 PASSED")
        print(f"  All identifier fields (track_id, title, artist, source) populated")

    def test_2_4_required_audio_features_survive_assembly(self):
        """
        Checks that the 4 audio features build_dataframe uses in dropna
        are non-null on every assembled record. AWS_API_tests already validates
        the LLM returns these fields; this test confirms they survive the
        assembly join and aren't lost or overwritten.
        """
        logger.info("TEST 2.4: Required audio features non-null after assembly")
        required = ["energy", "valence", "danceability", "acousticness"]

        for i, record in enumerate(_RECORDS):
            for field in required:
                self.assertIsNotNone(record.get(field),
                    f"Record {i} ('{record.get('title')}') has null '{field}' "
                    f"— this row will be silently dropped by build_dataframe")

        logger.info("TEST 2.4 PASSED")
        print(f"  Required audio features (energy/valence/danceability/acousticness) non-null on all records")

    def test_2_5_track_metadata_fields_present(self):
        """
        Checks that track metadata fields used by engineer_features are present:
        popularity (popularity_norm), duration_ms, explicit, release_year (era_score).
        """
        logger.info("TEST 2.5: Track metadata fields")
        metadata_fields = ["popularity", "duration_ms", "explicit", "release_year"]

        for i, record in enumerate(_RECORDS):
            for field in metadata_fields:
                self.assertIn(field, record,
                    f"Record {i} missing metadata field '{field}'")

        logger.info("TEST 2.5 PASSED")
        print(f"  Track metadata fields present on all records")

    def test_2_6_artist_genres_is_list(self):
        """
        artist_genres must be a list on every record.
        encode_genre_tags in features.py calls MultiLabelBinarizer on this field —
        a non-list value (None, string) will raise a TypeError at encode time.
        """
        logger.info("TEST 2.6: artist_genres type")

        for i, record in enumerate(_RECORDS):
            self.assertIn("artist_genres", record,
                f"Record {i} missing 'artist_genres'")
            self.assertIsInstance(record["artist_genres"], list,
                f"Record {i} artist_genres is {type(record['artist_genres']).__name__}, expected list")

        logger.info("TEST 2.6 PASSED")
        print(f"  artist_genres is a list on all {len(_RECORDS)} records")

    def test_2_7_feature_engineering_runs_without_error(self):
        """
        Passes the assembled records through the full run_feature_engineering()
        pipeline and confirms a non-empty feature matrix is produced.
        This is the end-to-end 'ready for UMAP' gate.
        """
        logger.info("TEST 2.7: Full feature engineering pipeline")

        try:
            df, X, genre_cols = run_feature_engineering(_RECORDS)

            self.assertGreater(len(df), 0,
                "Feature engineering produced an empty dataframe — "
                "check that required audio features (energy, valence, danceability, acousticness) are non-null")
            self.assertGreater(X.shape[0], 0, "Feature matrix has no rows")
            self.assertGreater(X.shape[1], 0, "Feature matrix has no columns")

            logger.info(f"TEST 2.7 PASSED: feature matrix shape {X.shape}, genre cols: {len(genre_cols)}")
            print(f"  Feature matrix: {X.shape[0]} tracks x {X.shape[1]} features")
            print(f"  Genre columns: {len(genre_cols)}")

        except Exception as e:
            logger.error(f"TEST 2.7 FAILED: {e}")
            self.fail(f"run_feature_engineering raised: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
class VerboseTestResult(unittest.TextTestResult):
    pass


def run_tests():
    print("\n" + "=" * 65)
    print("  INGEST PIPELINE TEST SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None

    runner = unittest.TextTestRunner(
        resultclass=VerboseTestResult,
        verbosity=2 if "-v" in sys.argv else 1,
        stream=sys.stdout,
    )

    print("\n-- Suite 1: Raw Data " + "-" * 44)
    s1_result = runner.run(loader.loadTestsFromTestCase(TestRawData))

    print("\n-- Suite 2: Pipeline Output " + "-" * 37)
    s2_result = runner.run(loader.loadTestsFromTestCase(TestAssembledRecords))

    total_run      = s1_result.testsRun + s2_result.testsRun
    total_failures = len(s1_result.failures) + len(s2_result.failures)
    total_errors   = len(s1_result.errors)   + len(s2_result.errors)
    total_passed   = total_run - total_failures - total_errors

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Total tests run : {total_run}")
    print(f"  Passed          : {total_passed}")
    print(f"  Failed          : {total_failures}")
    print(f"  Errors          : {total_errors}")
    status = "ALL PASSED" if total_failures == 0 and total_errors == 0 else "SOME FAILED"
    print(f"  Result          : {status}")
    print("=" * 65)

    if total_failures > 0 or total_errors > 0:
        print("\n  Failed tests:")
        for result in [s1_result, s2_result]:
            for test, msg in result.failures + result.errors:
                print(f"  - {test}")
                print(f"    -> {msg.strip().split(chr(10))[-1]}")

    print(f"\n  Full log written to: ingest_tests.log\n")
    logger.info(f"Test run complete: {total_passed}/{total_run} passed")

    return total_failures + total_errors


if __name__ == "__main__":
    sys.exit(run_tests())
