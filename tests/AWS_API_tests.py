"""
AWS_API_tests.py
================
Three test suites for the AWS Bedrock integration:

  Suite 1 — Connection Tests        : Can we reach AWS and authenticate?
  Suite 2 — Feature Estimation Tests : Can the LLM estimate audio features
                                       for tracks pulled live from your
                                       Spotify library?
  Suite 3 — Genre Estimation Tests  : Can the LLM estimate genre tags for
                                       artists whose Spotify genres are empty?

Run with:
    python tests/AWS_API_tests.py           # summary report
    python tests/AWS_API_tests.py -v        # verbose

Requirements:
    pip install boto3 spotipy python-dotenv

.env file must exist in the project root with:
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    AWS_DEFAULT_REGION=us-east-1
    SPOTIFY_CLIENT_ID=...
    SPOTIFY_CLIENT_SECRET=...
    SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
"""

import os
import sys
import json
import unittest
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 output on Windows so box-drawing / tick characters don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── resolve project root so imports and cache paths work regardless of cwd ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Service_authentications"))

load_dotenv(PROJECT_ROOT / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="aws_api_tests.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "amazon.nova-lite-v1:0",
)
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

_FALLBACK_TRACKS = [
    {"id": "0VjIjW4GlUZAMYd2vXMi3b", "name": "Blinding Lights", "artists": [{"id": "1Xyo4u8uXC1ZmMpatF05PJ", "name": "The Weeknd"}]},
    {"id": "7qiZfU4dY1lWllzX7mPBI3", "name": "Shape of You",    "artists": [{"id": "6eUKZXaKkcviH0Ku9w2n3V", "name": "Ed Sheeran"}]},
    {"id": "3n3Ppam7vgaVa1iaRUIOKE", "name": "Mr. Brightside",  "artists": [{"id": "0C0XlULifJtAgn6ZNCW2eu", "name": "The Killers"}]},
]


# ─────────────────────────────────────────────────────────────────────────────
# Fetch real tracks from Spotify using the cached OAuth token.
# No browser prompt — if the cache is missing or expired this returns None
# and Suite 2 falls back to the hardcoded tracks above.
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_spotify_tracks(limit: int = 10):
    """
    Returns a list of slim track dicts using the cached Spotify token.
    Pulls recently_played first; falls back to top tracks if empty.
    Returns None if the cache is unavailable or the call fails.
    """
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth

        cache_path = str(PROJECT_ROOT / ".spotify_cache")
        auth_manager = SpotifyOAuth(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
            scope="user-read-recently-played user-library-read user-top-read playlist-read-private playlist-modify-public playlist-modify-private",
            cache_path=cache_path,
            open_browser=False,   # never block during tests
        )

        cached = auth_manager.get_cached_token()
        if not cached:
            logger.warning("No cached Spotify token — Suite 2 will use fallback tracks")
            return None

        sp = spotipy.Spotify(auth_manager=auth_manager)

        # try recently played first
        results = sp.current_user_recently_played(limit=limit)
        items = results.get("items", [])

        if not items:
            # fall back to top tracks
            results = sp.current_user_top_tracks(limit=limit, time_range="short_term")
            items = [{"track": t} for t in results.get("items", [])]

        tracks = []
        seen = set()
        for item in items:
            t = item["track"]
            if t["id"] not in seen:
                seen.add(t["id"])
                tracks.append({
                    "id":      t["id"],
                    "name":    t["name"],
                    "artists": t["artists"],
                })

        logger.info(f"Fetched {len(tracks)} tracks from Spotify for Suite 2")
        return tracks if tracks else None

    except Exception as e:
        logger.warning(f"Spotify fetch failed — Suite 2 will use fallback tracks: {e}")
        return None


# Fetch once at module load; Suite 2 & 3 tests share these lists
_spotify_tracks = _fetch_spotify_tracks(limit=10)
SAMPLE_TRACKS   = _spotify_tracks if _spotify_tracks else _FALLBACK_TRACKS
_TRACKS_SOURCE  = "Spotify (live)" if _spotify_tracks else "fallback (hardcoded)"

# Deduplicated artist list derived from SAMPLE_TRACKS — used by Suite 3
_seen_artist_ids: set = set()
SAMPLE_ARTISTS: list = []
for _t in SAMPLE_TRACKS:
    for _a in _t.get("artists", []):
        if _a.get("id") and _a["id"] not in _seen_artist_ids:
            _seen_artist_ids.add(_a["id"])
            SAMPLE_ARTISTS.append({
                "id":          _a["id"],
                "name":        _a["name"],
                "track_names": [_t["name"]],
            })


# ─────────────────────────────────────────────────────────────────────────────
# Shared AWS clients — built once, reused across all tests
# ─────────────────────────────────────────────────────────────────────────────
def _build_sts():
    try:
        import boto3
        return boto3.client("sts", region_name=REGION)
    except Exception as e:
        logger.error(f"Failed to build STS client: {e}")
        return None


def _build_bedrock():
    try:
        import boto3
        return boto3.client("bedrock-runtime", region_name=REGION)
    except Exception as e:
        logger.error(f"Failed to build Bedrock client: {e}")
        return None


STS     = _build_sts()
BEDROCK = _build_bedrock()


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 1 — CONNECTION TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestAWSConnection(unittest.TestCase):
    """
    Suite 1: Can we connect to AWS and reach Bedrock?
    Validates credentials and permissions before any model inference.
    """

    # ── Test 1.1 ─────────────────────────────────────────────────────────────
    def test_1_1_env_credentials_present(self):
        """Checks that all required AWS env variables are set."""
        logger.info("TEST 1.1: Checking AWS env credentials")

        key_id     = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region     = os.getenv("AWS_DEFAULT_REGION")

        self.assertIsNotNone(key_id,
            "AWS_ACCESS_KEY_ID is missing from .env")
        self.assertIsNotNone(secret_key,
            "AWS_SECRET_ACCESS_KEY is missing from .env")
        self.assertIsNotNone(region,
            "AWS_DEFAULT_REGION is missing from .env")
        self.assertNotEqual(key_id, "",    "AWS_ACCESS_KEY_ID is empty in .env")
        self.assertNotEqual(secret_key, "", "AWS_SECRET_ACCESS_KEY is empty in .env")

        logger.info("TEST 1.1 PASSED: All AWS credentials present")

    # ── Test 1.2 ─────────────────────────────────────────────────────────────
    def test_1_2_boto3_importable(self):
        """Checks that boto3 is installed. Run 'pip install boto3' if this fails."""
        logger.info("TEST 1.2: Checking boto3 is installed")
        try:
            import boto3
            version = boto3.__version__
            logger.info(f"TEST 1.2 PASSED: boto3 {version}")
            print(f"\n  ✓ boto3 version: {version}")
        except ImportError:
            self.fail("boto3 is not installed. Run: pip install boto3")

    # ── Test 1.3 ─────────────────────────────────────────────────────────────
    def test_1_3_aws_identity_reachable(self):
        """
        Calls STS GetCallerIdentity — the simplest authenticated AWS call.
        Confirms credentials are valid and AWS is reachable.
        """
        logger.info("TEST 1.3: Calling STS GetCallerIdentity")
        self.assertIsNotNone(STS, "STS client not available — check boto3 install")

        try:
            identity = STS.get_caller_identity()

            self.assertIn("Account", identity, "STS response missing 'Account'")
            self.assertIn("UserId",  identity, "STS response missing 'UserId'")
            self.assertIn("Arn",     identity, "STS response missing 'Arn'")

            logger.info(f"TEST 1.3 PASSED: {identity['Arn']} (account={identity['Account']})")
            print(f"\n  ✓ AWS identity : {identity['Arn']}")
            print(f"  ✓ Account      : {identity['Account']}")

        except Exception as e:
            logger.error(f"TEST 1.3 FAILED: {e}")
            self.fail(
                f"STS GetCallerIdentity failed: {e}\n"
                "Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are correct."
            )

    # ── Test 1.4 ─────────────────────────────────────────────────────────────
    def test_1_4_bedrock_client_builds(self):
        """Checks that a bedrock-runtime client was created successfully."""
        logger.info("TEST 1.4: Checking Bedrock client built successfully")
        self.assertIsNotNone(BEDROCK,
            f"Bedrock client is None — verify region '{REGION}' supports Bedrock")
        logger.info("TEST 1.4 PASSED: Bedrock client object created")
        print(f"\n  ✓ Bedrock client ready (region={REGION})")

    # ── Test 1.5 ─────────────────────────────────────────────────────────────
    def test_1_5_bedrock_invoke_permission(self):
        """
        Sends a minimal request to Bedrock to confirm bedrock:InvokeModel
        permission and that the model is enabled in your account.
        """
        logger.info("TEST 1.5: Checking bedrock:InvokeModel permission with Amazon Nova Lite")
        self.assertIsNotNone(BEDROCK, "Bedrock client not available — skipping")

        body = json.dumps({
            "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
            "inferenceConfig": {"maxTokens": 10},
        })

        try:
            response = BEDROCK.invoke_model(
                modelId=MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())

            self.assertIn("output", result, f"Nova response missing 'output' key: {result}")
            text = result["output"]["message"]["content"][0]["text"]
            self.assertIsInstance(text, str, "Nova response content is not a string")

            logger.info(f"TEST 1.5 PASSED: bedrock:InvokeModel works for {MODEL_ID}")
            print(f"\n  ✓ bedrock:InvokeModel confirmed")
            print(f"  ✓ Model: {MODEL_ID}")

        except Exception as e:
            logger.error(f"TEST 1.5 FAILED: {e}")
            err = str(e)
            if "AccessDeniedException" in err:
                self.fail(
                    f"Access denied — update your IAM policy so 'bedrock:InvokeModel' "
                    f"allows resource 'arn:aws:bedrock:{REGION}::foundation-model/*'."
                )
            elif "ResourceNotFoundException" in err or "ValidationException" in err:
                self.fail(
                    f"Model '{MODEL_ID}' is not enabled in region '{REGION}'. "
                    "Go to AWS Console → Bedrock → Model access, enable 'Nova Lite' "
                    "(Amazon section, instant approval), then re-run."
                )
            else:
                self.fail(f"Bedrock invoke failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 2 — FEATURE ESTIMATION TESTS
# Uses tracks fetched live from your Spotify library (falls back to hardcoded
# tracks only if the cached Spotify token is unavailable).
# ═════════════════════════════════════════════════════════════════════════════
class TestAudioFeatureEstimation(unittest.TestCase):
    """
    Suite 2: Can the LLM estimate audio features for your actual Spotify tracks?
    Each test validates a specific aspect of the output that the downstream
    feature engineering pipeline (features.py) depends on.
    """

    @classmethod
    def setUpClass(cls):
        print(f"\n  Track source : {_TRACKS_SOURCE}")
        print(f"  Track count  : {len(SAMPLE_TRACKS)}")
        for t in SAMPLE_TRACKS:
            artist = t["artists"][0]["name"] if t.get("artists") else "Unknown"
            print(f"    • \"{t['name']}\" — {artist}")

    # ── Test 2.1 ─────────────────────────────────────────────────────────────
    def test_2_1_returns_result_for_every_track(self):
        """
        Sends the Spotify tracks to the LLM and checks we get one result
        back per track. A missing entry means the JSON was malformed or
        a track was silently dropped.
        """
        logger.info("TEST 2.1: Checking one result returned per input track")
        self.assertIsNotNone(BEDROCK, "Bedrock client not available")

        try:
            from llm_features import estimate_audio_features
            result = estimate_audio_features(SAMPLE_TRACKS)

            self.assertEqual(
                len(result), len(SAMPLE_TRACKS),
                f"Expected {len(SAMPLE_TRACKS)} results, got {len(result)}"
            )
            for track in SAMPLE_TRACKS:
                self.assertIn(track["id"], result,
                    f"No result for '{track['name']}' (id={track['id']})")

            logger.info(f"TEST 2.1 PASSED: {len(result)} results returned")
            print(f"\n  ✓ All {len(result)} tracks returned a result")

        except Exception as e:
            logger.error(f"TEST 2.1 FAILED: {e}")
            self.fail(f"estimate_audio_features raised an exception: {e}")

    # ── Test 2.2 ─────────────────────────────────────────────────────────────
    def test_2_2_all_required_fields_present(self):
        """
        Checks every field that ingest.py reads from the features dict is present.
        A missing field becomes a silent None in the assembled record and causes
        that row to be dropped by build_dataframe() in features.py.
        """
        logger.info("TEST 2.2: Checking all required fields are present")
        self.assertIsNotNone(BEDROCK, "Bedrock client not available")

        required_fields = [
            "energy", "valence", "danceability", "acousticness",
            "speechiness", "instrumentalness", "liveness",
            "loudness", "tempo", "key", "mode", "time_signature",
        ]

        try:
            from llm_features import estimate_audio_features
            result = estimate_audio_features(SAMPLE_TRACKS[:1])
            features = next(iter(result.values()))

            missing = [f for f in required_fields if f not in features]
            self.assertEqual(missing, [],
                f"Missing fields in LLM response: {missing}")

            logger.info(f"TEST 2.2 PASSED: All {len(required_fields)} required fields present")
            print(f"  ✓ All {len(required_fields)} required fields present")

        except Exception as e:
            logger.error(f"TEST 2.2 FAILED: {e}")
            self.fail(f"Field check failed: {e}")

    # ── Test 2.3 ─────────────────────────────────────────────────────────────
    def test_2_3_continuous_features_in_valid_range(self):
        """
        Checks that all 0-1 float features are within [0, 1].
        Out-of-range values corrupt the feature matrix passed to UMAP —
        MinMaxScaler would clip them silently.
        """
        logger.info("TEST 2.3: Checking float features are in [0, 1]")
        self.assertIsNotNone(BEDROCK, "Bedrock client not available")

        bounded_fields = [
            "energy", "valence", "danceability", "acousticness",
            "speechiness", "instrumentalness", "liveness",
        ]

        try:
            from llm_features import estimate_audio_features
            result = estimate_audio_features(SAMPLE_TRACKS)

            for track_id, features in result.items():
                for field in bounded_fields:
                    value = features.get(field)
                    self.assertIsNotNone(value,
                        f"'{field}' is None for track {track_id}")
                    self.assertGreaterEqual(float(value), 0.0,
                        f"'{field}' = {value} is below 0.0 for track {track_id}")
                    self.assertLessEqual(float(value), 1.0,
                        f"'{field}' = {value} is above 1.0 for track {track_id}")

            logger.info("TEST 2.3 PASSED: All bounded features within [0, 1]")
            print(f"  ✓ All bounded features within [0.0, 1.0]")

        except Exception as e:
            logger.error(f"TEST 2.3 FAILED: {e}")
            self.fail(f"Range check failed: {e}")

    # ── Test 2.4 ─────────────────────────────────────────────────────────────
    def test_2_4_tempo_and_loudness_in_valid_range(self):
        """
        Checks that tempo (BPM) and loudness (dB) are within physically
        plausible ranges. features.py normalises both, so extreme values
        would silently pin every track at 0 or 1 after clipping.
        """
        logger.info("TEST 2.4: Checking tempo and loudness ranges")
        self.assertIsNotNone(BEDROCK, "Bedrock client not available")

        try:
            from llm_features import estimate_audio_features
            result = estimate_audio_features(SAMPLE_TRACKS)

            for track_id, features in result.items():
                tempo    = features.get("tempo")
                loudness = features.get("loudness")

                self.assertIsNotNone(tempo,    f"tempo is None for {track_id}")
                self.assertIsNotNone(loudness, f"loudness is None for {track_id}")

                self.assertGreaterEqual(float(tempo), 40.0,
                    f"tempo {tempo} BPM is unrealistically slow for {track_id}")
                self.assertLessEqual(float(tempo), 220.0,
                    f"tempo {tempo} BPM is unrealistically fast for {track_id}")
                self.assertGreaterEqual(float(loudness), -60.0,
                    f"loudness {loudness} dB is below -60 for {track_id}")
                self.assertLessEqual(float(loudness), 0.0,
                    f"loudness {loudness} dB is above 0 for {track_id}")

            logger.info("TEST 2.4 PASSED: tempo and loudness in valid ranges")
            print(f"  ✓ tempo (40–220 BPM) and loudness (-60–0 dB) in range")

        except Exception as e:
            logger.error(f"TEST 2.4 FAILED: {e}")
            self.fail(f"tempo/loudness range check failed: {e}")

    # ── Test 2.5 ─────────────────────────────────────────────────────────────
    def test_2_5_categorical_fields_are_valid_integers(self):
        """
        Checks that key (0-11), mode (0 or 1), and time_signature (3-7)
        are integers in their expected ranges. These feed into harmonic_sig
        in features.py — wrong types or values produce a silent 0.
        """
        logger.info("TEST 2.5: Checking categorical field types and ranges")
        self.assertIsNotNone(BEDROCK, "Bedrock client not available")

        try:
            from llm_features import estimate_audio_features
            result = estimate_audio_features(SAMPLE_TRACKS)

            for track_id, features in result.items():
                key      = features.get("key")
                mode     = features.get("mode")
                time_sig = features.get("time_signature")

                self.assertIsNotNone(key,      f"key is None for {track_id}")
                self.assertIsNotNone(mode,     f"mode is None for {track_id}")
                self.assertIsNotNone(time_sig, f"time_signature is None for {track_id}")

                self.assertIn(int(key), range(12),
                    f"key={key} is not in 0–11 for {track_id}")
                self.assertIn(int(mode), [0, 1],
                    f"mode={mode} is not 0 or 1 for {track_id}")
                self.assertIn(int(time_sig), range(3, 8),
                    f"time_signature={time_sig} is not in 3–7 for {track_id}")

            logger.info("TEST 2.5 PASSED: key, mode, time_signature all valid")
            print(f"  ✓ key (0–11), mode (0/1), time_signature (3–7) all valid")

        except Exception as e:
            logger.error(f"TEST 2.5 FAILED: {e}")
            self.fail(f"Categorical field check failed: {e}")

    # ── Test 2.6 ─────────────────────────────────────────────────────────────
    def test_2_6_estimated_flag_is_set(self):
        """
        Checks the 'estimated' flag is True on every result so downstream
        code can distinguish LLM estimates from real Spotify measurements.
        """
        logger.info("TEST 2.6: Checking 'estimated' flag is present and True")
        self.assertIsNotNone(BEDROCK, "Bedrock client not available")

        try:
            from llm_features import estimate_audio_features
            result = estimate_audio_features(SAMPLE_TRACKS[:1])
            features = next(iter(result.values()))

            self.assertIn("estimated", features,
                "Field 'estimated' missing from LLM response")
            self.assertTrue(features["estimated"],
                "Field 'estimated' is not True")

            logger.info("TEST 2.6 PASSED: 'estimated' flag present and True")
            print(f"  ✓ 'estimated' flag is present and True")

        except Exception as e:
            logger.error(f"TEST 2.6 FAILED: {e}")
            self.fail(f"estimated flag check failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Custom test runner
# ─────────────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
# SUITE 3 — GENRE ESTIMATION TESTS
# Verifies that estimate_artist_genres returns valid Spotify-style genre tags
# for artists who came back with an empty genre list from Spotify's API.
# ═════════════════════════════════════════════════════════════════════════════
class TestGenreEstimation(unittest.TestCase):
    """
    Suite 3: Can the LLM fill in missing artist genre tags?
    Mirrors the backfill step in run_ingest() where artists_missing_genres
    are sent to estimate_artist_genres before assembly.
    """

    @classmethod
    def setUpClass(cls):
        print(f"\n  Artist source : {_TRACKS_SOURCE}")
        print(f"  Artist count  : {len(SAMPLE_ARTISTS)}")
        for a in SAMPLE_ARTISTS:
            print(f"    • \"{a['name']}\" (tracks: {', '.join(a['track_names'])})")

    def test_3_1_returns_result_for_every_artist(self):
        """
        Checks we get one genre list back per artist submitted.
        A missing entry means the artist was silently dropped — their
        assembled records will have an empty genre list in features.py.
        """
        logger.info("TEST 3.1: Checking one result returned per input artist")
        self.assertIsNotNone(BEDROCK, "Bedrock client not available")
        self.assertGreater(len(SAMPLE_ARTISTS), 0, "No sample artists to test")

        try:
            from llm_features import estimate_artist_genres
            result = estimate_artist_genres(SAMPLE_ARTISTS)

            self.assertEqual(
                len(result), len(SAMPLE_ARTISTS),
                f"Expected {len(SAMPLE_ARTISTS)} results, got {len(result)}"
            )
            for artist in SAMPLE_ARTISTS:
                self.assertIn(artist["id"], result,
                    f"No result for '{artist['name']}' (id={artist['id']})")

            logger.info(f"TEST 3.1 PASSED: {len(result)} genre lists returned")
            print(f"\n  All {len(result)} artists returned a genre list")

        except Exception as e:
            logger.error(f"TEST 3.1 FAILED: {e}")
            self.fail(f"estimate_artist_genres raised an exception: {e}")

    def test_3_2_genres_are_non_empty_lists(self):
        """
        Checks that every artist got at least one genre tag and that
        the value is a list. encode_genre_tags in features.py calls
        MultiLabelBinarizer on this field — a non-list or empty list
        means the artist contributes nothing to the genre feature columns.
        """
        logger.info("TEST 3.2: Checking genre values are non-empty lists")
        self.assertIsNotNone(BEDROCK, "Bedrock client not available")

        try:
            from llm_features import estimate_artist_genres
            result = estimate_artist_genres(SAMPLE_ARTISTS)

            for artist_id, genres in result.items():
                self.assertIsInstance(genres, list,
                    f"Genres for {artist_id} is {type(genres).__name__}, expected list")
                self.assertGreater(len(genres), 0,
                    f"Empty genre list returned for artist {artist_id}")

            logger.info("TEST 3.2 PASSED: all genre values are non-empty lists")
            print(f"  All {len(result)} artists have at least one genre tag")

        except Exception as e:
            logger.error(f"TEST 3.2 FAILED: {e}")
            self.fail(f"Genre list check failed: {e}")

    def test_3_3_genres_are_lowercase_strings(self):
        """
        Checks that every genre tag is a lowercase string.
        Spotify genre tags are always lowercase and hyphenated
        (e.g. 'afro gospel', 'congolese rumba'). Uppercase tags would
        create duplicate genre columns in the multi-hot encoding step.
        """
        logger.info("TEST 3.3: Checking genre tags are lowercase strings")
        self.assertIsNotNone(BEDROCK, "Bedrock client not available")

        try:
            from llm_features import estimate_artist_genres
            result = estimate_artist_genres(SAMPLE_ARTISTS)

            for artist_id, genres in result.items():
                for tag in genres:
                    self.assertIsInstance(tag, str,
                        f"Genre tag {tag!r} for {artist_id} is not a string")
                    self.assertEqual(tag, tag.lower(),
                        f"Genre tag {tag!r} for {artist_id} is not lowercase")

            all_tags = [tag for genres in result.values() for tag in genres]
            logger.info(f"TEST 3.3 PASSED: {len(all_tags)} genre tags, all lowercase strings")
            print(f"  {len(all_tags)} genre tags across {len(result)} artists — all lowercase")
            sample = list({t for tags in result.values() for t in tags})[:6]
            print(f"  Sample tags: {sample}")

        except Exception as e:
            logger.error(f"TEST 3.3 FAILED: {e}")
            self.fail(f"Genre string check failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Custom test runner
# ─────────────────────────────────────────────────────────────────────────────
class VerboseTestResult(unittest.TextTestResult):
    pass


def run_tests():
    print("\n" + "=" * 65)
    print("  AWS BEDROCK API TEST SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model  : {MODEL_ID}")
    print(f"  Region : {REGION}")
    print("=" * 65)

    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None

    runner = unittest.TextTestRunner(
        resultclass=VerboseTestResult,
        verbosity=2 if "-v" in sys.argv else 1,
        stream=sys.stdout
    )

    print("\n-- Suite 1: Connection Tests " + "-" * 36)
    conn_result = runner.run(loader.loadTestsFromTestCase(TestAWSConnection))

    print("\n-- Suite 2: Feature Estimation Tests " + "-" * 28)
    feat_result = runner.run(loader.loadTestsFromTestCase(TestAudioFeatureEstimation))

    print("\n-- Suite 3: Genre Estimation Tests " + "-" * 30)
    genre_result = runner.run(loader.loadTestsFromTestCase(TestGenreEstimation))

    total_run      = conn_result.testsRun + feat_result.testsRun + genre_result.testsRun
    total_failures = len(conn_result.failures) + len(feat_result.failures) + len(genre_result.failures)
    total_errors   = len(conn_result.errors)   + len(feat_result.errors)   + len(genre_result.errors)
    total_passed   = total_run - total_failures - total_errors

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Total tests run : {total_run}")
    print(f"  Passed          : {total_passed}")
    print(f"  Failed          : {total_failures}")
    print(f"  Errors          : {total_errors}")
    print(f"  Result          : {'ALL PASSED' if total_failures == 0 and total_errors == 0 else 'SOME FAILED'}")
    print("=" * 65)

    if total_failures > 0 or total_errors > 0:
        print("\n  Failed tests:")
        for result in [conn_result, feat_result, genre_result]:
            for failure in result.failures + result.errors:
                test, msg = failure
                print(f"  - {test}")
                print(f"    → {msg.strip().split(chr(10))[-1]}")

    print(f"\n  Full log written to: aws_api_tests.log\n")
    logger.info(f"Test run complete: {total_passed}/{total_run} passed")

    return total_failures + total_errors


if __name__ == "__main__":
    sys.exit(run_tests())
