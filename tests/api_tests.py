"""
api_tests.py
============
Two test suites for the Spotify Web API integration:

  Suite 1 - Connection Tests  : Can we actually reach and authenticate with Spotify?
  Suite 2 - Data Access Tests : Can we read the specific data we need?

Note: Spotify's audio_features and batch /artists?ids= endpoints are blocked in
Development Mode (deprecated Nov 2024). Those tests have been removed. Artist
metadata is validated via the individual /artists/{id} endpoint which still works.

Run with:
    python tests/api_tests.py           # summary report
    python tests/api_tests.py -v        # verbose output
"""

import os
import sys
import time
import unittest
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    filename="api_tests.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Spotify client — built once, reused across all tests
# ─────────────────────────────────────────────────────────────────────────────
def build_spotify_client():
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
        import subprocess

        client_id     = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        redirect_uri  = os.getenv("SPOTIFY_REDIRECT_URI")

        if not all([client_id, client_secret, redirect_uri]):
            return None

        scopes = " ".join([
            "user-read-recently-played",
            "user-library-read",
            "user-top-read",
            "playlist-read-private",
        ])

        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scopes,
            cache_path=str(PROJECT_ROOT / ".spotify_cache"),
            open_browser=False,
            show_dialog=True,
        )

        cached_token = auth_manager.get_cached_token()

        if not cached_token:
            auth_url = auth_manager.get_authorize_url()
            print("\n" + "=" * 80)
            print("  SPOTIFY AUTHORIZATION REQUIRED")
            print("=" * 80)
            print(f"\n  Open this URL in your browser:\n\n  {auth_url}\n")
            print("  After authorizing, paste the full redirect URL below.")
            print("-" * 80)

            try:
                if sys.platform == "win32":
                    os.startfile(auth_url)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", auth_url])
                else:
                    subprocess.Popen(["xdg-open", auth_url])
            except Exception:
                pass

            redirect_url = input("  Paste the redirect URL here: ").strip()
            print("-" * 80 + "\n")

            if not redirect_url:
                raise ValueError("No redirect URL provided")

            code = auth_manager.parse_response_code(redirect_url)
            if not code:
                raise ValueError("Could not extract authorization code from URL")

            auth_manager.get_access_token(code)

        return spotipy.Spotify(auth_manager=auth_manager)

    except Exception as e:
        logger.error(f"Failed to build Spotify client: {e}")
        print(f"\nError during authorization: {e}")
        return None


SP = build_spotify_client()


# =============================================================================
# SUITE 1 — CONNECTION TESTS
# =============================================================================
class TestSpotifyConnection(unittest.TestCase):

    def test_1_1_env_credentials_present(self):
        """Checks that all three required env variables are set."""
        logger.info("TEST 1.1: Checking env credentials")

        client_id     = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        redirect_uri  = os.getenv("SPOTIFY_REDIRECT_URI")

        self.assertIsNotNone(client_id,     "SPOTIFY_CLIENT_ID is missing from .env")
        self.assertIsNotNone(client_secret, "SPOTIFY_CLIENT_SECRET is missing from .env")
        self.assertIsNotNone(redirect_uri,  "SPOTIFY_REDIRECT_URI is missing from .env")
        self.assertNotEqual(client_id, "",  "SPOTIFY_CLIENT_ID is empty in .env")
        self.assertNotEqual(client_secret, "", "SPOTIFY_CLIENT_SECRET is empty in .env")
        self.assertIn("127.0.0.1", redirect_uri,
            "Redirect URI should use 127.0.0.1 not localhost")

        logger.info("TEST 1.1 PASSED")

    def test_1_2_spotipy_importable(self):
        """Checks that the spotipy library is installed."""
        logger.info("TEST 1.2: Checking spotipy is installed")
        try:
            import spotipy
            version = getattr(spotipy, "__version__", "unknown")
            print(f"\n  spotipy version: {version}")
            logger.info(f"TEST 1.2 PASSED: spotipy {version}")
        except ImportError:
            self.fail("spotipy is not installed. Run: pip install spotipy")

    def test_1_3_client_builds_without_error(self):
        """Checks the Spotify client object was created successfully."""
        logger.info("TEST 1.3: Checking Spotify client built successfully")
        self.assertIsNotNone(SP,
            "Spotify client is None — check credentials and re-run to trigger OAuth flow")
        logger.info("TEST 1.3 PASSED")

    def test_1_4_can_reach_api_and_get_current_user(self):
        """Makes a real API call to /me — the simplest authenticated endpoint."""
        logger.info("TEST 1.4: Calling /me endpoint")
        self.assertIsNotNone(SP, "Spotify client not available")

        try:
            user = SP.current_user()
            self.assertIsNotNone(user, "API returned None for current user")
            self.assertIn("id",           user, "User object missing 'id'")
            self.assertIn("display_name", user, "User object missing 'display_name'")

            logger.info(f"TEST 1.4 PASSED: connected as {user.get('display_name')}")
            print(f"\n  Connected as: {user.get('display_name')} ({user.get('id')})")

        except Exception as e:
            logger.error(f"TEST 1.4 FAILED: {e}")
            self.fail(f"API call to /me failed: {e}")

    def test_1_5_token_has_required_scopes(self):
        """Verifies the cached token includes all the scopes we need."""
        logger.info("TEST 1.5: Checking token scopes")
        self.assertIsNotNone(SP, "Spotify client not available")

        required_scopes = [
            "user-read-recently-played",
            "user-library-read",
            "user-top-read",
            "playlist-read-private",
        ]

        try:
            token_info = SP.auth_manager.get_cached_token()
            self.assertIsNotNone(token_info, "No cached token found")

            granted = token_info.get("scope", "").split()
            missing = [s for s in required_scopes if s not in granted]

            self.assertEqual(missing, [],
                f"Token missing scopes: {missing}. "
                "Delete .spotify_cache and re-authenticate.")

            logger.info("TEST 1.5 PASSED: all required scopes granted")

        except Exception as e:
            logger.error(f"TEST 1.5 FAILED: {e}")
            self.fail(f"Scope check failed: {e}")


# =============================================================================
# SUITE 2 — DATA ACCESS TESTS
# =============================================================================
class TestSpotifyDataAccess(unittest.TestCase):

    def test_2_1_recently_played_returns_tracks(self):
        """
        Checks we can pull recently played tracks and that each item has the
        minimum fields needed for behavioral signals (id, name, artist, played_at).
        """
        logger.info("TEST 2.1: Fetching recently played tracks")
        self.assertIsNotNone(SP, "Spotify client not available")

        try:
            results = SP.current_user_recently_played(limit=10)

            self.assertIn("items", results, "Response missing 'items'")
            self.assertGreater(len(results["items"]), 0,
                "No recently played tracks — play some songs on Spotify first")

            item  = results["items"][0]
            track = item["track"]
            self.assertIn("played_at", item,    "Item missing 'played_at' timestamp")
            self.assertIn("id",        track,   "Track missing 'id'")
            self.assertIn("name",      track,   "Track missing 'name'")
            self.assertIn("artists",   track,   "Track missing 'artists'")
            self.assertGreater(len(track["artists"]), 0, "Track has no artists")

            count = len(results["items"])
            logger.info(f"TEST 2.1 PASSED: {count} recently played tracks")
            print(f"\n  Recently played: {count} tracks. Latest: '{track['name']}'")

        except Exception as e:
            logger.error(f"TEST 2.1 FAILED: {e}")
            self.fail(f"recently_played fetch failed: {e}")

    def test_2_2_liked_songs_returns_tracks(self):
        """
        Checks we can read the saved tracks library. Validates id, name, artist,
        added_at, explicit, and duration_ms — all used in ingest assembly.
        """
        logger.info("TEST 2.2: Fetching liked songs")
        self.assertIsNotNone(SP, "Spotify client not available")

        try:
            results = SP.current_user_saved_tracks(limit=10)

            self.assertIn("items", results, "Response missing 'items'")
            self.assertGreater(len(results["items"]), 0,
                "No liked songs — like some songs on Spotify first")

            item  = results["items"][0]
            track = item["track"]
            self.assertIn("added_at",    item,  "Item missing 'added_at'")
            self.assertIn("track",       item,  "Item missing 'track'")
            self.assertIn("id",          track, "Track missing 'id'")
            self.assertIn("name",        track, "Track missing 'name'")
            self.assertIn("explicit",    track, "Track missing 'explicit'")
            self.assertIn("duration_ms", track, "Track missing 'duration_ms'")

            count = len(results["items"])
            logger.info(f"TEST 2.2 PASSED: {count} liked songs")
            print(f"  Liked songs: {count} returned. First: '{track['name']}'")

        except Exception as e:
            logger.error(f"TEST 2.2 FAILED: {e}")
            self.fail(f"liked_songs fetch failed: {e}")

    def test_2_3_top_tracks_all_time_ranges(self):
        """
        Checks top tracks across all 3 time ranges. All 3 must be accessible
        because comparing short vs long term is how we detect taste drift.
        """
        logger.info("TEST 2.3: Fetching top tracks across time ranges")
        self.assertIsNotNone(SP, "Spotify client not available")

        for time_range in ["short_term", "medium_term", "long_term"]:
            try:
                results = SP.current_user_top_tracks(limit=10, time_range=time_range)

                self.assertIn("items", results,
                    f"Response for {time_range} missing 'items'")
                self.assertGreater(len(results["items"]), 0,
                    f"No top tracks for {time_range} — need more listening history")

                track = results["items"][0]
                self.assertIn("id",   track, f"{time_range} track missing 'id'")
                self.assertIn("name", track, f"{time_range} track missing 'name'")

                count = len(results["items"])
                logger.info(f"TEST 2.3 PASSED [{time_range}]: {count} tracks")
                print(f"  Top tracks ({time_range}): {count}. Top: '{track['name']}'")
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"TEST 2.3 FAILED [{time_range}]: {e}")
                self.fail(f"top_tracks failed for {time_range}: {e}")



# ─────────────────────────────────────────────────────────────────────────────
# Custom runner + summary
# ─────────────────────────────────────────────────────────────────────────────
class VerboseTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)

    def addFailure(self, test, err):
        super().addFailure(test, err)

    def addError(self, test, err):
        super().addError(test, err)


def run_tests():
    print("\n" + "=" * 65)
    print("  SPOTIFY API TEST SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None

    runner = unittest.TextTestRunner(
        resultclass=VerboseTestResult,
        verbosity=2 if "-v" in sys.argv else 1,
        stream=sys.stdout,
    )

    print("\n-- Suite 1: Connection Tests " + "-" * 36)
    conn_result = runner.run(loader.loadTestsFromTestCase(TestSpotifyConnection))

    print("\n-- Suite 2: Data Access Tests " + "-" * 35)
    data_result = runner.run(loader.loadTestsFromTestCase(TestSpotifyDataAccess))

    total_run      = conn_result.testsRun + data_result.testsRun
    total_failures = len(conn_result.failures) + len(data_result.failures)
    total_errors   = len(conn_result.errors)   + len(data_result.errors)
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
        for result in [conn_result, data_result]:
            for test, msg in result.failures + result.errors:
                print(f"  - {test}")
                print(f"    -> {msg.strip().split(chr(10))[-1]}")

    print(f"\n  Full log written to: api_tests.log\n")
    logger.info(f"Test run complete: {total_passed}/{total_run} passed")

    return total_failures + total_errors


if __name__ == "__main__":
    sys.exit(run_tests())
