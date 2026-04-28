import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

# These scopes define exactly what data we're allowed to pull
SCOPES = " ".join([
    "user-read-recently-played",      # recently played tracks
    "user-library-read",              # liked/saved songs
    "user-top-read",                  # top tracks and artists
    "playlist-read-private",          # your private playlists
    "playlist-modify-public",         # write public playlists later
    "playlist-modify-private",        # write private playlists later
])

def get_spotify_client() -> spotipy.Spotify:
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope=SCOPES,
        cache_path=".spotify_cache",   # saves token so you don't re-auth every run
        open_browser=True
    )
    return spotipy.Spotify(auth_manager=auth_manager)