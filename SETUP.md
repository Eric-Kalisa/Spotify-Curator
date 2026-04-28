# Spotify-Curator Setup Guide

## Virtual Environment Setup Complete ✓

Your virtual environment has been created and all dependencies are installed.

### Quick Start

**To activate the virtual environment on Windows:**

```bash
venv\Scripts\activate.bat
```

**To deactivate the virtual environment:**

```bash
deactivate
```

### Running the Tests

**Option 1: Using the activated venv (in PowerShell)**
```bash
venv\Scripts\activate.bat
python tests/api_tests.py -v
```

**Option 2: Direct venv Python (no activation needed)**
```bash
venv\Scripts\python.exe tests/api_tests.py -v
```

### Installed Packages

The following packages are installed in your virtual environment:

- **spotipy** - Spotify Web API client
- **python-dotenv** - Environment variable management
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning library

### Environment Configuration

Before running tests, ensure your `.env` file contains:

```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

### Project Structure

```
Spotify-Curator/
├── venv/                           # Virtual environment (isolated Python)
├── requirements.txt                # Project dependencies
├── .env                            # Credentials (keep secret!)
├── Service_authentications/
│   └── auth_spotify.py            # Spotify authentication
├── src/
│   ├── features.py                # Feature engineering
│   └── ingest.py                  # Data ingestion
└── tests/
    └── api_tests.py               # Integration tests
```

### Updating Dependencies

If you add new packages later:

```bash
venv\Scripts\pip.exe install package_name
```

Then update requirements.txt:

```bash
venv\Scripts\pip.exe freeze > requirements.txt
```

### Troubleshooting

**Command not found: venv**
- Make sure you're in the `Spotify-Curator` directory
- Use `python -m venv venv` to recreate it

**ModuleNotFoundError when running tests**
- Ensure the virtual environment is activated
- Verify packages installed: `venv\Scripts\pip.exe list`

**Spotify authentication issues**
- Check `.env` file has correct credentials
- Ensure `REDIRECT_URI` is exactly: `http://127.0.0.1:8888/callback`
