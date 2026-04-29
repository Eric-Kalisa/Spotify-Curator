# Spotify Curator

## Summary

**Spotify Curator** is an agentic ML pipeline that learns from your music taste on your Spotify listening history, forms playlists and automatically discovers new tracks you are likely to enjoy — without you having to search for them.

The system clusters your listening history into natural "vibe buckets" using unsupervised learning, builds a mathematical profile of each cluster, then uses those profiles to score and vet candidate tracks before writing them into a Spotify playlist.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SPOTIFY CURATOR PIPELINE                     │
└─────────────────────────────────────────────────────────────────────┘

  INPUT                    AGENT STAGES                    OUTPUT
  ─────                    ────────────                    ──────

  Spotify API   ──►  ┌──────────────────┐
  (recently            │  Stage 1: INGEST │  ──►  raw track records
   played,             │  (ingest.py)     │        + audio features
   liked,              └──────────────────┘        + genres
   top tracks)                │
                              │  missing audio features/genres?
                              ▼
                    ┌──────────────────────┐
                    │  LLM FEATURE         │  ◄──  Amazon Nova Lite
                    │  ESTIMATOR           │        (Bedrock)
                    │  (llm_features.py)   │
                    └──────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  Stage 2: FEATURE    │  ──►  NumPy matrix X
                    │  ENGINEERING         │        (n_tracks × n_features)
                    │  (features.py)       │        - 10 normalized audio dims
                    └──────────────────────┘        - recency/source weights
                              │                     - cyclical time encoding
                              │                     - multi-hot genre encoding
                              ▼
                    ┌──────────────────────┐
                    │  Stage 3: SEGMENT    │  ──►  cluster labels per track
                    │  UMAP + HDBSCAN      │        (auto-determined K)
                    │  (segmenter.py)      │
                    └──────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  Stage 4: TASTE      │  ──►  ClusterProfile objects
                    │  MODEL               │        (centroid, vibe label,
                    │  (taste_model.py)    │         representative tracks)
                    └──────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  Stage 5: DISCOVERY  │  ──►  candidate tracks
                    │  Spotify related     │        (per cluster)
                    │  artists + Deezer    │
                    │  radio + new         │
                    │  releases            │
                    │  (discoverer.py)     │
                    └──────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  Stage 6: SCORER     │  ──►  confidence score 0–1
                    │  cosine similarity   │        per candidate track
                    │  vs cluster centroid │
                    │  (scorer.py)         │
                    └──────────────────────┘
                              │
                              │  threshold filter (≥ 50%)
                              ▼
                    ┌──────────────────────┐        ┌──────────────────┐
                    │  Stage 7: GUARDRAIL  │  ──►   │  HUMAN REVIEW    │
                    │  LLM self-critique   │  ◄──   │  (optional)      │
                    │  (guardrail.py)      │        │  inspect approved│
                    │                      │        │  / rejected lists│
                    │  Amazon Nova Pro     │        └──────────────────┘
                    │  (Bedrock)           │
                    └──────────────────────┘
                              │
                              │  approved tracks
                              ▼
                    ┌──────────────────────┐
                    │  Stage 8: PLAYLIST   │  ──►  Spotify playlist
                    │  WRITER              │        (written via API)
                    │  (playlist_writer.py)│
                    └──────────────────────┘

  ── TEST LAYER ───────────────────────────────────────────────────────
  tests/training_tests.py  runs 28 unit tests covering:
    • Feature matrix column contracts   • Cosine similarity math
    • UMAP + HDBSCAN output shapes      • Cluster profile weighting
    • LLM guardrail JSON parsing        • scorer ↔ features alignment
```

**Where humans are involved:**
- The `main.py` demo runner lets you inspect every stage's output in the terminal before anything is written to Spotify.
- The guardrail stage surfaces LLM approve/reject verdicts with reasons, which can be reviewed before the playlist writer runs.
- Unit tests encode human-defined invariants (e.g. `AUDIO_FEATURE_COLS` in scorer must exactly match `build_feature_matrix` columns) that catch silent regressions automatically; or data writing tests to catch when data is out of bounds,empty or unexpected value.

---

## Setup Instructions

### Prerequisites

- Python 3.10+ (tested on 3.14)
- A Spotify Developer account with a registered app
- AWS credentials with Bedrock access (for LLM stages)

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd Spotify-Curator
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure credentials

Create a `.env` file in the project root (it is already gitignored):

```env
# Spotify
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback

# AWS Bedrock (for LLM feature estimation + guardrail)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
```

### 4. Run the demo (no Spotify writes)

```bash
# Full run — fetches live Spotify data
python src/main.py

# Skip API calls, reuse cached data, after first run of main.py has collected songs and run llm estimations
python src/main.py --skip-ingest

# Skip the Bedrock guardrail call too if desired if not done training data
python src/main.py --skip-ingest --skip-guardrail
```

### 5. Run the full agent (writes playlists)

```bash
cd src/Training+Verifying
python agent.py
python agent.py --skip-ingest        # reuse cache
python agent.py --min-cluster-size 5 # smaller clusters
```

### 6. Run the test suite

```bash
cd tests
python -m pytest training_tests.py -v
```

---

## Sample Interactions

### Example 1 — Cluster Discovery

After ingesting 126 tracks the segmenter found 4 natural vibe clusters:

```
┌──  Cluster 3:  "Mid Energy · Upbeat · Electronic / Produced"
│    32 tracks  |  avg signal weight: 0.60
│
│    Energy       █████████░░░░░░░  0.59
│    Valence      ███████████░░░░░  0.69
│    Danceability ████████░░░░░░░░  0.50
│    Acousticness █░░░░░░░░░░░░░░░  0.09
│    Avg tempo    93 BPM
│    Genres       comedy-sketch, comedy, soukous
│
│    Representative tracks (closest to centroid):
│      • Mexicans vs. Puerto Ricans   — Anjelah Johnson-Reyes  dist=0.099
│      • Lessons from Marriage        — Anjelah Johnson-Reyes  dist=0.106
│      • Rosetta Stone                — Anjelah Johnson-Reyes  dist=0.108
└
```

The system correctly grouped a stand-up comedy cluster — it identified the genre pattern without any manual labeling.


---

### Example 2 — LLM Audio Feature Estimation

For tracks where Spotify's `audio_features` endpoint is unavailable (deprecated Nov 2024-- required business acc for access), Amazon Nova Lite estimates the features:

**Input:** `"Miracle" by Rachel Anyeme`

**LLM Output:**
```
Energy       ███████████░░░░░  0.70
Valence      █████████████░░░  0.80
Danceability ██████████░░░░░░  0.60
Acousticness ██░░░░░░░░░░░░░░  0.10
Tempo        120 BPM     Key: F major   Time sig: 4/4
Genres       worship, christian-gospel, praise-and-worship
```

The estimate is plausible for a gospel worship track: high valence, high energy, low acousticness (produced sound), 4/4 at 120 BPM.

---

### Example 3 — Scoring + Guardrail

Candidates discovered for Cluster 1 ("gospel/worship") were scored by cosine similarity against the cluster centroid:

```
Cluster 1: "Mid Energy · Neutral Mood · Electronic / Produced"
  Sources  deezer radio: 1
  Scored   1 candidates  |  1 passed threshold (≥50%)

       Conf  Title             Artist
  ────────────────────────────────────────────────────────
    ✓  ███████░ 86%  Warumagaye   Victor Rukotana
```

`Warumagaye` by Victor Rukotana scored 86% — already in the user's library — confirming that the scoring model correctly identifies familiar style.

---

## Design Decisions

### UMAP + HDBSCAN instead of K-Means

K-Means requires you to specify K in advance and assumes spherical clusters. Music taste is neither fixed-K nor spherical — someone might listen to gospel, comedy, and Afrobeats in naturally different-sized pockets. HDBSCAN discovers the number of clusters from the data and marks genuine outliers as noise instead of forcing them into the nearest bucket.

**Trade-off:** UMAP + HDBSCAN is slower and less deterministic than K-Means. Runs with a fixed `random_state=42` for reproducibility.

### Cosine Similarity for Scoring

Euclidean distance penalizes high-magnitude feature vectors (louder, more energetic tracks score worse just for being loud). Cosine similarity measures directional alignment — a track that has the same *shape* of features as the cluster centroid scores well regardless of raw magnitude.

**Trade-off:** Cosine similarity ignores magnitude entirely, so two tracks can score identically even if one is much more intense than the cluster average.

### LLM as Guardrail, Not Primary Scorer

The LLM is expensive per-call and non-deterministic. Using it as a second pass (after cosine similarity already filtered 50%+ of candidates) keeps Bedrock costs low while still catching edge cases a pure math model misses — live recordings, comedy interludes mislabeled as music, tracks that score well numerically but are contextually wrong.

**Trade-off:** The LLM can hallucinate or give inconsistent verdicts. The guardrail is a soft filter, not a hard gate — the playlist writer can be configured to ignore low-confidence rejections.

### Sklearn HDBSCAN over the `hdbscan` Package

The `hdbscan` PyPI package requires C++ Build Tools and has no pre-built wheel for Python 3.12+. `sklearn.cluster.HDBSCAN` (available since scikit-learn 1.3) provides equivalent clustering without a C compiler dependency.

**Trade-off:** `sklearn.cluster.HDBSCAN` does not support soft cluster membership probabilities via `all_points_membership_vectors()`. We fall back to `clusterer.probabilities_` (a 1D array of point-level confidence scores).

### Source-Weighted Feature Engineering

Not all listening events are equal signals. A liked song is a deliberate, durable preference; a recently-played track might be ambient background noise. Weights (`liked=1.0`, `top_short_term=0.9`, `recently_played=0.6`) are applied at the feature matrix level so clustering naturally emphasizes your strongest preferences.

---

**What worked well:**

- Spotify API correctly fetched the data as instructed. 
- LLM calls run smoothly both at features prediction before training and at grounding calls 
- UMAP dimension reductionality and noise reduction; assessing the clustering performance given that I used my own Spotify profile, I found that performance was decent

**What didn't work / required fixes:**

- discoverer is not fed good data. It almost uses none of the audio features collected with the LLM estimator to find new tracks since deezer radio is not audio feature enriched. The song finds are poorly related to the current clusters, a huge chunk does not pass the cosine similarity threshold and none pass the guardrail test 

**What I learned:**

- Industry level production requires extensive planning. If this was an app to be pushed to production, here are the considerations I'd think through: budget needs(token usage on LLM calls), threading process to cut wait times (considering Semaphore for Spotify API call paralled with LLM audio estimation), request API access, call 2-3 other models on a small dataset to test performance, a deliberate design pattern to protect src code from corruption and client data

- step-by-step, well thought out tests were implemented and were helpful to track what worked before moving on to next steps. This saved time and avoided frustration.


## Reflection

Building Spotify Curator taught me that AI systems fail silently more than they fail loudly. A missing feature column doesn't crash the pipeline — it just makes every score slightly wrong in a way that's hard to notice without a test that explicitly checks the contract between two modules.

The most important design insight was treating the LLM as a *last-mile critic* rather than a primary decision-maker. Cosine similarity over learned centroids does the heavy lifting cheaply and deterministically. The LLM's role is to catch the edge cases that math can't — contextual mismatches, metadata quirks, the difference between a studio recording and its live version. This division keeps costs controlled while still leveraging what LLMs are genuinely good at: reading context.

The project also illustrated the gap between "model works on toy data" and "pipeline works end-to-end." Every stage interface (what columns exist, what shapes arrays have, what encoding list-typed fields use across serialization boundaries) is a potential failure point. Writing tests that encode those interface contracts — not just unit behavior — is what makes a multi-stage AI pipeline actually maintainable.

On the same point, building a one user E2E program showed me the need for access. There are are many things I had to tradeoff because I had no access. For example: the intent was to use Spotify's artist and audio metadata for training but due to policy I had no access. I also intended to try categorically different models for training and testing but most required form submission and approval processing times. System design, planning is necessary for successful execution. 


# Future Improvements

1. Gaining access to Spotify API on audio and artist metadata to enrich dataset if possible
2. Fill out Anthropic use form to use claude's models
3. Deploy an agent that forms playlists and routinely adds music based on new realeases and user behavior (reinforcement learning)
4. Capture data that can help to improve model performance, for example incentivize when the user adds an agent added song to liked songs