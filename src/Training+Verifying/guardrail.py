"""
guardrail.py
============
Stage 5: LLM Self-Critique / Guardrail

Before any track is written to a Spotify playlist, this module sends
the proposed additions to an LLM for review. The LLM acts as a music
taste critic — it reads the cluster's vibe profile, the proposed
tracks, and their confidence scores, then flags any mismatches.

This is the agentic self-check loop described in the architecture:
  score() → confidence check → LLM review → write or reject

Why this matters:
  Cosine similarity catches feature-level similarity well, but can
  miss things like: a track being a live recording, an interlude,
  a genre tag being misleading, or a song having the right energy
  but completely wrong cultural context for the playlist vibe.
  The LLM catches these edge cases that pure math misses.

Requirements:
  pip install boto3
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION must be set in .env
  Amazon Nova Pro requires no marketplace subscription — available by default
  on any Bedrock-enabled AWS account.
  Preprocessing uses Nova Lite; this uses Nova Pro to keep the two roles distinct.
"""

import os
import json
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv

from taste_model import ClusterProfile
from scorer import ScoredTrack

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailVerdict:
    """The LLM's verdict on a single candidate track."""
    track_id:   str
    title:      str
    artist:     str
    approved:   bool
    reason:     str
    confidence: float   # original cosine confidence score


@dataclass
class GuardrailReport:
    """Full review report for one cluster's proposed playlist additions."""
    cluster_id:     int
    cluster_label:  str
    approved:       List[GuardrailVerdict]
    rejected:       List[GuardrailVerdict]
    llm_summary:    str   # Claude's overall assessment of the batch
    raw_response:   str   # full LLM response for logging


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_review_prompt(
    profile: ClusterProfile,
    candidates: List[ScoredTrack],
) -> str:
    """
    Builds the prompt sent to Claude.

    We give Claude:
    1. The cluster's vibe profile (audio averages + top genres)
    2. Representative tracks already in the cluster (for tone/context)
    3. The proposed new tracks with their confidence scores
    4. Clear JSON output instructions

    The JSON output format is strict so we can parse it reliably.
    """
    # format representative tracks as a readable list
    rep_tracks_str = "\n".join([
        f"  - {t['title']} by {t['artist']}"
        for t in profile.representative_tracks[:5]
    ]) or "  (none available)"

    # format candidates as a structured list
    candidates_str = "\n".join([
        f"  [{i+1}] \"{c.title}\" by {c.artist}\n"
        f"       confidence={c.confidence:.2f} | "
        f"energy={c.audio_features.get('energy', 'N/A')} | "
        f"valence={c.audio_features.get('valence', 'N/A')} | "
        f"danceability={c.audio_features.get('danceability', 'N/A')} | "
        f"acousticness={c.audio_features.get('acousticness', 'N/A')}\n"
        f"       genres: {', '.join(c.audio_features.get('genres', [])[:4]) or 'unknown'}\n"
        f"       score reason: {c.explanation}"
        for i, c in enumerate(candidates)
    ])

    return f"""You are a music curator reviewing proposed additions to a Spotify playlist.

PLAYLIST VIBE PROFILE
=====================
Label:        {profile.label}
Cluster size: {profile.size} tracks

Audio fingerprint:
  Energy:       {profile.avg_energy:.2f}  (0=calm, 1=intense)
  Valence:      {profile.avg_valence:.2f}  (0=sad/dark, 1=happy/bright)
  Danceability: {profile.avg_danceability:.2f}  (0=not danceable, 1=highly danceable)
  Acousticness: {profile.avg_acousticness:.2f}  (0=electronic, 1=acoustic)
  Avg tempo:    {profile.avg_tempo:.0f} BPM

Top genre tags in this cluster:
  {', '.join(profile.top_genres) or 'unknown'}

Representative tracks already in this cluster:
{rep_tracks_str}

PROPOSED NEW TRACKS TO REVIEW
==============================
{candidates_str}

YOUR TASK
=========
Review each proposed track and decide: APPROVE or REJECT.

APPROVE if the track genuinely fits the playlist vibe based on:
- Audio features close to the cluster profile
- Genre and cultural context consistent with existing tracks
- Would feel natural to a listener of this playlist

REJECT if:
- Audio features are misleading (e.g. a live recording skewing acousticness)
- Genre tag overlap is superficial (e.g. tagged "pop" but clearly wrong subgenre)
- The track would feel jarring or out of place in this playlist
- It's an interlude, skit, comedy track, or non-music content

Respond ONLY with valid JSON in exactly this format (no preamble, no markdown):
{{
  "summary": "2-3 sentence overall assessment of this batch",
  "verdicts": [
    {{
      "title": "track title",
      "artist": "artist name",
      "approved": true,
      "reason": "one sentence reason"
    }}
  ]
}}

The "verdicts" array must have exactly {len(candidates)} entries, one per proposed track, in the same order as listed above.
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM call — Amazon Nova Pro via Bedrock
# ─────────────────────────────────────────────────────────────────────────────

# Nova Pro: no marketplace subscription required, distinct from the Nova Lite
# model used in llm_features.py during preprocessing — higher reasoning
# capability suits the curator/critic role here.
# Override with env var GUARDRAIL_MODEL_ID to use a different model.
_GUARDRAIL_MODEL_ID = os.getenv("GUARDRAIL_MODEL_ID", "amazon.nova-pro-v1:0")
_GUARDRAIL_SYSTEM = (
    "You are a music curator and taste critic. You review proposed new tracks "
    "for a Spotify playlist and decide whether each one genuinely fits the "
    "playlist's vibe. Be discerning: flag live recordings, interludes, tracks "
    "with misleading genre tags, and anything that would feel jarring. "
    "Respond ONLY with valid JSON — no preamble, no markdown fences."
)


def _call_nova_guardrail(prompt: str) -> str:
    """
    Calls Amazon Nova Pro via Bedrock for the guardrail review.
    Returns the raw response text.
    Raises on auth/API error so the agent can handle it gracefully.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 not installed. Run: pip install boto3")

    region = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1"))
    client = boto3.client("bedrock-runtime", region_name=region)

    body = json.dumps({
        "system": [{"text": _GUARDRAIL_SYSTEM}],
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.1},
    })

    response = client.invoke_model(
        modelId=_GUARDRAIL_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]


def _parse_llm_response(
    raw_response: str,
    candidates: List[ScoredTrack],
) -> Tuple[List[GuardrailVerdict], List[GuardrailVerdict], str]:
    """
    Parses Claude's JSON response into approved/rejected verdict lists.

    Falls back to approving all candidates if the response can't be parsed,
    with a warning logged — better to be permissive than silently discard
    everything due to a parsing error.
    """
    try:
        # strip any accidental markdown fences
        clean = raw_response.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])

        data = json.loads(clean)

        summary  = data.get("summary", "")
        verdicts = data.get("verdicts", [])

        if len(verdicts) != len(candidates):
            logger.warning(
                f"LLM returned {len(verdicts)} verdicts for {len(candidates)} candidates — "
                "padding with approvals for unreviewed tracks"
            )

        approved = []
        rejected = []

        for i, candidate in enumerate(candidates):
            if i < len(verdicts):
                v = verdicts[i]
                verdict = GuardrailVerdict(
                    track_id=  candidate.track_id,
                    title=     candidate.title,
                    artist=    candidate.artist,
                    approved=  bool(v.get("approved", True)),
                    reason=    v.get("reason", ""),
                    confidence=candidate.confidence,
                )
            else:
                # LLM didn't review this track — approve by default
                verdict = GuardrailVerdict(
                    track_id=  candidate.track_id,
                    title=     candidate.title,
                    artist=    candidate.artist,
                    approved=  True,
                    reason=    "Not reviewed by LLM — approved by default",
                    confidence=candidate.confidence,
                )

            if verdict.approved:
                approved.append(verdict)
            else:
                rejected.append(verdict)

        return approved, rejected, summary

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse LLM response: {e}\nRaw: {raw_response[:500]}")
        # fallback: approve everything rather than silently reject
        fallback_approved = [
            GuardrailVerdict(
                track_id=c.track_id, title=c.title, artist=c.artist,
                approved=True,
                reason="Approved by fallback (LLM response unparseable)",
                confidence=c.confidence,
            )
            for c in candidates
        ]
        return fallback_approved, [], "LLM response could not be parsed — all candidates approved by fallback."


# ─────────────────────────────────────────────────────────────────────────────
# Main guardrail runner
# ─────────────────────────────────────────────────────────────────────────────

def review_cluster_candidates(
    profile: ClusterProfile,
    candidates: List[ScoredTrack],
    max_candidates_per_review: int = 20,
) -> GuardrailReport:
    """
    Runs the LLM guardrail review for a single cluster's candidates.

    Only tracks that already passed the cosine similarity threshold
    are sent here. The LLM is the second gate, not the first.

    Parameters
    ----------
    profile                   : ClusterProfile for context
    candidates                : tracks that passed cosine threshold
    max_candidates_per_review : cap per LLM call to keep prompts manageable.
                                If more candidates exist, we take the top N
                                by confidence score.
    """
    if not candidates:
        logger.info(f"Cluster {profile.cluster_id}: no candidates to review")
        return GuardrailReport(
            cluster_id=    profile.cluster_id,
            cluster_label= profile.label,
            approved=[],
            rejected=[],
            llm_summary=   "No candidates passed the confidence threshold.",
            raw_response=  "",
        )

    # take top N by confidence — no point reviewing low-confidence stragglers
    to_review = sorted(candidates, key=lambda c: c.confidence, reverse=True)
    to_review = to_review[:max_candidates_per_review]

    logger.info(
        f"Cluster {profile.cluster_id} '{profile.label}': "
        f"sending {len(to_review)} candidates to LLM for review"
    )

    prompt       = _build_review_prompt(profile, to_review)
    raw_response = _call_nova_guardrail(prompt)

    logger.debug(f"LLM raw response for cluster {profile.cluster_id}:\n{raw_response}")

    approved, rejected, summary = _parse_llm_response(raw_response, to_review)

    report = GuardrailReport(
        cluster_id=    profile.cluster_id,
        cluster_label= profile.label,
        approved=      approved,
        rejected=      rejected,
        llm_summary=   summary,
        raw_response=  raw_response,
    )

    logger.info(
        f"Cluster {profile.cluster_id} guardrail result: "
        f"{len(approved)} approved / {len(rejected)} rejected\n"
        f"LLM summary: {summary}"
    )

    return report


def run_guardrail(
    profiles: List[ClusterProfile],
    scored_by_cluster: Dict[int, List[ScoredTrack]],
) -> Dict[int, GuardrailReport]:
    """
    Runs the LLM guardrail for every cluster.

    Parameters
    ----------
    profiles           : list of ClusterProfile from taste_model.py
    scored_by_cluster  : {cluster_id: [ScoredTrack]} from scorer.py
                         Only tracks with passed_threshold=True are reviewed.

    Returns
    -------
    {cluster_id: GuardrailReport}
    """
    logger.info("=== Starting LLM guardrail review ===")

    profile_map = {p.cluster_id: p for p in profiles}
    reports: Dict[int, GuardrailReport] = {}

    for cluster_id, scored_tracks in scored_by_cluster.items():
        profile = profile_map.get(cluster_id)
        if not profile:
            logger.warning(f"No profile for cluster {cluster_id} — skipping guardrail")
            continue

        # only pass tracks that cleared the cosine threshold
        passed = [t for t in scored_tracks if t.passed_threshold]

        report = review_cluster_candidates(profile, passed)
        reports[cluster_id] = report

    # print a human-readable summary to console
    print("\n" + "=" * 60)
    print("GUARDRAIL REVIEW SUMMARY")
    print("=" * 60)
    for cluster_id, report in reports.items():
        print(f"\nCluster {cluster_id}: '{report.cluster_label}'")
        print(f"  Approved : {len(report.approved)}")
        print(f"  Rejected : {len(report.rejected)}")
        print(f"  Assessment: {report.llm_summary}")
        if report.rejected:
            print("  Rejected tracks:")
            for v in report.rejected:
                print(f"    ✗ {v.title} by {v.artist} — {v.reason}")
    print("=" * 60)

    logger.info("=== LLM guardrail review complete ===")
    return reports
