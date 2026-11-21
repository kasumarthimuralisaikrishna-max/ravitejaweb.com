"""SQLite local persistence for RESUMATE AI analysis history.

Privacy note: Uploaded PDFs and full raw resume text are intentionally NOT
stored in SQLite. Only analysis metadata and structured AI feedback are persisted.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import DATA_DIR, DB_PATH
from modules.score_calculator import (
    ResumeAnalysisSchema,
    compute_score_breakdown,
    normalize_analysis_result,
    to_standard_json_dict,
)


@dataclass(frozen=True)
class AnalysisRecord:
    """Single stored analysis entry from SQLite history."""

    id: int
    created_at: str
    filename: str
    job_preview: str
    ats_score: int
    match_level: str
    keyword_coverage: float
    analysis: ResumeAnalysisSchema


@dataclass(frozen=True)
class DashboardStats:
    """Real aggregated metrics computed from stored analysis history."""

    total_analyzed: int
    average_score: float | None
    highest_score: int | None


def _get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize SQLite schema if it does not yet exist."""
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                job_preview TEXT NOT NULL,
                ats_score INTEGER NOT NULL,
                match_level TEXT NOT NULL,
                keyword_coverage REAL NOT NULL,
                analysis_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _make_job_preview(job_description: str, max_len: int = 90) -> str:
    """Extract a clean short preview from the first non-empty line of the job description."""
    for line in (job_description or "").splitlines():
        stripped = " ".join(line.split()).strip()
        if stripped:
            return stripped if len(stripped) <= max_len else stripped[: max_len - 1].rstrip() + "…"
    return "Target Role"


def save_analysis(
    filename: str,
    job_description: str,
    analysis: ResumeAnalysisSchema,
    db_path: Path = DB_PATH,
) -> int:
    """Persist an analysis result to SQLite (without storing the raw resume file/text)."""
    init_db(db_path)
    breakdown = compute_score_breakdown(analysis)
    created_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    job_preview = _make_job_preview(job_description)
    payload_json = json.dumps(to_standard_json_dict(analysis), ensure_ascii=False)

    with _get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO analysis_history (
                created_at, filename, job_preview, ats_score,
                match_level, keyword_coverage, analysis_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                Path(filename).name,
                job_preview,
                breakdown.ats_score,
                breakdown.match_level,
                breakdown.keyword_coverage_pct,
                payload_json,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def _row_to_record(row: sqlite3.Row) -> AnalysisRecord:
    raw: dict[str, Any] = json.loads(row["analysis_json"])
    analysis = normalize_analysis_result(raw)
    return AnalysisRecord(
        id=int(row["id"]),
        created_at=str(row["created_at"]),
        filename=str(row["filename"]),
        job_preview=str(row["job_preview"]),
        ats_score=int(row["ats_score"]),
        match_level=str(row["match_level"]),
        keyword_coverage=float(row["keyword_coverage"]),
        analysis=analysis,
    )


def get_all_history(limit: int = 100, db_path: Path = DB_PATH) -> list[AnalysisRecord]:
    """Retrieve stored analyses ordered from newest to oldest."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, filename, job_preview, ats_score,
                   match_level, keyword_coverage, analysis_json
            FROM analysis_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records: list[AnalysisRecord] = []
    for row in rows:
        try:
            records.append(_row_to_record(row))
        except Exception:
            continue
    return records


def get_analysis_by_id(record_id: int, db_path: Path = DB_PATH) -> AnalysisRecord | None:
    """Fetch a single analysis record by ID."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, created_at, filename, job_preview, ats_score,
                   match_level, keyword_coverage, analysis_json
            FROM analysis_history
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def get_dashboard_stats(db_path: Path = DB_PATH) -> DashboardStats:
    """Compute real aggregate dashboard KPIs from SQLite."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   AVG(ats_score) AS avg_score,
                   MAX(ats_score) AS max_score
            FROM analysis_history
            """
        ).fetchone()

    total = int(row["total"]) if row and row["total"] is not None else 0
    if total == 0:
        return DashboardStats(total_analyzed=0, average_score=None, highest_score=None)

    return DashboardStats(
        total_analyzed=total,
        average_score=round(float(row["avg_score"]), 1) if row["avg_score"] is not None else None,
        highest_score=int(row["max_score"]) if row["max_score"] is not None else None,
    )


def delete_analysis(record_id: int, db_path: Path = DB_PATH) -> None:
    """Delete a specific analysis history item."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
        conn.commit()


def clear_history(db_path: Path = DB_PATH) -> None:
    """Delete all stored analysis history."""
    init_db(db_path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM analysis_history")
        conn.commit()
