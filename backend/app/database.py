from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Generator, Dict, List, Optional

from .config import DB_PATH, ensure_directories


def _connect() -> sqlite3.Connection:
    ensure_directories()
    connection = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def db() -> Generator[sqlite3.Connection, None, None]:
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_tasks (
                task_id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                negative_prompt TEXT DEFAULT '',
                width INTEGER DEFAULT 512,
                height INTEGER DEFAULT 768,
                num_inference_steps INTEGER DEFAULT 9,
                motion_type TEXT DEFAULT 'zoom_in',
                num_frames INTEGER DEFAULT 30,
                fps INTEGER DEFAULT 15,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0.0,
                detail TEXT DEFAULT '',
                image_filename TEXT DEFAULT NULL,
                video_filename TEXT DEFAULT NULL,
                error TEXT DEFAULT NULL,
                created_at REAL NOT NULL,
                completed_at REAL DEFAULT NULL,
                duration_seconds REAL DEFAULT NULL
            )
            """
        )


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return dict(row)


def create_task(
    task_id: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1376,
    height: int = 768,
    num_inference_steps: int = 9,
    motion_type: str = "zoom_in",
    num_frames: int = 30,
    fps: int = 15,
) -> Dict[str, Any]:
    now = time.time()
    with db() as conn:
        conn.execute(
            """
            INSERT INTO generation_tasks (
                task_id, prompt, negative_prompt, width, height,
                num_inference_steps, motion_type, num_frames, fps,
                status, progress, detail, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                prompt,
                negative_prompt,
                width,
                height,
                num_inference_steps,
                motion_type,
                num_frames,
                fps,
                "queued",
                0.0,
                "Tác vụ đang chờ xử lý trong hàng chờ...",
                now,
            ),
        )
    return get_task(task_id)  # type: ignore


def update_task_progress(
    task_id: str,
    status: str,
    progress: float,
    detail: str = "",
    image_filename: Optional[str] = None,
    video_filename: Optional[str] = None,
    error: Optional[str] = None,
    completed: bool = False,
) -> None:
    now = time.time()
    with db() as conn:
        row = conn.execute("SELECT created_at FROM generation_tasks WHERE task_id = ?", (task_id,)).fetchone()
        duration = (now - row["created_at"]) if (completed and row) else None

        fields = ["status = ?", "progress = ?", "detail = ?"]
        params = [status, progress, detail]

        if image_filename is not None:
            fields.append("image_filename = ?")
            params.append(image_filename)

        if video_filename is not None:
            fields.append("video_filename = ?")
            params.append(video_filename)

        if error is not None:
            fields.append("error = ?")
            params.append(error)

        if completed:
            fields.append("completed_at = ?")
            params.append(now)
            fields.append("duration_seconds = ?")
            params.append(duration)

        params.append(task_id)

        query = f"UPDATE generation_tasks SET {', '.join(fields)} WHERE task_id = ?"
        conn.execute(query, params)


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    with db() as conn:
        row = conn.execute("SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)).fetchone()
        return _row_to_dict(row)


def list_tasks(limit: int = 50) -> List[Dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM generation_tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows if r is not None]  # type: ignore


def delete_task(task_id: str) -> bool:
    with db() as conn:
        cursor = conn.execute("DELETE FROM generation_tasks WHERE task_id = ?", (task_id,))
        return cursor.rowcount > 0
