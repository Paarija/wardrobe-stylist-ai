"""Small persistent store for the local portfolio app."""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import uuid
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FASHION_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "wardrobe.db"
UPLOAD_DIR = DATA_DIR / "uploads"
CATEGORIES = ("top", "bottom", "shoes", "dress", "outerwear")


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                salt BLOB NOT NULL,
                password_hash BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                color TEXT NOT NULL,
                style TEXT NOT NULL,
                occasion TEXT NOT NULL,
                image_path TEXT NOT NULL,
                available INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS items_user_idx ON items(user_id);
            CREATE TABLE IF NOT EXISTS color_feedback (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                item_id TEXT NOT NULL,
                label TEXT NOT NULL,
                lab_l REAL NOT NULL,
                lab_a REAL NOT NULL,
                lab_b REAL NOT NULL,
                corrected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, item_id)
            );
            CREATE TABLE IF NOT EXISTS outfit_ratings (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                study_id TEXT NOT NULL,
                occasion TEXT NOT NULL,
                outfit_key TEXT NOT NULL,
                model_rank INTEGER NOT NULL,
                model_score REAL NOT NULL,
                method TEXT NOT NULL DEFAULT 'rules',
                candidate_pool_size INTEGER NOT NULL DEFAULT 0,
                display_position INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                rated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, study_id, outfit_key)
            );
            """
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(items)")}
        if "reviewed" not in columns:
            connection.execute("ALTER TABLE items ADD COLUMN reviewed INTEGER NOT NULL DEFAULT 0")
        rating_columns = {row["name"] for row in connection.execute("PRAGMA table_info(outfit_ratings)")}
        if "method" not in rating_columns:
            connection.execute("ALTER TABLE outfit_ratings ADD COLUMN method TEXT NOT NULL DEFAULT 'rules'")
        if "candidate_pool_size" not in rating_columns:
            connection.execute("ALTER TABLE outfit_ratings ADD COLUMN candidate_pool_size INTEGER NOT NULL DEFAULT 0")


def register_user(username: str, password: str) -> int:
    username = username.strip().lower()
    if len(username) < 3 or len(password) < 8:
        raise ValueError("Use a username of at least 3 characters and a password of at least 8 characters.")
    salt = os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    try:
        with _connect() as connection:
            cursor = connection.execute(
                "INSERT INTO users(username, salt, password_hash) VALUES (?, ?, ?)",
                (username, salt, password_hash),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError("That username is already taken.") from exc


def authenticate(username: str, password: str) -> int | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT id, salt, password_hash FROM users WHERE username = ?",
            (username.strip().lower(),),
        ).fetchone()
    if row is None:
        return None
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), row["salt"], 310_000)
    return int(row["id"]) if hmac.compare_digest(candidate, row["password_hash"]) else None


def add_item(
    user_id: int,
    image: Image.Image,
    *,
    name: str,
    category: str,
    color: str,
    style: str,
    occasion: str,
) -> str:
    if category not in CATEGORIES:
        raise ValueError("Unknown clothing category.")
    if not name.strip():
        raise ValueError("Give the item a name.")
    item_id = str(uuid.uuid4())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{item_id}.jpg"
    prepared = image.convert("RGB")
    prepared.thumbnail((1200, 1200))
    prepared.save(path, format="JPEG", quality=85, optimize=True)
    try:
        with _connect() as connection:
            connection.execute(
                """INSERT INTO items(id, user_id, name, category, color, style, occasion, image_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (item_id, user_id, name.strip()[:80], category, color, style, occasion, str(path)),
            )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return item_id


def list_items(user_id: int) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM items WHERE user_id = ? ORDER BY created_at DESC, id DESC", (user_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def set_availability(user_id: int, item_id: str, available: bool) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE items SET available = ? WHERE id = ? AND user_id = ?",
            (int(available), item_id, user_id),
        )


def update_item(
    user_id: int,
    item_id: str,
    *,
    name: str,
    category: str,
    color: str,
    style: str,
    occasion: str,
) -> None:
    if category not in CATEGORIES:
        raise ValueError("Unknown clothing category.")
    if not name.strip():
        raise ValueError("Give the item a name.")
    with _connect() as connection:
        connection.execute(
            """UPDATE items SET name = ?, category = ?, color = ?, style = ?, occasion = ?, reviewed = 1
               WHERE id = ? AND user_id = ?""",
            (name.strip()[:80], category, color, style, occasion, item_id, user_id),
        )


def update_item_color(user_id: int, item_id: str, color: str) -> None:
    if not color.strip():
        raise ValueError("Detected colour cannot be empty.")
    with _connect() as connection:
        connection.execute(
            "UPDATE items SET color = ? WHERE id = ? AND user_id = ?",
            (color.strip().lower(), item_id, user_id),
        )


def save_color_feedback(
    user_id: int,
    item_id: str,
    label: str,
    signature: tuple[float, float, float],
) -> None:
    with _connect() as connection:
        connection.execute(
            """INSERT INTO color_feedback(user_id, item_id, label, lab_l, lab_a, lab_b)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, item_id) DO UPDATE SET
               label=excluded.label, lab_l=excluded.lab_l, lab_a=excluded.lab_a,
               lab_b=excluded.lab_b, corrected_at=CURRENT_TIMESTAMP""",
            (user_id, item_id, label, *signature),
        )


def list_color_feedback(user_id: int) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT item_id, label, lab_l, lab_a, lab_b FROM color_feedback WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_item(user_id: int, item_id: str) -> None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT image_path FROM items WHERE id = ? AND user_id = ?", (item_id, user_id)
        ).fetchone()
        if row is None:
            return
        connection.execute("DELETE FROM color_feedback WHERE user_id = ? AND item_id = ?", (user_id, item_id))
        connection.execute("DELETE FROM items WHERE id = ? AND user_id = ?", (item_id, user_id))
    Path(row["image_path"]).unlink(missing_ok=True)


def save_outfit_ratings(user_id: int, study_id: str, occasion: str, choices: list[dict]) -> None:
    if not study_id or not choices or any(choice["rating"] not in (1, 2, 3, 4, 5) for choice in choices):
        raise ValueError("Choose a rating from 1 to 5 for each outfit.")
    with _connect() as connection:
        connection.executemany(
            """INSERT INTO outfit_ratings
               (user_id, study_id, occasion, outfit_key, model_rank, model_score, method, candidate_pool_size, display_position, rating)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, study_id, outfit_key) DO UPDATE SET
               rating = excluded.rating, rated_at = CURRENT_TIMESTAMP""",
            [(user_id, study_id, occasion, choice["outfit_key"], choice["model_rank"],
              choice["model_score"], choice.get("method", "rules"), choice.get("candidate_pool_size", 0),
              choice["display_position"], choice["rating"]) for choice in choices],
        )


def list_outfit_ratings(user_id: int | None = None) -> list[dict]:
    with _connect() as connection:
        if user_id is None:
            rows = connection.execute("SELECT * FROM outfit_ratings ORDER BY user_id, study_id, display_position").fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM outfit_ratings WHERE user_id = ? ORDER BY study_id, display_position", (user_id,)
            ).fetchall()
    return [dict(row) for row in rows]
