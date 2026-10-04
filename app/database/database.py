import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from app.config import AppConfig
from app.database.models import FaceRecord, EventRecord
from app.logging.event_logger import get_logger


class Database:
    """SQLite Persistence Engine for Registered Faces and Entry/Exit Events."""

    def __init__(self, config_or_path: AppConfig | Path | str):
        if isinstance(config_or_path, AppConfig):
            self.db_path = config_or_path.resolve_path(config_or_path.database.path)
        else:
            self.db_path = Path(config_or_path).resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns SQLite connection with row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Creates 'faces' and 'events' tables safely if they do not exist."""
        create_faces_table = """
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id TEXT UNIQUE NOT NULL,
            embedding BLOB NOT NULL,
            registered_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            first_crop_path TEXT NOT NULL
        );
        """

        create_events_table = """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            tracking_id INTEGER NOT NULL,
            confidence REAL NOT NULL,
            similarity REAL NOT NULL,
            FOREIGN KEY (face_id) REFERENCES faces(face_id)
        );
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_faces_table)
                cursor.execute(create_events_table)
                conn.commit()
            self.logger.info(f"[DATABASE] Database initialized successfully at: {self.db_path}")
        except Exception as e:
            self.logger.error(f"[DATABASE] Failed to initialize database: {e}")
            raise RuntimeError(f"Database initialization error: {e}") from e

    def add_face(
        self,
        face_id: str,
        embedding: np.ndarray,
        first_crop_path: str,
        registered_at: str,
        last_seen_at: str,
    ) -> bool:
        """Inserts a new persistent face record into database."""
        blob = embedding.astype(np.float32).tobytes()
        sql = """
        INSERT OR IGNORE INTO faces (face_id, embedding, registered_at, last_seen_at, first_crop_path)
        VALUES (?, ?, ?, ?, ?);
        """
        try:
            with self._get_connection() as conn:
                conn.cursor().execute(sql, (face_id, blob, registered_at, last_seen_at, first_crop_path))
                conn.commit()
            self.logger.info(f"[DATABASE] Added persistent identity record '{face_id}' to database.")
            return True
        except Exception as e:
            self.logger.error(f"[DATABASE] Failed to insert face record '{face_id}': {e}")
            return False

    def update_face_last_seen(self, face_id: str, last_seen_at: str) -> bool:
        """Updates last_seen_at timestamp for an existing face_id."""
        sql = "UPDATE faces SET last_seen_at = ? WHERE face_id = ?;"
        try:
            with self._get_connection() as conn:
                conn.cursor().execute(sql, (last_seen_at, face_id))
                conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"[DATABASE] Failed to update last_seen for '{face_id}': {e}")
            return False

    def get_face(self, face_id: str) -> Optional[FaceRecord]:
        """Retrieves FaceRecord by face_id."""
        sql = "SELECT id, face_id, embedding, registered_at, last_seen_at, first_crop_path FROM faces WHERE face_id = ?;"
        try:
            with self._get_connection() as conn:
                row = conn.cursor().execute(sql, (face_id,)).fetchone()
                if row:
                    emb = np.frombuffer(row["embedding"], dtype=np.float32)
                    return FaceRecord(
                        id=row["id"],
                        face_id=row["face_id"],
                        registered_at=row["registered_at"],
                        last_seen_at=row["last_seen_at"],
                        embedding=emb,
                        first_crop_path=row["first_crop_path"],
                    )
        except Exception as e:
            self.logger.error(f"[DATABASE] Error querying face '{face_id}': {e}")
        return None

    def get_all_faces(self) -> List[FaceRecord]:
        """Retrieves all registered FaceRecords from database."""
        sql = "SELECT id, face_id, embedding, registered_at, last_seen_at, first_crop_path FROM faces ORDER BY id ASC;"
        records = []
        try:
            with self._get_connection() as conn:
                rows = conn.cursor().execute(sql).fetchall()
                for row in rows:
                    emb = np.frombuffer(row["embedding"], dtype=np.float32)
                    records.append(
                        FaceRecord(
                            id=row["id"],
                            face_id=row["face_id"],
                            registered_at=row["registered_at"],
                            last_seen_at=row["last_seen_at"],
                            embedding=emb,
                            first_crop_path=row["first_crop_path"],
                        )
                    )
        except Exception as e:
            self.logger.error(f"[DATABASE] Error retrieving all faces: {e}")
        return records

    def add_event(
        self,
        face_id: str,
        event_type: str,
        timestamp: str,
        image_path: str,
        tracking_id: int,
        confidence: float = 0.0,
        similarity: float = 0.0,
    ) -> bool:
        """Inserts an event record (ENTRY, EXIT, REGISTRATION) into database."""
        sql = """
        INSERT INTO events (face_id, event_type, timestamp, image_path, tracking_id, confidence, similarity)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        try:
            with self._get_connection() as conn:
                conn.cursor().execute(sql, (face_id, event_type, timestamp, image_path, tracking_id, confidence, similarity))
                conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"[DATABASE] Failed to insert event for '{face_id}': {e}")
            return False

    def get_all_events(self) -> List[EventRecord]:
        """Retrieves all event records from database."""
        sql = "SELECT id, face_id, event_type, timestamp, image_path, tracking_id, confidence, similarity FROM events ORDER BY id ASC;"
        records = []
        try:
            with self._get_connection() as conn:
                rows = conn.cursor().execute(sql).fetchall()
                for row in rows:
                    records.append(
                        EventRecord(
                            id=row["id"],
                            face_id=row["face_id"],
                            event_type=row["event_type"],
                            timestamp=row["timestamp"],
                            image_path=row["image_path"],
                            tracking_id=row["tracking_id"],
                            confidence=row["confidence"],
                            similarity=row["similarity"],
                        )
                    )
        except Exception as e:
            self.logger.error(f"[DATABASE] Error querying events: {e}")
        return records

    def get_distinct_face_count(self) -> int:
        """Returns total unique registered face count from faces table."""
        sql = "SELECT COUNT(DISTINCT face_id) as total FROM faces;"
        try:
            with self._get_connection() as conn:
                row = conn.cursor().execute(sql).fetchone()
                return int(row["total"]) if row else 0
        except Exception as e:
            self.logger.error(f"[DATABASE] Error querying distinct face count: {e}")
            return 0

    def get_event_counts(self) -> Tuple[int, int]:
        """Returns (total_entries, total_exits)."""
        sql_entry = "SELECT COUNT(*) as cnt FROM events WHERE event_type = 'ENTRY';"
        sql_exit = "SELECT COUNT(*) as cnt FROM events WHERE event_type = 'EXIT';"
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                r_in = cur.execute(sql_entry).fetchone()
                r_out = cur.execute(sql_exit).fetchone()
                return (int(r_in["cnt"]) if r_in else 0, int(r_out["cnt"]) if r_out else 0)
        except Exception as e:
            self.logger.error(f"[DATABASE] Error querying event counts: {e}")
            return 0, 0
