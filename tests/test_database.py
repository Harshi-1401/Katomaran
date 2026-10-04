import numpy as np
import pytest
from pathlib import Path

from app.database.database import Database
from app.database.models import FaceRecord, EventRecord


def test_database_persistence_and_queries(tmp_path):
    db_file = tmp_path / "test_faces.db"
    db = Database(db_file)

    # 1. Add Face RECORD
    emb1 = np.ones(512, dtype=np.float32)
    success = db.add_face("FACE_0001", emb1, "logs/entries/crop1.jpg", "2026-10-03 10:00:00", "2026-10-03 10:00:00")
    assert success is True

    # Query face
    rec = db.get_face("FACE_0001")
    assert rec is not None
    assert rec.face_id == "FACE_0001"
    assert rec.embedding.shape == (512,)
    assert np.allclose(rec.embedding, emb1)

    # 2. Add Event RECORD
    ev_ok = db.add_event("FACE_0001", "ENTRY", "2026-10-03 10:00:00", "logs/entries/crop1.jpg", 17, 0.95, 1.0)
    assert ev_ok is True

    events = db.get_all_events()
    assert len(events) == 1
    assert events[0].face_id == "FACE_0001"
    assert events[0].event_type == "ENTRY"

    # 3. Add second face
    emb2 = np.zeros(512, dtype=np.float32)
    db.add_face("FACE_0002", emb2, "logs/entries/crop2.jpg", "2026-10-03 10:05:00", "2026-10-03 10:05:00")

    assert db.get_distinct_face_count() == 2

    # 4. Connection restart test
    db2 = Database(db_file)
    assert db2.get_distinct_face_count() == 2
    rec2 = db2.get_face("FACE_0001")
    assert rec2 is not None
    assert np.allclose(rec2.embedding, emb1)
