from datetime import datetime
import numpy as np
import pytest

from app.config import AppConfig, PROJECT_ROOT
from app.analytics.visit_manager import VisitManager, VisitEvent


def test_visit_manager_entry_exit_lifecycle():
    cfg = AppConfig.load_from_json(PROJECT_ROOT / "config.json")
    vm = VisitManager(cfg)

    crop = np.full((50, 50, 3), 128, dtype=np.uint8)
    dt1 = datetime(2026, 10, 3, 10, 0, 0)

    # 1. First presence -> ENTRY event
    ev_entry1 = vm.process_presence("FACE_0001", track_id=1, face_crop=crop, confidence=0.9, dt=dt1)
    assert ev_entry1 is not None
    assert ev_entry1.event_type == "ENTRY"
    assert ev_entry1.face_id == "FACE_0001"
    assert ev_entry1.track_id == 1

    # 2. Repeated presence in same visit -> NO duplicate ENTRY event
    dt2 = datetime(2026, 10, 3, 10, 0, 5)
    ev_dup = vm.process_presence("FACE_0001", track_id=1, face_crop=crop, confidence=0.92, dt=dt2)
    assert ev_dup is None  # Exactly one ENTRY!

    # 3. Absence -> EXIT event
    dt3 = datetime(2026, 10, 3, 10, 2, 0)
    ev_exit1 = vm.process_absence("FACE_0001", track_id=1, dt=dt3)
    assert ev_exit1 is not None
    assert ev_exit1.event_type == "EXIT"
    assert ev_exit1.face_id == "FACE_0001"

    # 4. Returning visit -> New ENTRY event under SAME face_id
    dt4 = datetime(2026, 10, 3, 10, 10, 0)
    ev_entry2 = vm.process_presence("FACE_0001", track_id=2, face_crop=crop, confidence=0.88, dt=dt4)
    assert ev_entry2 is not None
    assert ev_entry2.event_type == "ENTRY"
    assert ev_entry2.face_id == "FACE_0001"
    assert ev_entry2.track_id == 2

    # Verify event history: 2 ENTRYs, 1 EXIT
    history = vm.get_event_history()
    assert len(history) == 3
    types = [e.event_type for e in history]
    assert types == ["ENTRY", "EXIT", "ENTRY"]
