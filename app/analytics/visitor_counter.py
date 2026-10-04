from typing import Any, Dict

from app.database.database import Database
from app.analytics.visit_manager import VisitManager
from app.logging.event_logger import get_logger


class VisitorCounter:
    """Unique Visitor and Presence Analytics Engine."""

    def __init__(self, database: Database, visit_manager: VisitManager):
        self.database = database
        self.visit_manager = visit_manager
        self.logger = get_logger()

    def get_unique_visitor_count(self) -> int:
        """Returns total distinct registered persistent face IDs count."""
        return self.database.get_distinct_face_count()

    def get_active_visitor_count(self) -> int:
        """Returns count of currently present active visitors."""
        return len(self.visit_manager.get_active_visitor_ids())

    def get_entry_exit_counts(self) -> tuple[int, int]:
        """Returns (total_entries, total_exits)."""
        return self.database.get_event_counts()

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured dictionary of analytics metrics."""
        entries, exits = self.get_entry_exit_counts()
        return {
            "unique_visitors": self.get_unique_visitor_count(),
            "active_visitors": self.get_active_visitor_count(),
            "total_entries": entries,
            "total_exits": exits,
        }
