from typing import Dict, List, Optional, Tuple
import numpy as np

from app.config import AppConfig
from app.recognition.face_recognizer import FaceRecognizer
from app.logging.event_logger import get_logger


class FaceMatcher:
    """In-memory ArcFace vector matching and persistent identity index."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger()
        self.similarity_threshold = config.recognition.similarity_threshold

        # Registered face identity index: face_id -> List[np.ndarray] (prototype embeddings)
        self._registry: Dict[str, List[np.ndarray]] = {}

        # Temporary session track mapping: track_id -> face_id
        self._track_map: Dict[int, str] = {}

        # Counter for generating FACE_XXXX IDs
        self._next_face_counter: int = 1

    def generate_next_face_id(self) -> str:
        """Generates formatted persistent face ID (e.g. FACE_0001)."""
        face_id = f"FACE_{self._next_face_counter:04d}"
        return face_id

    def match_embedding(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Compares target embedding against all registered identities using Cosine Similarity.

        Args:
            embedding (np.ndarray): 512-d L2-normalized float32 vector.

        Returns:
            Tuple[Optional[str], float]:
                - matched_face_id (str | None): Matched persistent face_id or None if below threshold.
                - max_similarity (float): Highest cosine similarity score observed.
        """
        if embedding is None or len(self._registry) == 0:
            return None, 0.0

        best_face_id: Optional[str] = None
        max_sim: float = -1.0

        for face_id, registered_embeddings in self._registry.items():
            for reg_emb in registered_embeddings:
                sim = FaceRecognizer.cosine_similarity(embedding, reg_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_face_id = face_id

        if max_sim >= self.similarity_threshold and best_face_id is not None:
            return best_face_id, max_sim
        else:
            return None, max_sim if max_sim > -1.0 else 0.0

    def register_identity(self, face_id: str, embedding: np.ndarray) -> None:
        """Registers a new or updated face identity in the in-memory vector index."""
        if face_id not in self._registry:
            self._registry[face_id] = []
            # Extract number if matching format FACE_XXXX to keep counter in sync
            if face_id.startswith("FACE_"):
                try:
                    num = int(face_id.replace("FACE_", ""))
                    if num >= self._next_face_counter:
                        self._next_face_counter = num + 1
                except ValueError:
                    pass

        # Maintain up to 3 representative prototype embeddings per face_id to avoid unbounded growth
        if len(self._registry[face_id]) < 3:
            self._registry[face_id].append(embedding.copy())

        self.logger.info(f"[REGISTRATION] Registered identity '{face_id}' into memory vector index (Total Registered: {len(self._registry)})")

    def bind_track(self, track_id: int, face_id: str) -> None:
        """Binds a temporary track_id to a persistent face_id."""
        self._track_map[track_id] = face_id

    def get_face_for_track(self, track_id: int) -> Optional[str]:
        """Retrieves persistent face_id bound to a temporary track_id."""
        return self._track_map.get(track_id)

    def unbind_track(self, track_id: int) -> None:
        """Unbinds temporary track_id without deleting persistent face identity."""
        if track_id in self._track_map:
            del self._track_map[track_id]

    def get_all_registered_ids(self) -> List[str]:
        """Returns list of all registered persistent face IDs."""
        return list(self._registry.keys())

    def get_embedding(self, face_id: str) -> Optional[np.ndarray]:
        """Retrieves primary prototype embedding for a registered face_id."""
        embeddings = self._registry.get(face_id)
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        return None

    def clear(self) -> None:
        """Clears memory index and reset counter."""
        self._registry.clear()
        self._track_map.clear()
        self._next_face_counter = 1
