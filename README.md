# Intelligent Face Tracker with AutoRegistration and Visitor Counting

An end-to-end AI-powered computer vision system designed for intelligent face detection, temporal face tracking, deep face embedding extraction, persistent identity auto-registration, re-identification, visit lifecycle management (exactly-once ENTRY/EXIT events), and unique visitor counting.

---

## 1. Problem Statement

Traditional video analytics and CCTV systems rely on temporary tracking IDs or raw bounding box detection counts. This leads to duplicate counts whenever a person briefly steps out of frame, gets occluded, or turns away. 

**Goal:** Build an intelligent computer vision solution that:
1. Detects faces using YOLOv8.
2. Tracks temporary appearances across frames using a custom IoU + Hungarian assignment tracker.
3. Extracts 512-dimensional facial feature embeddings using ArcFace (InsightFace MobileFaceNet ONNX).
4. Auto-registers new visitors with a persistent `FACE_ID` (e.g., `FACE_0001`).
5. Re-identifies returning visitors using cosine similarity matching against stored embeddings.
6. Enforces **exactly one ENTRY event** and **exactly one EXIT event** per physical visit.
7. Saves timestamped cropped face images for every ENTRY and EXIT event.
8. Stores all identity records and event logs in SQLite and a structured `events.log` file.
9. Accurately calculates **Unique Visitor Count** based on distinct persistent `FACE_ID`s rather than temporary track IDs.

---

## 2. Solution Overview

The application features a modular processing pipeline that ingests frames from local video files or RTSP network camera streams. It tracks faces frame-by-frame, extracts facial embeddings, matches faces against an in-memory index loaded from a persistent SQLite database, logs stateful ENTRY and EXIT events, and saves cropped evidence images.

---

## 3. AI Pipeline Architecture

```text
               +----------------------------------+
               |        VIDEO / RTSP STREAM       |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |       YOLO FACE DETECTOR         |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |      FACE TRACKER (IoU+Hungarian)|
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |            FACE CROP             |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |   ARCFACE / INSIGHTFACE ONNX     |
               |     (512-D Face Embedding)       |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |   COSINE SIMILARITY MATCHING     |
               +----------------------------------+
                             /     \
                Sim >= 0.50 /       \ Sim < 0.50
                           v         v
             +-----------------+   +------------------+
             | EXISTING FACE_ID|   |   NEW FACE_ID    |
             |   (Re-identify) |   | (Auto-register)  |
             +-----------------+   +------------------+
                           \         /
                            v       v
               +----------------------------------+
               |      VISIT LIFECYCLE ENGINE      |
               |      (1 ENTRY / 1 EXIT visit)    |
               +----------------------------------+
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
     +---------------+  +---------------+  +---------------+
     |  SQLITE DB    |  |   events.log  |  | EVENT IMAGES  |
     | (faces,events)|  |  Audit Log    |  | (entries/exits|
     +---------------+  +---------------+  +---------------+
                                |
                                v
               +----------------------------------+
               |      UNIQUE VISITOR COUNT        |
               |   COUNT(DISTINCT persistent_id)  |
               +----------------------------------+
```

---

## 4. Technologies Used

- **YOLO (YOLOv8-Face):** Real-time bounding box face detection (`models/yolov8n-face.pt`).
- **ArcFace / InsightFace (MobileFaceNet ONNX):** Deep facial feature extraction producing 512-D L2-normalized vectors (`models/insightface/w600k_mbf.onnx`).
- **Custom IoU + Hungarian Face Tracker:** Lightweight multi-object temporal association using Intersection-over-Union cost matrices and SciPy Hungarian linear sum assignment.
- **Python 3.13:** Core programming language.
- **OpenCV (`cv2`):** Video decoding, frame processing, face cropping, rendering GUI HUD overlays.
- **SQLite 3:** Persistent local database for faces and event records.
- **JSON:** Configuration file (`config.json`).

---

## 5. Project Structure

```text
katomaran/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Main execution CLI entry point
│   ├── config.py                # Configuration loader & system path resolver
│   ├── input/
│   │   ├── base_stream.py       # Abstract stream interface
│   │   ├── video_stream.py      # Local mp4/avi file ingestion
│   │   ├── rtsp_stream.py       # RTSP camera stream ingestion with reconnect
│   │   └── stream_factory.py    # Stream creator factory
│   ├── detection/
│   │   └── face_detector.py     # YOLOv8 face detector module
│   ├── recognition/
│   │   ├── face_recognizer.py   # ArcFace ONNX embedding generator
│   │   └── face_matcher.py      # Cosine similarity vector matcher & ID registry
│   ├── tracking/
│   │   └── face_tracker.py      # Custom IoU + Hungarian assignment tracker
│   ├── database/
│   │   ├── database.py          # SQLite database connection & schema manager
│   │   └── models.py            # FaceRecord and EventRecord dataclasses
│   ├── analytics/
│   │   ├── visit_manager.py     # State machine for 1 ENTRY / 1 EXIT visit lifecycle
│   │   └── visitor_counter.py   # Unique & active visitor counter engine
│   ├── pipeline/
│   │   └── processor.py         # End-to-end pipeline orchestrator
│   └── logging/
│       └── event_logger.py      # Formatted events.log logger
├── tests/                       # Pytest unit testing suite
│   ├── test_database.py
│   ├── test_face_detector.py
│   ├── test_face_recognizer.py
│   ├── test_face_tracker.py
│   ├── test_visit_manager.py
│   └── test_pipeline.py
├── data/
│   └── .gitkeep                 # Data directory (sample.mp4 placed here by user)
├── logs/
│   ├── entries/
│   │   └── .gitkeep             # Cropped ENTRY evidence face images
│   ├── exits/
│   │   └── .gitkeep             # Cropped EXIT evidence face images
│   └── events.log               # Human-readable event audit log
├── models/
│   ├── README.md                # Model weight placement instructions
│   └── .gitkeep
├── config.json                  # System configuration parameters
├── requirements.txt             # Python dependencies
├── README.md                    # Main documentation
├── AI_PLANNING.md               # AI planning and design breakdown
├── ARCHITECTURE.md              # Architecture diagram and specifications
└── .gitignore                   # Git ignore rules for large binaries and logs
```

---

## 6. Setup Instructions

### Prerequisites
- Python 3.10 or higher
- PyTorch & ONNX Runtime
- OpenCV (`opencv-python`)

### Installation Steps

1. **Clone Repository:**
   ```bash
   git clone https://github.com/your-repo/katomaran-face-tracker.git
   cd katomaran-face-tracker
   ```

2. **Create Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Download Model Weights:**
   Place model weight files according to [models/README.md](file:///d:/katomaran/models/README.md):
   - `models/yolov8n-face.pt`
   - `models/insightface/w600k_mbf.onnx`

4. **Download Sample Video:**
   Download the official Katomaran sample test video from Google Drive and place it at `data/sample.mp4`:
   - [Google Drive Sample Video Link](https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIfrUBFUJ?usp=sharing)

---

## 7. Configuration

System behavior is controlled via `config.json`:

```json
{
  "input": {
    "source_type": "video",
    "video_path": "data/sample.mp4",
    "rtsp_url": "rtsp://admin:12345@192.168.1.100:554/stream1"
  },
  "detection": {
    "model_path": "models/yolov8n-face.pt",
    "confidence_threshold": 0.5,
    "detection_skip_frames": 5
  },
  "recognition": {
    "model_path": "models/insightface/w600k_mbf.onnx",
    "similarity_threshold": 0.5,
    "minimum_face_size": 30
  },
  "tracking": {
    "max_disappeared_frames": 30
  },
  "database": {
    "path": "data/faces.db"
  },
  "logging": {
    "log_file": "logs/events.log"
  }
}
```

Key Parameters:
- `detection_skip_frames`: Number of frames between YOLO inference calls (extrapolates bounding boxes on skipped frames using motion vectors).
- `similarity_threshold`: Cosine similarity threshold (`0.50`) to consider a face a match for an existing identity.
- `max_disappeared_frames`: Number of consecutive missing frames (`30`) before confirming a visitor has exited.

---

## 8. Running the Application

### Validate Configuration
```bash
python -m app.main --check-config
```

### Run Full End-to-End Pipeline
```bash
python -m app.main --run
```

### Run Custom Video File
```bash
python -m app.main --run --config config.json
```

### RTSP Stream Testing
To switch to live camera testing, set `"source_type": "rtsp"` in `config.json` and provide your camera stream URL.

> [!NOTE]
> RTSP input is implemented and ready for live camera testing. Validation in this submission was performed using the provided sample video.

---

## 9. How Face Recognition Works

1. **Embedding Generation:** When a face crop is passed to ArcFace (InsightFace ONNX model), it yields a 512-element floating-point vector representing facial features (distance between eyes, nose bridge, jawline geometry). The vector is L2-normalized so its magnitude equals 1.0.
2. **Cosine Similarity Matching:** To compare a new face embedding $A$ against a registered embedding $B$, we compute the dot product:
   $$\text{Similarity} = \mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{512} A_i B_i$$
   - Values range from -1.0 to +1.0.
   - If $\text{Similarity} \ge 0.50$, the system recognizes the person as an existing identity.
   - If $\text{Similarity} < 0.50$, the person is recognized as a new visitor.

---

## 10. Auto-Registration & Persistent Identity

- **First Visit:** When a new face appears with no matching embedding in SQLite, the system automatically creates a persistent identity (e.g., `FACE_0001`), saves its embedding vector to SQLite, and logs a registration event.
- **Return Visit:** When `FACE_0001` leaves and returns later in the day (with a new temporary `track_id`), the embedding matcher links the new track to `FACE_0001`. The database identity is reused, preventing duplicate visitor records.

---

## 11. Entry / Exit Logging & Image Evidence

To guarantee reliable visit counts, the application uses a visit state engine:
`NEW → ACTIVE → MISSING → EXITED`

- **ENTRY Event:** Triggered **exactly once** when a face is confirmed active. A timestamped face crop image is saved to `logs/entries/YYYY-MM-DD/FACE_XXXX_ENTRY_...jpg`.
- **EXIT Event:** Triggered **exactly once** when a face disappears for more than `max_disappeared_frames` (30 frames). A timestamped face crop image is saved to `logs/exits/YYYY-MM-DD/FACE_XXXX_EXIT_...jpg`.

---

## 12. Unique Visitor Counting

Unique Visitor Count is calculated as:
$$\text{Unique Visitors} = \text{COUNT}(\text{DISTINCT } \text{face\_id})$$

Temporary `track_id` values assigned by the tracker are discarded upon exit and never used for counting visitors. If `FACE_0001` enters, exits, and returns 5 times, **Unique Visitor Count remains 1**.

---

## 13. Database Schema

The SQLite database (`data/faces.db`) maintains two main tables:

### `faces` Table
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Autoincrement surrogate key |
| `face_id` | TEXT UNIQUE | Persistent identity (e.g. `FACE_0001`) |
| `embedding` | BLOB | 512-float32 binary embedding vector |
| `registered_at` | TEXT | Timestamp of first registration |
| `last_seen_at` | TEXT | Timestamp of most recent appearance |
| `first_crop_path` | TEXT | Path to initial cropped face image |

### `events` Table
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Autoincrement event ID |
| `face_id` | TEXT | Foreign key to `faces(face_id)` |
| `event_type` | TEXT | Event type (`ENTRY` or `EXIT`) |
| `timestamp` | TEXT | Timestamp string |
| `image_path` | TEXT | Path to saved evidence crop image |
| `tracking_id` | INTEGER | Temporary tracker track_id |
| `confidence` | REAL | Face detection confidence score |
| `similarity` | REAL | Cosine similarity match score |

---

## 14. Sample Output & Benchmark Results

Ran full end-to-end execution on the official Katomaran sample video (`data/sample.mp4`):

```text
====================================================================
[PIPELINE] Pipeline Execution Summary:
  Total Frames Processed:    732
  Elapsed Processing Time:   17.29s
  Average Processing FPS:    42.3
  Unique Visitors Count:     13
  Total Entry Events:        21
  Total Exit Events:         19
====================================================================
```

### Empirical Metrics Table
| Metric | Value |
|---|---|
| Total Video Frames Processed | 732 |
| Processing Speed | 42.3 FPS (CPU execution) |
| **Unique Visitor Count (`COUNT(DISTINCT face_id)`)** | **13** |
| Total ENTRY Events Logged | 21 |
| Total EXIT Events Logged | 19 |
| Active Visitors Remaining in Frame at End | 2 |
| Saved Entry Evidence Images (`logs/entries/`) | 21 |
| Saved Exit Evidence Images (`logs/exits/`) | 19 |
| Total SQLite Event Records (`events` table) | 40 |
| Pytest Unit Test Results | 31 passed in 6.09s |

---

## 15. Performance & Compute Estimates

- **CPU Workload:** Video decoding, bounding box scaling, IoU calculation, Hungarian assignment, SQLite transactions, image cropping & saving.
- **GPU Acceleration (CUDA available):** YOLOv8 face detection inference and ArcFace ONNX embedding extraction run on GPU if PyTorch/ONNX CUDA execution providers are detected; otherwise automatically fallback to optimized CPU threads.
- **Benchmark Performance:** Achieved ~42 FPS on standard CPU execution using 5-frame detection skipping.

---

## 16. Assumptions

1. A visitor is considered to have exited after being unobserved for `max_disappeared_frames` (30 consecutive frames).
2. Development and offline validation were performed on the provided Katomaran sample video file.
3. RTSP input stream code is fully supported and ready for live camera demonstration.
4. Persistent face identity association is based on L2-normalized 512-D ArcFace cosine similarity thresholds.

---

## 17. AI Development Process

Artificial Intelligence coding assistance (Antigravity AI) was utilized during development to assist with modular code architecture generation, refactoring, test case generation, error log analysis, and documentation writing. All code, algorithms, unit tests, and runtime metrics were rigorously reviewed, tested, and empirically validated on local sample inputs.

---

## 18. Demo Video

Demo Video: [(https://drive.google.com/drive/folders/1iiGLGQQlj00KInTIpUPnfQFhHD877_Fi?usp=sharing)]

---

