# Model Assets Documentation

This directory contains the neural network model weights required for real-time face detection and face recognition.

> [!NOTE]
> Large model binary files (`.pt`, `.onnx`) are excluded from Git repository tracking via `.gitignore`. You must place the required model weight files into their respective paths before running the application.

---

## 1. Face Detection Model

- **Model Name:** YOLOv8n-Face
- **Purpose:** Real-time face detection and bounding box extraction.
- **Expected File Path:** `models/yolov8n-face.pt`
- **Framework:** PyTorch / Ultralytics YOLOv8
- **Placement:** Download `yolov8n-face.pt` and place it directly inside the `models/` directory.

---

## 2. Face Recognition & Embedding Model

- **Model Name:** ArcFace / InsightFace (MobileFaceNet - `w600k_mbf`)
- **Purpose:** 512-dimensional facial feature embedding extraction for face matching and identity recognition.
- **Expected File Path:** `models/insightface/w600k_mbf.onnx`
- **Framework:** ONNX Runtime
- **Placement:** Create the `models/insightface/` directory and place `w600k_mbf.onnx` inside it.

---

## Model Path Verification

When launching the application, `config.json` verifies these paths automatically:

```json
{
  "detection": {
    "model_path": "models/yolov8n-face.pt"
  },
  "recognition": {
    "model_path": "models/insightface/w600k_mbf.onnx"
  }
}
```
