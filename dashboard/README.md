# Deepfake Detection Dashboard

Simple Streamlit-based dashboard for multimodal deepfake detection.

## Quick Start

```bash
# Activate venv (already done)
.\venv\Scripts\Activate.ps1

# Run dashboard
streamlit run dashboard/app.py
```

Opens at: **http://localhost:8501**

## Features

### Upload & Analyze
- **Image Analysis** - Upload JPG/PNG for deepfake detection
- **Video Analysis** - Upload MP4/MOV (extracts 30 frames)
- **Audio Analysis** - Upload MP3/WAV (mel spectrogram)

### Results
- Confidence score (0-100%)
- Verdict (Deepfake / Authentic)
- JSON report export

### Pages
1. **📤 Analyze** - Main upload & detection interface
2. **📊 Metrics** - Training metrics (placeholder)
3. **ℹ️ About** - Project information

## Supported File Types

| Type | Extensions |
|------|-----------|
| Image | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp` |
| Video | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac` |

## Current Capabilities

✅ **Working:**
- File upload interface
- Media preprocessing (image, video, audio)
- Mock inference (for testing UI)
- Results display
- JSON export

🚧 **Coming Soon:**
- Real model integration (once training complete)
- Training metrics visualization
- PDF report generation
- Batch analysis

## Architecture

```
dashboard/
└── app.py          # Single-file Streamlit app with all functionality
```

**All-in-one design** - Everything in one file for simplicity. Will be modularized later if needed.

## Model Integration

To use real models once trained:

1. Update `load_models()` function in `app.py`
2. Replace mock inference with real predictions:
   ```python
   # Instead of: confidence = float(np.random.uniform(0.3, 0.9))
   # Use: output = model.predict(preprocessed_data)
   ```

## Environment

- Python 3.10+
- Streamlit 1.28.0
- TensorFlow 2.13+ (for models when ready)
- OpenCV, Librosa (preprocessing)

## Troubleshooting

**"Cannot find streamlit"**
```bash
.\venv\Scripts\Activate.ps1
pip install streamlit
```

**"Port 8501 already in use"**
```bash
streamlit run dashboard/app.py --server.port=8502
```

**"Module not found" error**
```bash
pip install -r requirements.txt
# or individual packages
```

---

**Status:** ✅ Dashboard ready | 🚧 Models pending training | 📅 Phase 4
