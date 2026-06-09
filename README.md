# DFDetective: A Multimodal Forensic System for Deepfake Detection and Disruption

DFDetective is a research-grade, multimodal forensic system designed to identify and disrupt manipulated media across spatial, temporal, and acoustic domains. The system integrates hardware-accelerated deep learning models, an interactive forensic analysis dashboard, a real-time browser extension for client-side triage, and an adversarial immunization engine to proactively protect source media from unauthorized synthetic manipulation.

---

## System Architecture
| System Layer | Component | Technical Stack | Core Responsibility | Data Flows & Interfaces |
| :--- | :--- | :--- | :--- | :--- |
| **Client Interface** | Streamlit Forensic Dashboard | Streamlit, Matplotlib, ReportLab | Interactive media upload, forensic rendering, and analytical PDF export. | Ingests user media; requests analytical charts and outputs from the ML inference pipeline. |
| **Client Interface** | Browser Extension | Manifest V3, JavaScript, ONNX Web | Passive DOM inspection, hash caching checks, and local WebGL classification. | Computes image SHA-256; queries REST API; falls back to local execution on database misses. |
| **Service API** | REST API Backend | PHP, Apache | Proxy endpoint for hash verification and secure report logging. | Receives signature-verified payloads from the extension; queries and updates the database. |
| **Inference Pipeline** | Ensemble ML Pipeline | PyTorch XPU (IPEX), OpenVINO, MTCNN | Executes multi-branch networks evaluating spatial, temporal, and acoustic features. | Processes extracted video frames and vocals; returns weighted late-fusion probability scores. |
| **Forensic Analytics** | Forensic Engine | OpenCV, Pillow, Librosa | Generates Error Level Analysis, FFT magnitude heatmaps, and audio spectrograms. | Transforms uploaded media assets into raw visual metrics for display and PDF rendering. |
| **Adversarial Defense** | NoiseNet Protection | TensorFlow, Keras | Computes and embeds imperceptible perturbations to immunize source images. | Modifies pixel arrays under constraint ($\epsilon \le 0.01$, SSIM > 0.95) to disrupt unauthorized generative models. |
| **Data Store** | MySQL Cache Store | MySQL | Structured index of scanned file hashes and associated classification states. | Serves as the primary query target for the REST API verification backend. |

The system is organized into decoupled layers to ensure security, modularity, and high-performance execution:

*   **Client Layer:** Comprises the interactive Streamlit Dashboard for detailed analyst-level investigations and the Chrome Extension for passive, real-time background inspection of assets as they render in-browser.
*   **Service API Layer:** A signature-verified REST API serving as a verification router. It cross-references computed media hashes against historical logs.
*   **Inference Pipeline:** A multi-branch neural network structure that performs late-fusion evaluation of extracted spatial, temporal, and acoustic vectors.
*   **Adversarial Immunization:** An independent proactive pipeline that optimizes high-frequency noise inputs to immunize target images against unauthorized generative models.

---

## Core Modules

### Multimodal Detection Engine
*   **3D Spatiotemporal Video Classifier (VideoCNNLSTM):** Extracts frame-level spatial feature maps via 3D convolutions and models temporal transitions using a dual-layer Long Short-Term Memory (LSTM) network to flag blending anomalies and temporal inconsistencies. Accelerated via Intel OpenVINO.
*   **2D Spatial Image Classifiers:** A model ensemble combining patch-level Vision Transformers (ViT) for boundary analysis and a dual-path ResNet-50 architecture. The ResNet model processes the spatial pixel domain in path 1, and processes the frequency domain (via 2D Fast Fourier Transform magnitude and phase spectra) in path 2 to identify typical generator grid artifacts.
*   **Acoustic Spectrogram CNN:** A 2D CNN that processes Mel-spectrogram voice representations of extracted audio tracks to identify vocal anomalies common in synthetic speech. Powered by PyTorch XPU/Intel Extension for PyTorch (IPEX) for hardware-accelerated processing.
*   **Weighted Late-Fusion:** A mathematical fusion layer that combines isolated confidence scores (Video: 45%, Image: 35%, Audio: 20%) to determine the authenticity of a multimodal asset.

### Proactive Adversarial Immunization
*   **NoiseNet Autoencoder:** A 6-layer symmetric convolutional neural network that generates imperceptible adversarial perturbations ($\epsilon \le 0.01$, SSIM > 0.95) mapped to source image channels. The generated perturbations degrade the performance of GANs and diffusion-based deepfake architectures when they attempt to reconstruct or modify the immunized image.

### Forensic Analysis & Web Dashboard
*   **Analytical Visualizations:** Generates Error Level Analysis (ELA) for compression delta detection, FFT spectrum heatmaps for high-frequency noise profiles, and temporal frame-by-frame probability graphs.
*   **Automated Document Compilation:** Integrates forensic analysis metrics and visualizations into a comprehensive PDF report using ReportLab.

### Real-Time Browser Triage (Chrome Extension)
*   **Client-Side Verification:** Computes SHA-256 hashes of online media elements locally and queries the backend database cache to bypass unnecessary network inference.
*   **On-Device ONNX Engine:** If a database miss occurs, the extension runs local browser-side inference of the 2D image model using ONNX Runtime Web via WebGL/WASM execution providers.

### Hash Synchronization & Backend Database
*   **Verification API:** A PHP-based backend using HMAC-SHA256 headers to validate requests.
*   **Persistence Schema:** Stores unique file signatures, referrers, and classification states in a normalized MySQL database to build a federated cache of evaluated media.

---

## Project Directory Structure

```
DeepFake-Detection/
├── analytics/             # Compiled forensic reports and model metrics
├── chrome_extension/      # Manifest V3 browser extension codebase
│   ├── background/        # Service worker for API queries and CORS bypass
│   ├── content/           # DOM media interception and local ONNX runtime
│   ├── lib/               # ONNX Runtime Web WASM/WebGL dependencies
│   └── popup/             # Extension popup console and detection logs
├── config/                # Training configurations and hyper-parameters
├── dashboard/             # Streamlit web application
│   ├── utils/             # Image, audio preprocessing and PDF builders
│   └── app.py             # Dashboard application controller
├── hostinger_api/         # PHP cloud sync API and database configuration
├── openvino_model/        # Intermediate Representations (IR) for OpenVINO
├── saved_models/          # Trained model weights (HDF5 and PyTorch checkpoints)
├── scripts/               # Model conversion and developer utility scripts
└── src/                   # Python core package
    ├── models/            # Model architecture definitions (Keras & PyTorch)
    ├── preprocessing/     # Frame extraction and audio parsing
    └── predict_ensemble.py # Local late-fusion CLI predictor
```

---

## Technical Limitations & Forensic Compliance

*   **Forensic Verification:** The outputs of the system serve as investigative indicators rather than definitive legal proof. Factors such as extreme compression, resolution downscaling, or successive re-encoding can influence classifier accuracy.
*   **Dynamic Weighted Fusion:** The late-fusion system dynamically redistributes modality weights to handle partial media formats (e.g., videos containing no audio stream or standalone static images) without throwing runtime errors.
