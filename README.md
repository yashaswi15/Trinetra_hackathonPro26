# 👁️ TRINETRA — The All-Seeing Eye

> **AI-Powered Real-Time Fight & Aggressive Behavior Detection System**
> Built with CNN + Bidirectional LSTM Dual Model Architecture

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-95.78%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Achievement ^_^

> **Winner — Hackstreet 4.0** | SRM University KTR, Chennai
> Built in under 24 hours by Team Segfault

---

## Table of Contents

- [The Idea](#the-idea)
- [Problem Statement](#problem-statement)
- [Our Solution](#our-solution)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Dataset Sources](#dataset-sources)
- [Development Journey](#development-journey)
- [Model Pipeline](#model-pipeline)
- [Dual Model Strategy](#dual-model-strategy)
- [Web Dashboard](#web-dashboard)
- [Results](#results)
- [Challenges and Solutions](#challenges-and-solutions)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Future Roadmap](#future-roadmap)
- [Team](#team)

---

## 💡 The Idea

The idea for TRINETRA came from a simple observation — **cameras are everywhere, but nobody is really watching them.**

In India alone, millions of CCTV cameras are installed across schools, metro stations, malls, hospitals, and public spaces. Yet surveillance is still a reactive process — you review the footage *after* the incident. By the time a human operator notices something is wrong, calls for backup, and someone physically reaches the spot — **4 to 7 minutes have passed.**

In a fight, that is an eternity.

We asked: *What if the camera could watch itself? What if an AI could detect violence the moment it begins and alert the right people instantly, without any human needing to watch the screen?*

That question built **TRINETRA** — named after the third eye of Lord Shiva in Hindu mythology. The eye that sees what ordinary eyes cannot. The eye that acts with absolute precision the moment it sees evil.

---

## Problem Statement~

Traditional CCTV surveillance systems suffer from three fundamental problems:

| Problem | Impact |
|---|---|
| **Human fatigue** | Operators miss incidents due to attention lapses |
| **Slow response** | 4–7 minute average reaction time to detected incidents |
| **Unscalable** | One operator cannot monitor 50+ cameras simultaneously |
| **Reactive only** | Footage reviewed after damage is done |

According to research, up to 95% of CCTV footage is never reviewed in real time. The infrastructure exists — the intelligence does not.

---

## ✅ Our Solution

TRINETRA is a **real-time AI-powered fight and aggressive behavior detection system** that:

- Watches live camera feeds 24/7 without human intervention
- Detects fights using a dual deep learning model architecture
- Alerts security personnel within 1 second of detection
- Provides a professional web dashboard with live feeds, threat levels, and alert logs
- Auto-saves timestamped screenshots at every fight event
- Supports multi-camera surveillance infrastructure

---

## 🏗️ System Architecture

```
Live Camera Feed (Webcam / IP Camera)
            │
            ▼
    Frame Extraction
    (16 frames per window, 224×224, uniform sampling)
            │
            ▼
    EfficientNet-B0 (Frozen, Pretrained ImageNet)
    Feature Extraction → 1280-dim vector per frame
            │
            ▼
    Feature Matrix: (16, 1280) — 16 timesteps
            │
      ┌─────┴──────┐
      ▼            ▼
  BiLSTM       CNN Image
  (temporal    Classifier
   patterns)   (appearance)
      │            │
      └─────┬──────┘
            ▼
      AND Logic Gate
      (both must agree)
            │
      ┌─────┴─────┐
      ▼           ▼
    FIGHT        SAFE
      │
      ▼
Flask + WebSocket
      │
      ▼
TRINETRA Dashboard (Browser)
+ Auto Screenshot Saved
+ Alert Log Updated
+ Notification Chain Triggered
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10 |
| **Deep Learning** | PyTorch 2.x |
| **CNN Backbone** | EfficientNet-B0 (pretrained ImageNet) |
| **Sequence Model** | Bidirectional LSTM (2 layers, hidden=256) |
| **Image Classifier** | Fine-tuned EfficientNet-B0 (UCF Crime Dataset) |
| **Computer Vision** | OpenCV 4.x |
| **Web Backend** | Flask + Flask-SocketIO |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript, WebSocket |
| **GPU** | NVIDIA RTX 4050 Laptop GPU |
| **CUDA** | 12.1 |
| **Environment** | Conda, Python 3.10 |
| **Data Format** | NumPy .npy (pre-extracted features) |

---

## 📦 Dataset Sources

### 1. 🏒 Hockey Fight Dataset
- **Source:** [Kaggle — Hockey Fight Detection](https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes)
- **Total Size:** 1,000 videos
- **Breakdown:** 500 fight + 500 non-fight
- **Format:** .avi video files
- **Used for:** Fight class training videos
- **Why:** Real sports footage — fast motion, physical contact, crowd background

### 2. 🎥 Real Life Violence Situations Dataset
- **Source:** [Kaggle — Real Life Violence Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- **Total Size:** 2,000 videos
- **Breakdown:** 1,000 fight + 1,000 non-fight
- **Format:** .mp4 video files
- **Used for:** Both fight and non-fight training videos
- **Why:** Real-world street violence + everyday non-violent scenes — diverse environments

### 3. 🚨 UCF Crime Dataset
- **Source:** [Kaggle — UCF Crime Dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset)
- **Total Size:** 1,377,653 images (1,266,345 train + 111,308 test)
- **Classes:** 14 crime categories across real surveillance footage
- **Categories used:**
  - Fight → Fighting + Assault + Abuse classes
  - Non-fight → NormalVideos class
- **Format:** .jpg surveillance images
- **Used for:** Training CNN image classifier (Model 2)
- **Why:** Real CCTV surveillance appearance — trains the model on what fight scenes *look* like from security cameras

### Final Dataset Scale

```
╔══════════════════════════════════════════════════════════╗
║              TOTAL DATA USED BY TRINETRA                 ║
╠══════════════════════════════════════════════════════════╣
║  Videos (LSTM training)                                  ║
║  ├── Hockey Fight Dataset        :     1,000 videos      ║
║  ├── Real Life Violence Dataset  :     2,000 videos      ║
║  └── Total (balanced)            :     3,000 videos      ║
║                                                          ║
║  Images (CNN classifier training)                        ║
║  └── UCF Crime Dataset           : 1,377,653 images      ║
║      ├── Train subset            : 1,266,345 images      ║
║      └── Test subset             :   111,308 images      ║
║                                                          ║
║  Extracted frames (16 per video)                         ║
║  └── Total frames processed      :    48,000 frames      ║
╚══════════════════════════════════════════════════════════╝

Video split (Train / Val / Test): 70% / 15% / 15%
├── Train : 2,100 videos  →  33,600 frames
├── Val   :   450 videos  →   7,200 frames
└── Test  :   450 videos  →   7,200 frames

Frame extraction settings:
├── 16 frames per video (uniform sampling)
├── Resize: 224 × 224 pixels
└── Normalize: ImageNet mean/std [0.485, 0.456, 0.406]
```

---

## 🚀 Development Journey

### Phase 1 — Research and Setup
We started by researching existing fight detection approaches. Most papers used optical flow or 3D CNNs which are computationally expensive. We chose the **CNN feature extraction + LSTM** approach because EfficientNet-B0 is fast and accurate, LSTM naturally handles temporal sequences, and pre-extracting features means LSTM training takes seconds not hours.

Set up Conda environment with PyTorch + CUDA 12.1 on RTX 4050 laptop.

### Phase 2 — Dataset Collection and Frame Extraction
Combined Hockey Fight and Real Life Violence datasets for 3000 balanced videos. Built `main.py` to extract 16 uniformly sampled frames per video using OpenCV. Tested on 50+50 videos first, then ran the full dataset.

**Problem encountered:** Some videos had H.264 codec warnings from OpenCV.
**Solution:** Errors caught gracefully — 0 failed videos out of 3000.

### Phase 3 — CNN Feature Extraction
Built `feature_extractor.py` to run all frames through frozen EfficientNet-B0. This produced `.npy` feature files of shape `(N, 16, 1280)` — the entire dataset compressed to 234 MB.

**Key insight:** Pre-extracting features means the LSTM trains on tiny numpy arrays, not images. Total training time: ~15 seconds for 21 epochs.

### Phase 4 — LSTM Training
Built `FightLSTM` — a Bidirectional LSTM with input projection (1280→512), BiLSTM (2 layers, hidden=256, bidirectional → 512 output), and a classifier head (512→128→2).

Used AdamW optimizer, CrossEntropy with label smoothing 0.1, ReduceLROnPlateau scheduler, and gradient clipping (max_norm=1.0).

**Result:** 95.78% test accuracy in 21 epochs with early stopping.

### Phase 5 — The False Positive Problem
**Problem:** When testing on webcam, dancing and fast movement triggered FIGHT alerts at 80–100% confidence.

**Root cause:** Non-fight training data (Hockey + Real Life Violence) contained mostly calm scenes. The model never learned that fast motion does not always mean fighting. Also, testing with no shirt on caused the model to associate exposed skin with fight scenes (fight videos often show torn clothing).

**Solution:** Built a second model — CNN image classifier trained on UCF Crime dataset images. Applied AND logic so both models must agree before firing an alert.

### Phase 6 — CNN Image Classifier
Fine-tuned EfficientNet-B0 on UCF Crime images. Unfroze last 3 feature blocks, added new classifier head (1280→256→2), used cosine annealing LR scheduler and label smoothing.

**Result:** 99.33% validation accuracy.

### Phase 7 — Real-Time Prediction System
Built `predict_dual.py` and `app.py` — loads both models, processes webcam frames in a rolling 16-frame buffer, applies AND logic with smoothing over 5 predictions.

### Phase 8 — Web Dashboard
Built the TRINETRA dashboard using Flask + WebSockets. Live camera feed streamed as base64 JPEG. Features: 6-camera grid, threat ring, heatmap overlay, probability chart, alert log, live event ticker, auto screenshots.

**Problem:** `render_template` returned blank page.
**Solution:** Read HTML file directly with `open()` and `utf-8` encoding.

**Problem:** Screenshot filenames had `\f` escape character bug.
**Solution:** Used `os.path.join()` for cross-platform safe paths.

### Phase 9 — Hackathon Win
Named the system TRINETRA after Shiva's third eye. Built complete pitch scripts, README, and jury Q&A preparation for all 3 team members.

**Result: Won Hackstreet 4.0** 🏆

---

## 🔬 Model Pipeline

### Step 1 — Frame Extraction
```python
# 16 uniformly sampled frames per video
step = total_frames // 16
frame = cv2.resize(frame, (224, 224))
```

### Step 2 — CNN Feature Extraction
```python
# EfficientNet-B0 frozen, outputs 1280-dim per frame
# Processes all 16 frames in one GPU batch
features = extractor(frames).view(B, T, -1)  # (B, 16, 1280)
```

### Step 3 — LSTM Training
```python
# BiLSTM processes (batch, 16, 1280) sequences
# Uses last timestep output for classification
lstm_out, _ = self.lstm(x)         # (B, 16, 512)
last_out    = lstm_out[:, -1, :]   # (B, 512)
logits      = self.classifier(last_out)  # (B, 2)
```

### Step 4 — Dual Model Inference
```python
# AND logic — both must exceed threshold
lstm_says_fight = smoothed_lstm >= 0.90
cnn_says_fight  = smoothed_cnn  >= 0.50
label = "FIGHT" if (lstm_says_fight and cnn_says_fight) else "SAFE"
```

---

## 🧠 Dual Model Strategy

```
Single model problem:
  Fast motion (dancing, sports) → misclassified as FIGHT
  80–100% confidence on wrong predictions

Dual model solution:

  Model 1: CNN + LSTM
  ├── Sees 16 frames (temporal patterns)
  ├── Detects: HOW the person moves over time
  └── Threshold: 90%

  Model 2: CNN Image Classifier
  ├── Sees 1 frame (appearance)
  ├── Detects: WHAT the scene looks like
  └── Threshold: 50%

  AND Logic:
  FIGHT = Model1 says FIGHT AND Model2 says FIGHT
  Otherwise → SAFE

Result: False positives dramatically reduced
```

---

## 🖥️ Web Dashboard

The TRINETRA dashboard is a production-grade real-time surveillance interface.

**Features:**
- Live webcam feed with corner bracket overlay
- Threat level ring (0–10 scale, color coded green→amber→red)
- FIGHT / SAFE status with animated indicators and glow effects
- Fight probability chart (last 60 readings)
- 6-camera grid (1 live + 5 offline with animated static noise)
- LSTM + CNN confidence meters (turn red at threshold)
- Detection heatmap overlay on live feed
- Real-time alert log with timestamps and icons
- Live event ticker (scrolling bottom bar)
- 4-tone audio alert on fight detection
- Auto-screenshot every 2 seconds during fight
- WebSocket real-time updates (no page refresh needed)

**Access:** `http://localhost:5000`

---

## 📊 Results

| Metric | Value |
|---|---|
| LSTM Test Accuracy | **95.78%** |
| Fight Detection Accuracy | **96.46%** |
| Non-Fight Detection Accuracy | **95.09%** |
| CNN Image Classifier Val Accuracy | **99.33%** |
| Training Time (LSTM, 21 epochs) | **~15 seconds** |
| Feature Extraction Time (3000 videos) | **~15 minutes** |
| Inference Speed | **10–14 FPS** real-time |
| End-to-End Latency (detection to dashboard) | **~60–80ms** |
| Fight-to-Alert Latency | **< 1 second** |

---

## 🧩 Challenges and Solutions

### RAM overflow with large dataset
**Problem:** Loading 3000 videos × 16 frames × 224×224 × 3 channels into RAM requires ~10GB.
**Solution:** Custom PyTorch Dataset class loads frames on-demand. Pre-extracted CNN features (234 MB) load fully into RAM safely.

### False positives on fast motion
**Problem:** Dancing, sports, fast arm movement → LSTM confident it was a fight.
**Solution:** Dual model AND logic. CNN image classifier blocks false alarms from fast-but-harmless motion.

### Windows multiprocessing crash
**Problem:** `num_workers > 0` in DataLoader crashed on Windows.
**Solution:** `num_workers=0` on Windows. GPU-bound extraction doesn't need multiprocessing.

### Blank dashboard page
**Problem:** `render_template("index.html")` returned blank white page.
**Solution:** Read HTML directly: `open(html_path, "r", encoding="utf-8").read()`

### Screenshot path bug
**Problem:** `f"{DIR}\fight_..."` — `\f` treated as form feed escape character.
**Solution:** `os.path.join(SCREENSHOT_DIR, filename)` for safe cross-platform paths.

### CNN model key mismatch
**Problem:** Saved model keys had no prefix, loaded model expected `net.` prefix.
**Solution:** Restructured `CNNImageClassifier` to expose `self.features`, `self.avgpool`, `self.classifier` directly.

---

## 📁 Project Structure

```
hackstreet_26/
│
├── datasets/
│   ├── final_dataset/
│   │   ├── fight/              ← 1500 fight videos
│   │   └── non_fight/          ← 1500 non-fight videos
│   ├── processed/
│   │   ├── fight/              ← extracted frames (16 per video)
│   │   └── non_fight/
│   ├── features/
│   │   ├── features_train.npy  (2100, 16, 1280)
│   │   ├── features_val.npy    ( 450, 16, 1280)
│   │   ├── features_test.npy   ( 450, 16, 1280)
│   │   ├── labels_train.npy
│   │   ├── labels_val.npy
│   │   └── labels_test.npy
│   ├── ucf_crime/
│   │   ├── Train/
│   │   │   ├── Fighting/
│   │   │   ├── Assault/
│   │   │   ├── Abuse/
│   │   │   └── NormalVideos/
│   │   └── Test/
│   └── ucf_binary/
│       ├── train/
│       │   ├── fight/
│       │   └── non_fight/
│       └── val/
│
├── models/
│   ├── best_model.pth          ← trained LSTM (95.78% test acc)
│   ├── last_model.pth
│   ├── cnn_image_model.pth     ← CNN classifier (99.33% val acc)
│   └── training_log.csv
│
├── screenshots/
│   └── fight_YYYYMMDD_HHMMSS_lstmXX_cnnYY.jpg
│
└── work/
    ├── main.py                 ← frame extraction
    ├── dataset.py              ← PyTorch Dataset class
    ├── feature_extractor.py    ← EfficientNet feature extraction
    ├── lstm_model.py           ← FightLSTM model definition
    ├── train_lstm.py           ← LSTM training script
    ├── cnn_image_classifier.py ← CNN classifier training
    ├── predict.py              ← single model OpenCV prediction
    ├── predict_dual.py         ← dual model OpenCV prediction
    ├── app.py                  ← Flask + SocketIO web server
    └── templates/
        └── index.html          ← TRINETRA web dashboard
```

---

## ⚙️ How to Run

### Prerequisites

```bash
conda create -n ml_env python=3.10
conda activate ml_env

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python flask flask-socketio tqdm Pillow
```

### Step 1 — Prepare Dataset
Download datasets and place videos in:
```
datasets/final_dataset/fight/
datasets/final_dataset/non_fight/
```

### Step 2 — Extract Frames
```bash
python work/main.py
```

### Step 3 — Extract CNN Features
```bash
python work/feature_extractor.py
```

### Step 4 — Train LSTM
```bash
python work/train_lstm.py
```

### Step 5 — Train CNN Image Classifier
```bash
python work/cnn_image_classifier.py
```

### Step 6 — Run Web Dashboard
```bash
python work/app.py
```
Open browser: `http://localhost:5000`
Click **AUDIO** button to enable alert sound.

### Step 7 — Run Standalone (OpenCV window)
```bash
python work/predict_dual.py --source webcam
python work/predict_dual.py --source video --path "path/to/video.mp4"
```

---

## 🔮 Future Roadmap

### Short Term
- [ ] RTSP IP camera support
- [ ] Twilio SMS alerts to security personnel
- [ ] PostgreSQL database for persistent incident logging
- [ ] Multi-camera threading

### Medium Term
- [ ] Edge deployment on Jetson Nano
- [ ] Add hard negative training data (dance, sports videos)
- [ ] WebRTC for lower latency streaming
- [ ] REST API for third-party integration

### Long Term
- [ ] Police dispatch system integration
- [ ] Smart city scale deployment
- [ ] Night vision / infrared support
- [ ] Drone surveillance integration

---

## 👥 Team

**Team Segfault — Hackstreet 4.0, SRM University KTR**

| Member | Role |
|---|---|
| **Ayushman Yashaswi** | ML Pipeline — Dataset, CNN+LSTM models, training, dual model strategy · Backend — Flask server, WebSocket, alert system |
| **Dweep Khatki** | Frontend + Pitch — Dashboard UI, presentation, jury Q&A |

---

## 📄 License

MIT License — free to use, modify and distribute with attribution.

---

## 🙏 Acknowledgements

- [Hockey Fight Dataset](https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes) — Kaggle
- [Real Life Violence Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) — Kaggle
- [UCF Crime Dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) — Kaggle
- [EfficientNet-B0](https://arxiv.org/abs/1905.11946) — Tan & Le, Google Brain
- [PyTorch](https://pytorch.org/) — Meta AI
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/)

---

<div align="center">

**TRINETRA — The All-Seeing Eye**

*"It never closes. It never misses. It never forgives."*

👁️

</div>
