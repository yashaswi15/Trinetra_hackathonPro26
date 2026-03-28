"""
app.py
------
TRINETRA — Fight Detection Web Dashboard
Flask backend with WebSocket for real-time streaming.

Install dependencies:
    pip install flask flask-socketio --break-system-packages

Usage:
    python work/app.py
    Then open: http://localhost:5000
"""

import cv2
import os
import time
import base64
import threading
import collections
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

from flask import Flask, jsonify
from flask_socketio import SocketIO

from lstm_model import FightLSTM


# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
LSTM_MODEL_PATH = r"D:\projects__\hackstreet_26\models\best_model.pth"
CNN_MODEL_PATH  = r"D:\projects__\hackstreet_26\models\cnn_image_model.pth"

NUM_FRAMES      = 16
FRAME_SIZE      = (224, 224)
BUFFER_STEP     = 2
SMOOTH_WINDOW   = 5

LSTM_THRESHOLD  = 0.90
CNN_THRESHOLD   = 0.50    # lowered from 0.80 to catch more real fights

CAMERA_INDEX    = 0
STREAM_WIDTH    = 640
STREAM_HEIGHT   = 480
JPEG_QUALITY    = 70

MAX_ALERT_LOG    = 50
MAX_PROB_HISTORY = 60

# Screenshot config
SCREENSHOT_DIR      = r"D:\projects__\hackstreet_26\screenshots"
SCREENSHOT_INTERVAL = 2   # save screenshot every 2 seconds during a fight


# ══════════════════════════════════════════════
#  FLASK + SOCKETIO SETUP
# ══════════════════════════════════════════════
app = Flask(__name__)
app.config["SECRET_KEY"] = "trinetra_secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ══════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════
class EfficientNetExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = net.features
        self.pool     = net.avgpool
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


class CNNImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.efficientnet_b0(weights=None)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 2),
        )
        self.features   = net.features
        self.avgpool    = net.avgpool
        self.classifier = net.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)


TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════
#  GLOBAL STATE
# ══════════════════════════════════════════════
state = {
    "running":      False,
    "label":        "SAFE",
    "lstm_prob":    0.0,
    "cnn_prob":     0.0,
    "fps":          0.0,
    "alerts_today": 0,
    "start_time":   None,
    "alert_log":    [],
    "prob_history": [],
    "last_fight":   False,
}
state_lock = threading.Lock()


# ══════════════════════════════════════════════
#  DETECTION THREAD
# ══════════════════════════════════════════════
def detection_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # Load models
    print("[Models] Loading...")
    extractor = EfficientNetExtractor(device)

    lstm = FightLSTM(input_dim=1280, hidden_dim=256, num_layers=2).to(device)
    ckpt = torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=False)
    lstm.load_state_dict(ckpt["model_state_dict"])
    lstm.eval()

    cnn_clf = CNNImageClassifier().to(device)
    ckpt2   = torch.load(CNN_MODEL_PATH, map_location=device, weights_only=False)
    sd = ckpt2["model_state_dict"]
    cnn_clf.load_state_dict(sd)
    cnn_clf.eval()
    print("[Models] All loaded ✓")
    print(f"[Config] LSTM threshold : {LSTM_THRESHOLD}")
    print(f"[Config] CNN threshold  : {CNN_THRESHOLD}")
    print(f"[Config] Screenshots    : {SCREENSHOT_DIR}")

    # Create screenshot directory
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)

    frame_buffer         = collections.deque(maxlen=NUM_FRAMES)
    lstm_history         = collections.deque(maxlen=SMOOTH_WINDOW)
    cnn_history          = collections.deque(maxlen=SMOOTH_WINDOW)
    fps_history          = collections.deque(maxlen=30)
    new_frame_count      = 0
    prev_time            = time.time()
    last_screenshot_time = 0.0

    smoothed_lstm = 0.0
    smoothed_cnn  = 0.0
    label         = "SAFE"

    with state_lock:
        state["running"]    = True
        state["start_time"] = time.time()

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            continue

        # FPS
        now = time.time()
        fps_history.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = sum(fps_history) / len(fps_history)

        # Preprocess
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, FRAME_SIZE)
        tensor  = TRANSFORM(resized)
        frame_buffer.append(tensor)
        new_frame_count += 1

        # Predict
        if len(frame_buffer) == NUM_FRAMES and new_frame_count >= BUFFER_STEP:
            new_frame_count = 0
            with torch.no_grad():
                frames    = torch.stack(list(frame_buffer)).to(device)
                features  = extractor(frames).unsqueeze(0)
                logits    = lstm(features)
                lstm_prob = float(torch.softmax(logits, dim=1)[0, 1])

                f_batch  = tensor.unsqueeze(0).to(device)
                cnn_out  = cnn_clf(f_batch)
                cnn_prob = float(torch.softmax(cnn_out, dim=1)[0, 1])

            lstm_history.append(lstm_prob)
            cnn_history.append(cnn_prob)
            smoothed_lstm = sum(lstm_history) / len(lstm_history)
            smoothed_cnn  = sum(cnn_history)  / len(cnn_history)

            label = "FIGHT" if (
                smoothed_lstm >= LSTM_THRESHOLD and
                smoothed_cnn  >= CNN_THRESHOLD
            ) else "SAFE"

        # Encode frame
        _, buf    = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_b64 = base64.b64encode(buf).decode("utf-8")

        # Update state
        with state_lock:
            was_fight = state["last_fight"]
            is_fight  = label == "FIGHT"

            # New fight started
            if is_fight and not was_fight:
                state["alerts_today"] += 1
                state["alert_log"].insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "msg":  "Fight detected",
                    "type": "fight",
                })
                if len(state["alert_log"]) > MAX_ALERT_LOG:
                    state["alert_log"].pop()

            # Fight ended
            elif not is_fight and was_fight:
                state["alert_log"].insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "msg":  "All clear",
                    "type": "safe",
                })
                if len(state["alert_log"]) > MAX_ALERT_LOG:
                    state["alert_log"].pop()

            # Auto screenshot every SCREENSHOT_INTERVAL seconds during fight
            if is_fight:
                now_ts = time.time()
                if now_ts - last_screenshot_time >= SCREENSHOT_INTERVAL:
                    last_screenshot_time = now_ts
                    ts_str      = datetime.now().strftime("%Y%m%d_%H%M%S")
                    lstm_pct    = round(smoothed_lstm * 100)
                    cnn_pct     = round(smoothed_cnn  * 100)
                    fname       = f"fight_{ts_str}_lstm{lstm_pct}_cnn{cnn_pct}.jpg"
                    ss_path     = os.path.join(SCREENSHOT_DIR, fname)
                    cv2.imwrite(ss_path, frame)
                    print(f"[Screenshot] Saved → {ss_path}")

            state["last_fight"]  = is_fight
            state["label"]       = label
            state["lstm_prob"]   = round(smoothed_lstm * 100, 1)
            state["cnn_prob"]    = round(smoothed_cnn  * 100, 1)
            state["fps"]         = round(fps, 1)

            state["prob_history"].append(round(smoothed_lstm * 100, 1))
            if len(state["prob_history"]) > MAX_PROB_HISTORY:
                state["prob_history"].pop(0)

        # Emit to browser
        socketio.emit("update", {
            "frame":        frame_b64,
            "label":        label,
            "lstm_prob":    round(smoothed_lstm * 100, 1),
            "cnn_prob":     round(smoothed_cnn  * 100, 1),
            "fps":          round(fps, 1),
            "alerts_today": state["alerts_today"],
            "prob_history": state["prob_history"],
            "alert_log":    state["alert_log"][:10],
            "uptime":       int(time.time() - state["start_time"]),
            "new_fight":    is_fight and not was_fight,
        })

    cap.release()
    print("[Camera] Released.")


# ══════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════
@app.route("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.route("/api/state")
def api_state():
    with state_lock:
        return jsonify(state)


# ══════════════════════════════════════════════
#  SOCKETIO EVENTS
# ══════════════════════════════════════════════
@socketio.on("connect")
def on_connect():
    print("[WS] Client connected")

@socketio.on("disconnect")
def on_disconnect():
    print("[WS] Client disconnected")


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    thread = threading.Thread(target=detection_loop, daemon=True)
    thread.start()

    print("\n" + "="*50)
    print("  TRINETRA — Fight Detection Dashboard")
    print("="*50)
    print("  Open in browser: http://localhost:5000")
    print("  Screenshots  ->  " + SCREENSHOT_DIR)
    print("  Press Ctrl+C to stop")
    print("="*50 + "\n")

    socketio.run(app, host="0.0.0.0", port=5000, debug=False)