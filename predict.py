"""
predict.py
----------
Real-Time Fight Detection System.

Supports:
    - Live webcam feed
    - Pre-recorded video files

Pipeline per frame window:
    Webcam/Video → 16 frame buffer → EfficientNet-B0 → LSTM → FIGHT / SAFE label

Usage:
    # Webcam (default camera)
    python work/predict.py --source webcam

    # Video file
    python work/predict.py --source video --path "D:/path/to/video.mp4"

    # Webcam with custom camera index
    python work/predict.py --source webcam --camera 1
"""

import cv2
import time
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path

from lstm_model import FightLSTM


# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
MODEL_PATH   = r"D:\projects__\hackstreet_26\models\best_model.pth"
NUM_FRAMES   = 16        # must match training
FRAME_SIZE   = (224, 224)
BUFFER_STEP  = 2         # predict every N new frames (lower = smoother but slower)
CONF_SMOOTH  = 5         # rolling average over last N predictions

# Display
WINDOW_NAME  = "Fight Detection System"
FONT         = cv2.FONT_HERSHEY_DUPLEX

# Colors (BGR)
COLOR_FIGHT    = (0,   0,   255)   # Red
COLOR_SAFE     = (0,   200, 80)    # Green
COLOR_LOADING  = (200, 200, 0)     # Yellow
COLOR_WHITE    = (255, 255, 255)
COLOR_BLACK    = (0,   0,   0)
COLOR_OVERLAY  = (20,  20,  20)


# ══════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════
class EfficientNetExtractor(nn.Module):
    """Frozen EfficientNet-B0 — identical to feature_extractor.py."""

    def __init__(self, device):
        super().__init__()
        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.features = efficientnet.features
        self.pool     = efficientnet.avgpool
        self.to(device)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)   # (B, 1280)


def load_models(model_path: str, device: torch.device):
    """Load EfficientNet extractor + trained LSTM."""

    print("[Model] Loading EfficientNet-B0...")
    cnn = EfficientNetExtractor(device)

    print("[Model] Loading trained FightLSTM...")
    lstm = FightLSTM(input_dim=1280, hidden_dim=256, num_layers=2).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm.eval()

    val_acc = checkpoint.get("val_acc", 0) * 100
    print(f"[Model] LSTM loaded — Val accuracy: {val_acc:.2f}%")

    return cnn, lstm


# ══════════════════════════════════════════════
#  FRAME PREPROCESSING
# ══════════════════════════════════════════════
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
    ),
])

def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """BGR frame → normalized tensor (3, 224, 224)."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, FRAME_SIZE)
    return TRANSFORM(frame)


# ══════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════
@torch.no_grad()
def predict(frame_buffer, cnn, lstm, device):
    """
    Args:
        frame_buffer : list of NUM_FRAMES tensors, each (3, 224, 224)
    Returns:
        label      : "FIGHT" or "SAFE"
        confidence : float 0.0–1.0
        probs      : (2,) numpy array [prob_nonfight, prob_fight]
    """
    # Stack frames → (16, 3, 224, 224)
    frames = torch.stack(frame_buffer).to(device)         # (16, 3, 224, 224)

    # CNN extract → (16, 1280)
    features = cnn(frames)                                 # (16, 1280)

    # Add batch dim → (1, 16, 1280)
    features = features.unsqueeze(0)

    # LSTM predict → (1, 2)
    logits = lstm(features)
    probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # probs[0] = non_fight, probs[1] = fight
    fight_prob = float(probs[1])
    label      = "FIGHT" if fight_prob >= 0.5 else "SAFE"
    confidence = fight_prob if label == "FIGHT" else float(probs[0])

    return label, confidence, probs


# ══════════════════════════════════════════════
#  DISPLAY OVERLAY
# ══════════════════════════════════════════════
def draw_overlay(
    frame: np.ndarray,
    label: str,
    confidence: float,
    fps: float,
    buffer_fill: int,
    smoothed_fight_prob: float,
) -> np.ndarray:
    """Draws the HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # ── Dark top bar ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), COLOR_OVERLAY, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # ── Label + Confidence ──
    color = COLOR_FIGHT if label == "FIGHT" else COLOR_SAFE

    if buffer_fill < NUM_FRAMES:
        # Still filling buffer
        status_text = f"Buffering... {buffer_fill}/{NUM_FRAMES}"
        cv2.putText(frame, status_text, (15, 45), FONT, 0.9, COLOR_LOADING, 2)
    else:
        label_text = f"{label}  {confidence * 100:.1f}%"
        cv2.putText(frame, label_text, (15, 48), FONT, 1.2, color, 2)

    # ── FPS (top right) ──
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, FONT, 0.65, 1)
    cv2.putText(frame, fps_text, (w - tw - 15, 30), FONT, 0.65, COLOR_WHITE, 1)

    # ── Confidence bar (bottom of frame) ──
    bar_h     = 18
    bar_y     = h - bar_h
    bar_width = int(w * smoothed_fight_prob)

    # Background bar (green = safe side)
    cv2.rectangle(frame, (0, bar_y), (w, h), COLOR_SAFE, -1)
    # Fight probability fill (red)
    if bar_width > 0:
        cv2.rectangle(frame, (0, bar_y), (bar_width, h), COLOR_FIGHT, -1)

    # Center threshold line
    cv2.line(frame, (w // 2, bar_y), (w // 2, h), COLOR_WHITE, 2)

    # Bar labels
    cv2.putText(frame, "SAFE", (8, h - 3), FONT, 0.45, COLOR_WHITE, 1)
    cv2.putText(frame, "FIGHT", (w - 55, h - 3), FONT, 0.45, COLOR_WHITE, 1)

    # ── Alert flash when FIGHT detected ──
    if label == "FIGHT" and confidence > 0.75:
        alert_overlay = frame.copy()
        cv2.rectangle(alert_overlay, (0, 0), (w, h), COLOR_FIGHT, -1)
        cv2.addWeighted(alert_overlay, 0.08, frame, 0.92, 0, frame)

        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), COLOR_FIGHT, 3)

    return frame


# ══════════════════════════════════════════════
#  MAIN PREDICTION LOOP
# ══════════════════════════════════════════════
def run_prediction(source: str, video_path: str = None, camera_index: int = 0):

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # ── Load Models ──
    cnn, lstm = load_models(MODEL_PATH, device)

    # ── Open Video Source ──
    if source == "webcam":
        print(f"\n[Source] Webcam (index {camera_index})")
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        print(f"\n[Source] Video file: {video_path}")
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        return

    # ── State ──
    frame_buffer    = collections.deque(maxlen=NUM_FRAMES)   # raw preprocessed tensors
    new_frame_count = 0

    # Smoothing: rolling average of fight probability
    prob_history    = collections.deque(maxlen=CONF_SMOOTH)

    current_label   = "SAFE"
    current_conf    = 0.0
    smoothed_prob   = 0.0

    fps_counter     = collections.deque(maxlen=30)
    prev_time       = time.time()

    print(f"\n[Running] Press Q to quit\n")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    while True:
        ret, frame = cap.read()

        if not ret:
            if source == "video":
                print("[Done] End of video.")
            else:
                print("[ERROR] Frame read failed.")
            break

        # ── FPS calculation ──
        now = time.time()
        fps_counter.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = sum(fps_counter) / len(fps_counter)

        # ── Preprocess and buffer ──
        tensor = preprocess_frame(frame)
        frame_buffer.append(tensor)
        new_frame_count += 1

        # ── Predict when buffer is full and every BUFFER_STEP frames ──
        if len(frame_buffer) == NUM_FRAMES and new_frame_count >= BUFFER_STEP:
            new_frame_count = 0

            label, conf, probs = predict(list(frame_buffer), cnn, lstm, device)

            # Smooth the fight probability
            prob_history.append(float(probs[1]))
            smoothed_prob = sum(prob_history) / len(prob_history)

            current_label = "FIGHT" if smoothed_prob >= 0.5 else "SAFE"
            current_conf  = smoothed_prob if current_label == "FIGHT" else (1 - smoothed_prob)

        # ── Draw HUD ──
        display_frame = draw_overlay(
            frame.copy(),
            label        = current_label,
            confidence   = current_conf,
            fps          = fps,
            buffer_fill  = len(frame_buffer),
            smoothed_fight_prob = smoothed_prob,
        )

        cv2.imshow(WINDOW_NAME, display_frame)

        # ── Keyboard controls ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:   # Q or ESC
            print("[Quit] Exiting.")
            break
        elif key == ord("s"):
            # Screenshot
            ts = int(time.time())
            save_path = f"screenshot_{ts}.jpg"
            cv2.imwrite(save_path, display_frame)
            print(f"[Screenshot] Saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fight Detection - Real-Time Prediction")

    parser.add_argument(
        "--source",
        type=str,
        choices=["webcam", "video"],
        default="webcam",
        help="Input source: 'webcam' or 'video'",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to video file (required when --source video)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 0)",
    )

    args = parser.parse_args()

    if args.source == "video" and args.path is None:
        parser.error("--path is required when --source is 'video'")

    run_prediction(
        source       = args.source,
        video_path   = args.path,
        camera_index = args.camera,
    )