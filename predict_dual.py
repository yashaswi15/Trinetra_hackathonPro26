

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
LSTM_MODEL_PATH = r"D:\projects__\hackstreet_26\models\best_model.pth"
CNN_MODEL_PATH  = r"D:\projects__\hackstreet_26\models\cnn_image_model.pth"

NUM_FRAMES      = 16
FRAME_SIZE      = (224, 224)
BUFFER_STEP     = 2       # predict every N new frames
SMOOTH_WINDOW   = 5       # rolling average window

# AND logic thresholds — both must exceed their threshold to say FIGHT
LSTM_THRESHOLD  = 0.90   # LSTM fight probability threshold
CNN_THRESHOLD   = 0.50    # CNN fight probability threshold

WINDOW_NAME     = "Fight Detection — Dual Model"
FONT            = cv2.FONT_HERSHEY_DUPLEX

COLOR_FIGHT   = (0,   0,   255)
COLOR_SAFE    = (0,   200, 80)
COLOR_LOADING = (200, 200, 0)
COLOR_WHITE   = (255, 255, 255)
COLOR_GRAY    = (150, 150, 150)
COLOR_OVERLAY = (20,  20,  20)


# ══════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════
class EfficientNetExtractor(nn.Module):
    """Frozen EfficientNet backbone — used by LSTM pipeline."""
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
        self.features   = None
        self.avgpool    = None
        self.classifier = None
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

# ══════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════
def load_all_models(device):
    print("[Models] Loading EfficientNet feature extractor (for LSTM)...")
    extractor = EfficientNetExtractor(device)

    print("[Models] Loading LSTM model...")
    lstm = FightLSTM(input_dim=1280, hidden_dim=256, num_layers=2).to(device)
    ckpt = torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=False)
    lstm.load_state_dict(ckpt["model_state_dict"])
    lstm.eval()
    print(f"[Models] LSTM loaded — val_acc: {ckpt.get('val_acc', 0)*100:.1f}%")

    print("[Models] Loading CNN image classifier...")
    cnn_clf = CNNImageClassifier().to(device)
    ckpt2   = torch.load(CNN_MODEL_PATH, map_location=device, weights_only=False)
    cnn_clf.load_state_dict(ckpt2["model_state_dict"])
    cnn_clf.eval()
    print(f"[Models] CNN loaded  — val_acc: {ckpt2.get('val_acc', 0)*100:.1f}%")

    return extractor, lstm, cnn_clf


# ══════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess(frame: np.ndarray) -> torch.Tensor:
    """BGR frame → normalized tensor (3, 224, 224)."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, FRAME_SIZE)
    return TRANSFORM(frame)


# ══════════════════════════════════════════════
#  DUAL PREDICTION
# ══════════════════════════════════════════════
@torch.no_grad()
def predict_dual(frame_buffer, current_frame_tensor, extractor, lstm, cnn_clf, device):
    """
    Run both models and apply AND logic.

    Returns:
        final_label  : "FIGHT" or "SAFE"
        lstm_prob    : float, LSTM fight probability
        cnn_prob     : float, CNN fight probability
        both_agree   : bool
    """
    # ── Model 1: LSTM ──
    frames   = torch.stack(frame_buffer).to(device)         # (16, 3, 224, 224)
    features = extractor(frames).unsqueeze(0)               # (1, 16, 1280)
    logits   = lstm(features)                               # (1, 2)
    lstm_prob = float(torch.softmax(logits, dim=1)[0, 1])   # fight prob

    # ── Model 2: CNN single frame ──
    frame_batch = current_frame_tensor.unsqueeze(0).to(device)   # (1, 3, 224, 224)
    cnn_logits  = cnn_clf(frame_batch)                           # (1, 2)
    cnn_prob    = float(torch.softmax(cnn_logits, dim=1)[0, 1])  # fight prob

    # ── AND logic ──
    lstm_says_fight = lstm_prob >= LSTM_THRESHOLD
    cnn_says_fight  = cnn_prob  >= CNN_THRESHOLD
    both_agree      = lstm_says_fight and cnn_says_fight
    final_label     = "FIGHT" if both_agree else "SAFE"

    return final_label, lstm_prob, cnn_prob, both_agree


# ══════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════
def draw_overlay(frame, final_label, lstm_prob, cnn_prob,
                 fps, buffer_fill, smoothed_lstm, smoothed_cnn):
    h, w = frame.shape[:2]

    # ── Top bar ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), COLOR_OVERLAY, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    if buffer_fill < NUM_FRAMES:
        cv2.putText(frame, f"Buffering {buffer_fill}/{NUM_FRAMES}",
                    (15, 50), FONT, 0.9, COLOR_LOADING, 2)
    else:
        color = COLOR_FIGHT if final_label == "FIGHT" else COLOR_SAFE
        cv2.putText(frame, final_label, (15, 52), FONT, 1.4, color, 2)

    # ── FPS ──
    fps_txt = f"FPS {fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_txt, FONT, 0.6, 1)
    cv2.putText(frame, fps_txt, (w - tw - 12, 28), FONT, 0.6, COLOR_WHITE, 1)

    # ── Model indicators (top right) ──
    lstm_color = COLOR_FIGHT if smoothed_lstm >= LSTM_THRESHOLD else COLOR_SAFE
    cnn_color  = COLOR_FIGHT if smoothed_cnn  >= CNN_THRESHOLD  else COLOR_SAFE

    cv2.putText(frame, f"LSTM {smoothed_lstm*100:.0f}%",
                (w - 130, 52), FONT, 0.55, lstm_color, 1)
    cv2.putText(frame, f"CNN  {smoothed_cnn*100:.0f}%",
                (w - 130, 72), FONT, 0.55, cnn_color, 1)

    # ── AND indicator ──
    and_label = "BOTH AGREE" if (smoothed_lstm >= LSTM_THRESHOLD and
                                  smoothed_cnn  >= CNN_THRESHOLD) else "disagreement"
    and_color = COLOR_FIGHT if "BOTH" in and_label else COLOR_GRAY
    cv2.putText(frame, and_label, (15, 78), FONT, 0.5, and_color, 1)

    # ── Bottom confidence bar (LSTM) ──
    bar_h = 16
    bar_y = h - bar_h
    lstm_fill = int(w * smoothed_lstm)
    cv2.rectangle(frame, (0, bar_y), (w, h), COLOR_SAFE, -1)
    if lstm_fill > 0:
        cv2.rectangle(frame, (0, bar_y), (lstm_fill, h), COLOR_FIGHT, -1)
    threshold_x = int(w * LSTM_THRESHOLD)
    cv2.line(frame, (threshold_x, bar_y), (threshold_x, h), COLOR_WHITE, 2)
    cv2.putText(frame, "LSTM", (6, h - 3), FONT, 0.38, COLOR_WHITE, 1)

    # ── Alert border when both agree ──
    if final_label == "FIGHT":
        cv2.rectangle(frame, (2, 2), (w-2, h-2), COLOR_FIGHT, 3)

    return frame


# ══════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════
def run(source: str, video_path: str = None, camera_index: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    extractor, lstm, cnn_clf = load_all_models(device)

    # Open source
    cap = cv2.VideoCapture(camera_index if source == "webcam" else video_path)
    if source == "webcam":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    frame_buffer    = collections.deque(maxlen=NUM_FRAMES)
    new_frame_count = 0

    lstm_history = collections.deque(maxlen=SMOOTH_WINDOW)
    cnn_history  = collections.deque(maxlen=SMOOTH_WINDOW)

    final_label   = "SAFE"
    smoothed_lstm = 0.0
    smoothed_cnn  = 0.0

    fps_history = collections.deque(maxlen=30)
    prev_time   = time.time()

    print("\n[Running] Press Q or ESC to quit | S to screenshot\n")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Done] End of stream.")
            break

        # FPS
        now = time.time()
        fps_history.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = sum(fps_history) / len(fps_history)

        # Preprocess
        tensor = preprocess(frame)
        frame_buffer.append(tensor)
        new_frame_count += 1

        # Predict every BUFFER_STEP frames once buffer is full
        if len(frame_buffer) == NUM_FRAMES and new_frame_count >= BUFFER_STEP:
            new_frame_count = 0
            final_label, lstm_prob, cnn_prob, _ = predict_dual(
                list(frame_buffer), tensor,
                extractor, lstm, cnn_clf, device
            )
            lstm_history.append(lstm_prob)
            cnn_history.append(cnn_prob)
            smoothed_lstm = sum(lstm_history) / len(lstm_history)
            smoothed_cnn  = sum(cnn_history)  / len(cnn_history)

            # Re-apply AND logic on smoothed values
            if smoothed_lstm >= LSTM_THRESHOLD and smoothed_cnn >= CNN_THRESHOLD:
                final_label = "FIGHT"
            else:
                final_label = "SAFE"

        # Draw
        display = draw_overlay(
            frame.copy(), final_label,
            lstm_prob=smoothed_lstm, cnn_prob=smoothed_cnn,
            fps=fps, buffer_fill=len(frame_buffer),
            smoothed_lstm=smoothed_lstm, smoothed_cnn=smoothed_cnn,
        )

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]:
            print("[Quit]")
            break
        elif key == ord("s"):
            path = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(path, display)
            print(f"[Screenshot] {path}")

    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["webcam", "video"], default="webcam")
    parser.add_argument("--path",   type=str, default=None)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if args.source == "video" and args.path is None:
        parser.error("--path required when --source is video")

    run(args.source, args.path, args.camera)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
predict_dual.py
---------------
Dual-Model Fight Detection System.

Model 1: CNN (EfficientNet) + LSTM  → detects motion-based fight patterns
Model 2: CNN image classifier        → detects appearance-based fight patterns

Final decision: AND logic
    → FIGHT only when BOTH models agree it's a fight
    → This eliminates false positives like dancing

Usage:
    python work/predict_dual.py --source webcam
    python work/predict_dual.py --source video --path "D:/path/to/video.mp4"
"""