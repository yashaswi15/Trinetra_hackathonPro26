
import torch
import torch.nn as nn


class FightLSTM(nn.Module):
    """
    Bidirectional LSTM classifier for video fight detection.

    Args:
        input_dim   : Feature dimension from CNN (1280 for EfficientNet-B0)
        hidden_dim  : LSTM hidden units (256 recommended)
        num_layers  : Stacked LSTM layers (2 recommended)
        num_classes : Output classes (2: fight / non_fight)
        dropout     : Dropout probability
    """

    def __init__(
        self,
        input_dim:   int = 1280,
        hidden_dim:  int = 256,
        num_layers:  int = 2,
        num_classes: int = 2,
        dropout:     float = 0.5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ── 1. Input Projection ──────────────────────────────
        # Compress 1280 → 512 before feeding to LSTM
        # Reduces parameters, prevents overfitting
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),   # lighter dropout here
        )

        # ── 2. Bidirectional LSTM ────────────────────────────
        # bidirectional=True → sees sequence forward AND backward
        # Output dim = hidden_dim * 2 (because bi-directional)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # input: (batch, seq, feature)
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── 3. Classifier Head ───────────────────────────────
        lstm_output_dim = hidden_dim * 2   # 256 * 2 = 512 (bidirectional)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, input_dim)  e.g. (32, 16, 1280)
        Returns:
            logits : (batch, num_classes)    e.g. (32, 2)
        """
        # Project input features
        x = self.input_proj(x)                 # (B, 16, 512)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)             # (B, 16, 512)

        # Use ONLY the last timestep output for classification
        # This captures the final "decision" after seeing all 16 frames
        last_out = lstm_out[:, -1, :]          # (B, 512)

        # Classify
        logits = self.classifier(last_out)     # (B, 2)
        return logits


def get_model_summary(model: nn.Module, device: torch.device):
    """Prints parameter count and a quick forward pass shape check."""
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Quick shape test
    dummy = torch.zeros(2, 16, 1280).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Input  shape: (2, 16, 1280)")
    print(f"  Output shape: {tuple(out.shape)}  ← (batch, 2 classes)")
    
    


"""
lstm_model.py
-------------
LSTM-based Fight Detection Model.

Input  : Pre-extracted CNN features → shape (batch, 16, 1280)
Output : Binary classification → fight (1) or non_fight (0)

Architecture:
    Linear(1280 → 512)    ← project down features
    LSTM(512, hidden=256, layers=2, bidirectional=True)
    Dropout(0.5)
    Linear(512 → 128)     ← 512 because bidirectional doubles hidden size
    ReLU
    Dropout(0.3)
    Linear(128 → 2)       ← fight / non_fight logits
"""
