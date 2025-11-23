import torch
import torch.nn as nn


class FusionAutoEncoder(nn.Module):
    """
    텍스트(KoBERT 임베딩 768) + 수치 피처(예: 3개)
    총 feature_dim = 771

    AutoEncoder 구조:
    Encoder: 771 → 512 → 256 → 128
    Decoder: 128 → 256 → 512 → 771
    """

    def __init__(self, input_dim=771, hidden_dim=128, dropout=0.1):
        super().__init__()

        # --------------------
        # Encoder
        # --------------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, hidden_dim),  # bottleneck
        )

        # --------------------
        # Decoder
        # --------------------
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, input_dim),  # 복원
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z
