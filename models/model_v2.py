# models/model_v2.py
import torch
from torch import nn


class FusionAutoEncoderV2(nn.Module):
    """
    v2 AutoEncoder
      - 입력: numeric(5) + KoBERT CLS 임베딩(768) = 773차원
      - 구조: 773 -> 512 -> hidden_dim -> latent_dim -> hidden_dim -> 512 -> 773
      - train_v2.py 에서 MSELoss(recon, features)로 학습
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # latent bottleneck
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, input_dim)
        return:
          recon: (batch_size, input_dim)
          extra: dict with 'z' (latent)
        """
        h = self.encoder(x)          # (B, hidden_dim)
        z = self.to_latent(h)        # (B, latent_dim)
        recon = self.decoder(z)      # (B, input_dim)
        return recon, {"z": z}
