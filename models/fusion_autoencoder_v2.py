# models/fusion_autoencoder_v2.py
import torch
from torch import nn


class FusionAutoEncoderV2(nn.Module):
    """
    v2용 AutoEncoder
    - 입력: 숫자 피처 + KoBERT 임베딩 (총 input_dim 차원)
    - encoder: input_dim -> hidden_dim -> bottleneck_dim
    - decoder: bottleneck_dim -> hidden_dim -> input_dim
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # --------- Encoder ---------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
        )

        # --------- Decoder ---------
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, input_dim),
            # 출력은 그대로 회귀이므로 activation 없음
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, input_dim)
        return:
          recon: (batch_size, input_dim)
          z:     (batch_size, bottleneck_dim)
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
