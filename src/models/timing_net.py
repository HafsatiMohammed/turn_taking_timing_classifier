import torch
import torch.nn as nn
from typing import Dict, Optional
from .branches import FrameBranchTCN, FrameBranchGRU, ScalarBranch, FusionModule


class TimingActionNet(nn.Module):
    """
    Two-branch network for turn-taking timing prediction.

    Architecture:
    - Frame branch: TCN/GRU on [B, seq_len, 7]
    - Scalar branch: MLP on [B, 6]
    - Fusion: MLP on concatenated embeddings → [B, 3]
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Model configuration dict with frame_branch, scalar_branch, fusion
        """
        super().__init__()

        model_cfg = config.get("model", {})
        frame_cfg = model_cfg.get("frame_branch", {})
        scalar_cfg = model_cfg.get("scalar_branch", {})
        fusion_cfg = model_cfg.get("fusion", {})

        # Frame branch
        frame_type = frame_cfg.get("type", "tcn").lower()
        if frame_type == "tcn":
            self.frame_branch = FrameBranchTCN(
                in_channels=frame_cfg.get("in_channels", 7),
                out_channels_list=frame_cfg.get("out_channels", [32, 64, 128]),
                kernel_size=frame_cfg.get("kernel_size", 3),
                dropout=frame_cfg.get("dropout", 0.3),
                output_dim=frame_cfg.get("output_dim", 128),
            )
        elif frame_type == "gru":
            self.frame_branch = FrameBranchGRU(
                in_channels=frame_cfg.get("in_channels", 7),
                hidden_dim=frame_cfg.get("hidden_dim", 128),
                num_layers=frame_cfg.get("num_layers", 2),
                dropout=frame_cfg.get("dropout", 0.3),
                output_dim=frame_cfg.get("output_dim", 128),
                bidirectional=frame_cfg.get("bidirectional", True),
            )
        else:
            raise ValueError(f"Unknown frame branch type: {frame_type}")

        # Scalar branch
        self.scalar_branch = ScalarBranch(
            in_dim=scalar_cfg.get("in_dim", 6),
            hidden_dims=scalar_cfg.get("hidden_dims", [64, 64]),
            output_dim=scalar_cfg.get("output_dim", 64),
            dropout=scalar_cfg.get("dropout", 0.2),
        )

        # Fusion
        frame_dim = frame_cfg.get("output_dim", 128)
        scalar_dim = scalar_cfg.get("output_dim", 64)

        self.fusion = FusionModule(
            in_dim=frame_dim + scalar_dim,
            hidden_dims=fusion_cfg.get("hidden_dims", [256, 128]),
            output_dim=fusion_cfg.get("output_dim", 3),
            dropout=fusion_cfg.get("dropout", 0.3),
        )

        self.config = config

    def forward(
        self,
        frame: torch.Tensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            frame: [B, seq_len, frame_dim]
            scalar: [B, scalar_dim]

        Returns:
            logits: [B, 3]
        """
        # Frame branch
        frame_emb = self.frame_branch(frame)  # [B, 128]

        # Scalar branch
        scalar_emb = self.scalar_branch(scalar)  # [B, 64]

        # Fusion
        fused = torch.cat([frame_emb, scalar_emb], dim=1)  # [B, 192]
        logits = self.fusion(fused)  # [B, 3]

        return logits

    def get_probabilities(self, frame: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities instead of logits."""
        logits = self.forward(frame, scalar)
        return torch.softmax(logits, dim=1)

    def predict(self, frame: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(frame, scalar)
        return torch.argmax(logits, dim=1)
