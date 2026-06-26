import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple
import json


class TimingDataset(Dataset):
    """Dataset for turn-taking timing prediction with frame and scalar features."""

    LABEL_TO_IDX = {
        "WAIT": 0,
        "BACKCHANNEL": 1,
        "START_SPEAKING": 2,
    }

    IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

    def __init__(
        self,
        parquet_path: str,
        split: str = "train",
        frame_seq_len: int = 120,
        frame_dim: int = 7,
        scalar_dim: int = 6,
        normalize: bool = True,
        exclude_low_confidence: bool = False,
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            parquet_path: Path to parquet file
            split: Dataset split name (train/val/test)
            frame_seq_len: Length of frame sequence (120 for 6s)
            frame_dim: Number of frame features (7)
            scalar_dim: Number of scalar features (6)
            normalize: Whether to normalize features
            exclude_low_confidence: Filter samples with low training_weight
            confidence_threshold: Threshold for filtering
        """
        self.split = split
        self.frame_seq_len = frame_seq_len
        self.frame_dim = frame_dim
        self.scalar_dim = scalar_dim
        self.normalize = normalize

        # Load parquet
        print(f"Loading {split} dataset from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)

        print(f"Original {split} size: {len(self.df)}")

        # Filter excluded samples
        if "exclude_from_training" in self.df.columns:
            mask = ~self.df["exclude_from_training"].astype(bool)
            self.df = self.df[mask]
            print(f"After filtering excluded: {len(self.df)}")

        # Filter by confidence
        if exclude_low_confidence and "training_weight" in self.df.columns:
            mask = self.df["training_weight"] >= confidence_threshold
            self.df = self.df[mask]
            print(f"After confidence filtering: {len(self.df)}")

        self.df = self.df.reset_index(drop=True)

        # Extract labels
        self.labels = self.df["final_label"].map(self.LABEL_TO_IDX).values

        # Get training weights
        if "training_weight" in self.df.columns:
            self.weights = self.df["training_weight"].values
        else:
            self.weights = np.ones(len(self.df))

        print(f"Label distribution: {np.bincount(self.labels)}")
        print(f"Mean weight: {self.weights.mean():.4f}")

        # Compute normalization stats if needed
        if normalize:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute mean/std for normalization from training data."""
        print("Computing normalization statistics...")

        all_frames = []
        all_scalars = []

        for idx in range(len(self.df)):
            frame, scalar = self._get_raw_features(idx)
            all_frames.append(frame)
            all_scalars.append(scalar)

        all_frames = np.array(all_frames)  # [N, seq_len, 7]
        all_scalars = np.array(all_scalars)  # [N, 6]

        # Frame stats: mean/std across all time steps
        self.frame_mean = all_frames.mean(axis=(0, 1), keepdims=True)
        self.frame_std = all_frames.std(axis=(0, 1), keepdims=True) + 1e-8

        # Scalar stats
        self.scalar_mean = all_scalars.mean(axis=0, keepdims=True)
        self.scalar_std = all_scalars.std(axis=0, keepdims=True) + 1e-8

        print(f"Frame stats: mean shape {self.frame_mean.shape}, std shape {self.frame_std.shape}")
        print(f"Scalar stats: mean shape {self.scalar_mean.shape}, std shape {self.scalar_std.shape}")

    def _get_raw_features(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract raw frame and scalar features from parquet columns."""
        row = self.df.iloc[idx]

        # Frame features - try different column names
        frame_col = None
        for col in ["X_frame", "frame_features", "frame"]:
            if col in row.index:
                frame_col = col
                break

        if frame_col is None:
            raise ValueError(f"No frame features column found. Available: {row.index.tolist()}")

        # Scalar features
        scalar_col = None
        for col in ["X_scalar", "scalar_features", "scalar"]:
            if col in row.index:
                scalar_col = col
                break

        if scalar_col is None:
            raise ValueError(f"No scalar features column found. Available: {row.index.tolist()}")

        # Get features (handle numpy arrays and lists)
        frame = row[frame_col]
        if isinstance(frame, str):
            frame = json.loads(frame)
        if isinstance(frame, list):
            frame = np.array(frame, dtype=np.float32)
        elif not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=np.float32)
        else:
            frame = frame.astype(np.float32)

        scalar = row[scalar_col]
        if isinstance(scalar, str):
            scalar = json.loads(scalar)
        if isinstance(scalar, list):
            scalar = np.array(scalar, dtype=np.float32)
        elif not isinstance(scalar, np.ndarray):
            scalar = np.array(scalar, dtype=np.float32)
        else:
            scalar = scalar.astype(np.float32)

        return frame, scalar

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                "frame": torch.Tensor [seq_len, frame_dim],
                "scalar": torch.Tensor [scalar_dim],
                "label": torch.Tensor (scalar),
                "weight": torch.Tensor (scalar),
                "sample_id": str,
            }
        """
        frame, scalar = self._get_raw_features(idx)

        # Ensure correct shapes
        assert frame.shape == (self.frame_seq_len, self.frame_dim), \
            f"Frame shape mismatch: {frame.shape} vs {(self.frame_seq_len, self.frame_dim)}"
        assert scalar.shape == (self.scalar_dim,), \
            f"Scalar shape mismatch: {scalar.shape} vs {(self.scalar_dim,)}"

        # Normalize
        if self.normalize:
            frame = (frame - self.frame_mean) / self.frame_std
            scalar = (scalar - self.scalar_mean) / self.scalar_std

        label = self.labels[idx]
        weight = self.weights[idx]
        sample_id = self.df.iloc[idx].get("sample_id", f"sample_{idx}")

        return {
            "frame": torch.FloatTensor(frame),
            "scalar": torch.FloatTensor(scalar),
            "label": torch.LongTensor([label]).squeeze(),
            "weight": torch.FloatTensor([weight]).squeeze(),
            "sample_id": sample_id,
        }

    def get_class_weights(self) -> np.ndarray:
        """Compute class weights for balanced training."""
        counts = np.bincount(self.labels)
        # Weight inversely proportional to frequency
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * len(counts)
        return weights

    def get_label_distribution(self) -> Dict[str, float]:
        """Return label distribution as percentages."""
        counts = np.bincount(self.labels)
        total = len(self.labels)
        return {
            self.IDX_TO_LABEL[i]: float(counts[i]) / total * 100
            for i in range(len(counts))
        }
