#!/usr/bin/env python3
"""
Prepare dataset: Extract X_frame and X_scalar from HuggingFace AMI dataset and metadata.

This script:
1. Loads HuggingFace AMI dataset (with VAD and speaker information)
2. Reads metadata JSONL with sample timing information
3. Extracts X_frame [120, 7] and X_scalar [6] for each sample
4. Saves as parquet files (train/validation/test)

Usage:
    python scripts/prepare_dataset.py \
        --metadata data/processed/final_manifest.jsonl \
        --output data/processed/hf_export
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


class FeatureExtractor:
    """Extract X_frame and X_scalar features from HuggingFace AMI dataset."""

    # Frame feature indices
    FRAME_FEATURES = [
        "any_human_active",           # [0]
        "num_humans_active_norm",     # [1]
        "overlap_active",             # [2]
        "silence_active",             # [3]
        "human_onset",                # [4]
        "human_offset",               # [5]
        "pseudo_robot_past_active",   # [6]
    ]

    # Scalar feature indices
    SCALAR_FEATURES = [
        "silence_duration_before_t",           # [0]
        "current_human_speech_duration",       # [1]
        "human_speech_ratio_last_1s",          # [2]
        "human_speech_ratio_last_6s",          # [3]
        "overlap_ratio_last_6s",               # [4]
        "time_since_pseudo_robot_last_spoke",  # [5]
    ]

    def __init__(self, frame_shift: float = 0.05):
        """
        Args:
            frame_shift: Frame shift in seconds (0.05s = 20Hz)
        """
        self.frame_shift = frame_shift
        self.frames_per_second = 1.0 / frame_shift

        # Load HuggingFace AMI dataset
        self.ami_dataset = self._load_ami_dataset()
        self.ami_by_id = self._index_ami_dataset()

    def _load_ami_dataset(self):
        """Load HuggingFace AMI dataset."""
        print("Loading HuggingFace AMI dataset...")
        try:
            from datasets import load_dataset
            ami = load_dataset("edinburghcstr/ami")
            print(f"Loaded AMI with splits: {list(ami.keys())}")
            return ami
        except ImportError:
            print("ERROR: datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            raise

    def _index_ami_dataset(self) -> Dict:
        """Index AMI dataset by meeting_id for fast lookup."""
        print("Indexing AMI dataset by meeting_id...")
        ami_by_id = {}

        for split in ["train", "validation", "test"]:
            if split not in self.ami_dataset:
                continue

            for sample in self.ami_dataset[split]:
                meeting_id = sample.get("meeting_id")
                if meeting_id not in ami_by_id:
                    ami_by_id[meeting_id] = {}

                # Store sample by meeting and microphone type (IHM = individual headset)
                mic_id = sample.get("microphone_id", 0)
                if mic_id not in ami_by_id[meeting_id]:
                    ami_by_id[meeting_id][mic_id] = sample

        print(f"Indexed {len(ami_by_id)} unique meetings")
        return ami_by_id

    def extract_features(self, sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract X_frame and X_scalar for a single sample.

        Args:
            sample: Sample dict with timing and metadata

        Returns:
            (X_frame [120, 7], X_scalar [6])
        """
        meeting_id = sample["meeting_id"]
        pseudo_robot = sample["pseudo_robot"]
        context_start = sample["context_start"]
        context_end = sample["context_end"]
        time_t = sample["time"]

        # Get AMI sample for this meeting
        ami_sample = self._get_ami_sample(meeting_id)
        if ami_sample is None:
            raise ValueError(f"No AMI data found for meeting {meeting_id}")

        # Extract X_frame [120, 7]
        x_frame = self._extract_frame_features(
            ami_sample, context_start, context_end, pseudo_robot
        )

        # Extract X_scalar [6]
        x_scalar = self._extract_scalar_features(sample, ami_sample, context_start, context_end, time_t, pseudo_robot)

        return x_frame, x_scalar

    def _get_ami_sample(self, meeting_id: str) -> Optional[Dict]:
        """Get AMI sample for meeting_id."""
        if meeting_id not in self.ami_by_id:
            return None
        # Get first microphone (IHM)
        mic_samples = self.ami_by_id[meeting_id]
        return list(mic_samples.values())[0] if mic_samples else None

    def _extract_frame_features(
        self,
        ami_sample: Dict,
        context_start: float,
        context_end: float,
        pseudo_robot: str,
    ) -> np.ndarray:
        """
        Extract 120 frames × 7 features from AMI dataset.

        Frame sequence covers 6 seconds with 0.05s shift (120 frames total).

        Args:
            ami_sample: AMI dataset sample with recording data
            context_start: Start time (seconds)
            context_end: End time (seconds)
            pseudo_robot: Robot speaker ID (A, B, C, or D)

        Returns:
            X_frame [120, 7] with features:
            [0] any_human_active
            [1] num_humans_active_norm
            [2] overlap_active
            [3] silence_active
            [4] human_onset
            [5] human_offset
            [6] pseudo_robot_past_active
        """
        # Extract VAD and segmentation from AMI sample
        recording = ami_sample.get("recording", {})

        # Get speaker activity (segmentation data)
        # AMI provides speaker segments with timing
        speaker_segmentation = self._get_speaker_segments(ami_sample)

        # Frame times
        frame_times = np.arange(context_start, context_end, self.frame_shift)
        num_frames = len(frame_times)

        if num_frames != 120:
            print(f"Warning: Expected 120 frames, got {num_frames}")

        x_frame = np.zeros((num_frames, 7), dtype=np.float32)

        # Get all speakers except pseudo_robot
        all_speakers = ["A", "B", "C", "D"]
        human_speakers = [s for s in all_speakers if s != pseudo_robot]

        for frame_idx, frame_time in enumerate(frame_times):
            # Get active speakers at this frame time
            active_speakers = self._get_active_speakers_at_time(speaker_segmentation, frame_time)
            human_active_speakers = [s for s in active_speakers if s in human_speakers]

            # [0] any_human_active: Is any human speaking?
            x_frame[frame_idx, 0] = 1.0 if len(human_active_speakers) > 0 else 0.0

            # [1] num_humans_active_norm: Normalized count of active humans
            x_frame[frame_idx, 1] = len(human_active_speakers) / len(human_speakers)

            # [2] overlap_active: Are multiple humans overlapping?
            x_frame[frame_idx, 2] = 1.0 if len(human_active_speakers) > 1 else 0.0

            # [3] silence_active: Is it silent (no one speaking)?
            x_frame[frame_idx, 3] = 1.0 if len(active_speakers) == 0 else 0.0

            # [4] human_onset: Did a human just start speaking?
            if frame_idx == 0:
                x_frame[frame_idx, 4] = 0.0
            else:
                prev_active = self._get_active_speakers_at_time(speaker_segmentation, frame_times[frame_idx - 1])
                prev_human_active = [s for s in prev_active if s in human_speakers]
                x_frame[frame_idx, 4] = 1.0 if (
                    len(human_active_speakers) > len(prev_human_active)
                ) else 0.0

            # [5] human_offset: Did a human just stop speaking?
            if frame_idx == 0:
                x_frame[frame_idx, 5] = 0.0
            else:
                prev_active = self._get_active_speakers_at_time(speaker_segmentation, frame_times[frame_idx - 1])
                prev_human_active = [s for s in prev_active if s in human_speakers]
                x_frame[frame_idx, 5] = 1.0 if (
                    len(human_active_speakers) < len(prev_human_active)
                ) else 0.0

            # [6] pseudo_robot_past_active: Was robot speaking?
            robot_active = pseudo_robot in active_speakers
            x_frame[frame_idx, 6] = 1.0 if robot_active else 0.0

        return x_frame

    def _get_speaker_segments(self, ami_sample: Dict) -> Dict[str, List[Tuple]]:
        """
        Extract speaker segmentation (timing) from AMI sample.

        Returns dict mapping speaker ID to list of (start, end) tuples.
        """
        segments = {s: [] for s in ["A", "B", "C", "D"]}

        # AMI provides begin_time, end_time, and speaker fields
        if "segments" in ami_sample:
            for segment in ami_sample["segments"]:
                speaker = segment.get("speaker")
                begin_time = segment.get("begin_time", 0)
                end_time = segment.get("end_time", 0)

                if speaker and speaker in segments:
                    segments[speaker].append((begin_time, end_time))

        return segments

    def _get_active_speakers_at_time(self, segments: Dict[str, List[Tuple]], time_t: float) -> List[str]:
        """Get list of speakers active at time t."""
        active = []
        for speaker, intervals in segments.items():
            for start, end in intervals:
                if start <= time_t <= end:
                    active.append(speaker)
                    break
        return active

    def _extract_scalar_features(
        self,
        sample: Dict,
        ami_sample: Dict,
        context_start: float,
        context_end: float,
        time_t: float,
        pseudo_robot: str,
    ) -> np.ndarray:
        """
        Extract 6 scalar features at prediction time.

        Args:
            sample: Sample dict with metadata
            ami_sample: AMI dataset sample
            context_start: Context window start (seconds)
            context_end: Context window end (seconds)
            time_t: Prediction time (seconds)
            pseudo_robot: Robot speaker ID

        Returns:
            X_scalar [6] with features:
            [0] silence_duration_before_t
            [1] current_human_speech_duration
            [2] human_speech_ratio_last_1s
            [3] human_speech_ratio_last_6s
            [4] overlap_ratio_last_6s
            [5] time_since_pseudo_robot_last_spoke
        """
        x_scalar = np.zeros(6, dtype=np.float32)

        # Get state from metadata (if precomputed)
        state = sample.get("state_at_prediction_time", {})

        # If state has these values precomputed, use them
        if state:
            x_scalar[0] = state.get("time_since_last_human", state.get("silence_duration_before_t", 0.0))
            x_scalar[1] = state.get("current_human_speech_duration", 0.0)
            x_scalar[2] = state.get("human_speech_ratio_last_1s", 0.0)
            x_scalar[3] = state.get("human_speech_ratio_last_6s", 0.0)
            x_scalar[4] = state.get("overlap_ratio_last_6s", 0.0)
            x_scalar[5] = state.get("time_since_pseudo_robot_last_spoke", 0.0)
        else:
            # Compute from AMI data if not precomputed
            segments = self._get_speaker_segments(ami_sample)
            all_speakers = ["A", "B", "C", "D"]
            human_speakers = [s for s in all_speakers if s != pseudo_robot]

            # [0] silence_duration_before_t: Time since last human spoke
            silence_duration = self._compute_silence_duration(segments, human_speakers, time_t)
            x_scalar[0] = min(silence_duration, 10.0)  # Cap at 10 seconds

            # [1] current_human_speech_duration
            speech_duration = self._compute_speech_duration(segments, human_speakers, time_t)
            x_scalar[1] = min(speech_duration, 10.0)  # Cap at 10 seconds

            # [2] human_speech_ratio_last_1s
            x_scalar[2] = self._compute_speech_ratio(segments, human_speakers, time_t - 1.0, time_t)

            # [3] human_speech_ratio_last_6s
            x_scalar[3] = self._compute_speech_ratio(segments, human_speakers, time_t - 6.0, time_t)

            # [4] overlap_ratio_last_6s
            x_scalar[4] = self._compute_overlap_ratio(segments, time_t - 6.0, time_t)

            # [5] time_since_pseudo_robot_last_spoke
            robot_silence = self._compute_silence_duration(segments, [pseudo_robot], time_t)
            x_scalar[5] = min(robot_silence, 10.0)

        return x_scalar

    def _compute_silence_duration(self, segments: Dict[str, List[Tuple]], speakers: List[str], time_t: float) -> float:
        """Compute how long since any of the speakers was active."""
        last_end = 0.0

        for speaker in speakers:
            if speaker in segments:
                for start, end in segments[speaker]:
                    if end <= time_t:
                        last_end = max(last_end, end)

        return time_t - last_end

    def _compute_speech_duration(self, segments: Dict[str, List[Tuple]], speakers: List[str], time_t: float) -> float:
        """Compute continuous speech duration up to time_t."""
        # Find the most recent speech onset
        most_recent_start = 0.0

        for speaker in speakers:
            if speaker in segments:
                for start, end in segments[speaker]:
                    if end <= time_t and start <= time_t:
                        most_recent_start = max(most_recent_start, start)
                    elif start < time_t < end:
                        most_recent_start = max(most_recent_start, start)

        return time_t - most_recent_start if most_recent_start > 0 else 0.0

    def _compute_speech_ratio(self, segments: Dict[str, List[Tuple]], speakers: List[str], start: float, end: float) -> float:
        """Compute % of time interval with any speaker active."""
        if start >= end:
            return 0.0

        interval_length = end - start
        active_time = 0.0

        # For each speaker, sum active intervals
        for speaker in speakers:
            if speaker in segments:
                for seg_start, seg_end in segments[speaker]:
                    overlap_start = max(start, seg_start)
                    overlap_end = min(end, seg_end)
                    if overlap_start < overlap_end:
                        active_time += overlap_end - overlap_start

        # Avoid double counting overlaps (simplified: just cap at interval length)
        return min(active_time / interval_length, 1.0)

    def _compute_overlap_ratio(self, segments: Dict[str, List[Tuple]], start: float, end: float) -> float:
        """Compute % of time with multiple speakers overlapping."""
        if start >= end:
            return 0.0

        interval_length = end - start
        overlap_time = 0.0

        # Find all time points with 2+ speakers
        events = []
        for speaker, intervals in segments.items():
            for seg_start, seg_end in intervals:
                events.append((seg_start, 1, speaker))  # Speaker starts
                events.append((seg_end, -1, speaker))   # Speaker ends

        events.sort()

        active_speakers = set()
        prev_time = start

        for event_time, event_type, speaker in events:
            if event_time > end:
                break

            if prev_time >= start and event_time <= end and len(active_speakers) > 1:
                overlap_time += min(event_time, end) - max(prev_time, start)

            if event_time >= start:
                if event_type == 1:
                    active_speakers.add(speaker)
                else:
                    active_speakers.discard(speaker)
                prev_time = event_time

        return overlap_time / interval_length if interval_length > 0 else 0.0


def main(args):
    # Load metadata
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found: {args.metadata}")
        return

    print(f"Loading metadata from {args.metadata}...")
    samples = []
    with open(metadata_path) as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(samples)} samples")

    # Initialize feature extractor (loads HuggingFace AMI dataset)
    try:
        extractor = FeatureExtractor()
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Please install: pip install datasets")
        return
    except Exception as e:
        print(f"ERROR loading AMI dataset: {e}")
        return

    # Group by split
    samples_by_split = {}
    for sample in samples:
        split = sample.get("split", "train")
        if split not in samples_by_split:
            samples_by_split[split] = []
        samples_by_split[split].append(sample)

    # Extract features for each split
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        if split not in samples_by_split:
            print(f"No {split} samples")
            continue

        split_samples = samples_by_split[split]
        print(f"\nProcessing {split}: {len(split_samples)} samples...")

        # Extract features
        all_data = {col: [] for col in [
            "sample_id", "meeting_id", "pseudo_robot", "time",
            "context_start", "context_end", "split",
            "final_label", "training_weight", "exclude_from_training",
            "X_frame", "X_scalar",
            "weak_label", "llm_confidence"
        ]}

        success_count = 0
        error_count = 0

        for sample in tqdm(split_samples, desc=f"Extracting {split}"):
            try:
                x_frame, x_scalar = extractor.extract_features(sample)

                # Add to data dict
                all_data["sample_id"].append(sample["sample_id"])
                all_data["meeting_id"].append(sample["meeting_id"])
                all_data["pseudo_robot"].append(sample["pseudo_robot"])
                all_data["time"].append(sample["time"])
                all_data["context_start"].append(sample["context_start"])
                all_data["context_end"].append(sample["context_end"])
                all_data["split"].append(split)
                all_data["final_label"].append(sample["final_label"])
                all_data["training_weight"].append(float(sample.get("training_weight", 1.0)))
                all_data["exclude_from_training"].append(bool(sample.get("exclude_from_training", False)))
                all_data["X_frame"].append(x_frame)
                all_data["X_scalar"].append(x_scalar)
                all_data["weak_label"].append(sample.get("weak_label"))
                all_data["llm_confidence"].append(float(sample.get("llm_confidence", 0.5)))

                success_count += 1

            except Exception as e:
                if error_count < 5:  # Only print first 5 errors
                    print(f"Error processing {sample['sample_id']}: {e}")
                error_count += 1
                continue

        # Create dataframe and save
        print(f"Successfully processed {success_count} samples, {error_count} errors")

        if success_count > 0:
            df = pd.DataFrame(all_data)
            output_file = output_dir / f"{split}.parquet"
            df.to_parquet(output_file, index=False)
            print(f"✓ Saved {len(df)} samples to {output_file}")
        else:
            print(f"⚠ Skipped {split} - no successful samples")

    print("\n" + "=" * 80)
    print("Dataset preparation complete!")
    print(f"Output: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare turn-taking timing dataset from HuggingFace AMI dataset"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/processed/final_manifest.jsonl",
        help="Path to metadata JSONL file with sample definitions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/hf_export",
        help="Output directory for parquet files",
    )

    args = parser.parse_args()
    main(args)
