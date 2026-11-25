"""Nearest-value action quantization on the original uniform p1/p99 grid.

Floating codebooks are immutable metadata, not floating module buffers: model
BF16 conversion must not collapse nearby bins or alter their physical meaning.
The full manifest and its digest are checkpoint extra state. Device tables are
derived caches, always FP64 for physical values and FP32 for normalized values.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

VERSION = "uav_uniform256_p1p99_v1"
ACTION_ORDER = ["local_dx", "local_dy", "local_dz", "local_dyaw_radians"]


def manifest_digest(manifest):
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def make_manifest(stats, num_bins=256):
    source_low, source_high = (np.asarray(stats[k], dtype=np.float64) for k in ("action_min", "action_max"))
    if (
        source_low.shape != (4,)
        or source_high.shape != (4,)
        or not np.isfinite([source_low, source_high]).all()
        or not (source_high > source_low).all()
    ):
        raise ValueError("Codebook bounds must be finite p1/p99 with max > min on all axes")
    if stats.get("action_order", ACTION_ORDER) != ACTION_ORDER:
        raise ValueError("Unsupported action dimension order")
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins != 256:
        raise ValueError("Uniform256 requires exactly 256 codebook values")
    centers = np.stack([np.linspace(a, b, 256, dtype=np.float64) for a, b in zip(source_low, source_high)])
    return {
        "version": VERSION,
        "num_bins": 256,
        "action_order": ACTION_ORDER,
        "action_min": source_low.tolist(),
        "action_max": source_high.tolist(),
        "source_p1": source_low.tolist(),
        "source_p99": source_high.tolist(),
        "bounds_policy": "p1p99",
        "step": ((source_high - source_low) / 255).tolist(),
        "physical_centers": centers.tolist(),
        "encoding": "nearest_value_ties_lower",
        "clipping": "bounds",
    }


class ActionCodebook(nn.Module):
    def __init__(self, manifest, expected_sha256=None):
        super().__init__()
        self.manifest = copy.deepcopy(manifest)
        self.sha256 = manifest_digest(self.manifest)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("Action codebook checksum mismatch")
        if (
            self.manifest.get("version") != VERSION
            or self.manifest.get("encoding") != "nearest_value_ties_lower"
            or self.manifest.get("clipping") != "bounds"
        ):
            raise ValueError("Unsupported action codec version or encoding")
        self.num_bins = self.manifest["num_bins"]
        if isinstance(self.num_bins, bool) or not isinstance(self.num_bins, int) or self.num_bins < 3:
            raise ValueError("Invalid bin count")
        if self.manifest["action_order"] != ACTION_ORDER:
            raise ValueError("Unsupported action dimension order")
        self.low = np.asarray(self.manifest["action_min"], dtype=np.float64)
        self.high = np.asarray(self.manifest["action_max"], dtype=np.float64)
        self.centers = np.asarray(self.manifest["physical_centers"], dtype=np.float64)
        if self.low.shape != (4,) or self.high.shape != (4,) or self.centers.shape != (4, self.num_bins):
            raise ValueError("Invalid codebook dimensions")
        if not np.isfinite([self.low, self.high]).all() or not np.isfinite(self.centers).all():
            raise ValueError("Nonfinite codebook")
        if not (self.high > self.low).all() or not (np.diff(self.centers, axis=1) > 0).all():
            raise ValueError("Codebook must be strictly increasing")
        if not np.array_equal(self.centers[:, 0], self.low) or not np.array_equal(self.centers[:, -1], self.high):
            raise ValueError("Codebook endpoints differ from normalization bounds")
        # Validate the declared construction as well as its serialized values.
        rebuilt = make_manifest(
            {
                "action_min": self.manifest["source_p1"],
                "action_max": self.manifest["source_p99"],
                "action_order": self.manifest["action_order"],
            },
            self.num_bins,
        )
        if rebuilt != self.manifest:
            raise ValueError("Manifest does not match its declared codebook construction")
        self.boundaries = (self.centers[:, :-1] + self.centers[:, 1:]) / 2
        self.normalized_centers = 2 * (self.centers - self.low[:, None]) / (self.high - self.low)[:, None] - 1
        for array in (self.low, self.high, self.centers, self.boundaries, self.normalized_centers):
            array.setflags(write=False)
        self._tables = {}

    def get_extra_state(self):
        return {"version": VERSION, "sha256": self.sha256, "manifest": copy.deepcopy(self.manifest)}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("version") != VERSION or state.get("sha256") != self.sha256:
            raise RuntimeError("Checkpoint action codec version/checksum differs from this model")
        if state.get("manifest") != self.manifest or manifest_digest(state["manifest"]) != self.sha256:
            raise RuntimeError("Checkpoint contains a different or corrupted action codebook")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if prefix + "_extra_state" not in state_dict:
            error_msgs.append("Checkpoint is missing versioned action codec; legacy labels cannot restore this model")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _apply(self, fn, recurse=True):
        self._tables.clear()
        return super()._apply(fn, recurse=recurse)

    def _table(self, name, device):
        key = (name, str(device))
        if key not in self._tables:
            array = getattr(self, name)
            dtype = torch.float32 if name == "normalized_centers" else torch.float64
            self._tables[key] = torch.tensor(array.copy(), device=device, dtype=dtype)
        return self._tables[key]

    @staticmethod
    def _validate_numpy(values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 4 or not np.isfinite(values).all():
            raise ValueError("Actions must be finite with final dimension 4")
        return values

    def normalize_numpy(self, physical):
        values = self._validate_numpy(physical)
        return np.clip(2 * (values - self.low) / (self.high - self.low) - 1, -1, 1)

    def encode_numpy(self, physical):
        values = self._validate_numpy(physical)
        return np.stack(
            [np.searchsorted(self.boundaries[d], values[..., d], side="left") for d in range(4)], axis=-1
        ).astype(np.int64)

    def encode_physical(self, physical):
        if (
            not isinstance(physical, torch.Tensor)
            or physical.ndim < 1
            or physical.shape[-1] != 4
            or not torch.isfinite(physical).all()
        ):
            raise ValueError("Actions must be finite tensors with final dimension 4")
        values = physical.to(torch.float64)
        boundaries = self._table("boundaries", values.device)
        return torch.stack([torch.bucketize(values[..., d].contiguous(), boundaries[d]) for d in range(4)], dim=-1)

    def encode_normalized(self, normalized):
        if not isinstance(normalized, torch.Tensor) or normalized.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "Normalized policy actions must retain FP32/FP64 precision; use integer class IDs for storage"
            )
        low, high = self._table("low", normalized.device), self._table("high", normalized.device)
        return self.encode_physical((normalized.to(torch.float64) + 1) / 2 * (high - low) + low)

    def _lookup(self, indices, name):
        if (
            not isinstance(indices, torch.Tensor)
            or indices.ndim < 1
            or indices.shape[-1] != 4
            or indices.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)
        ):
            raise ValueError("Class indices must be integer tensors with final dimension 4")
        if ((indices < 0) | (indices >= self.num_bins)).any():
            raise ValueError("Action class index outside codebook")
        table = self._table(name, indices.device)
        return torch.stack([table[d, indices[..., d].long()] for d in range(4)], dim=-1)

    def decode_physical(self, indices):
        return self._lookup(indices, "centers")

    def decode_normalized(self, indices):
        return self._lookup(indices, "normalized_centers")


def load_codec_from_config(config):
    """Derive the uniform256 table from the selected normalization statistics."""
    spec = config.framework.get("action_codec")
    if spec is None:
        return None
    if spec.get("version") != VERSION:
        raise ValueError("Unsupported action codec version")
    stats_path = config.datasets.vla_data.get("normalization_stats_path")
    if not stats_path:
        raise ValueError("Versioned codec requires frozen normalization statistics")
    stats = load_normalization_stats(stats_path)
    codec = ActionCodebook(make_manifest(stats), spec.get("sha256"))
    if stats.get("action_codec_sha256", codec.sha256) != codec.sha256:
        raise ValueError("Normalization statistics/codec checksum mismatch")
    action = config.framework.action_model
    if action.action_dim != 4 or action.num_action_bins != codec.num_bins or action.action_head_type != "discrete":
        raise ValueError("Action head dimensions/type differ from codebook")
    spec.sha256 = codec.sha256
    # Older saved configurations may still contain a manifest file path.
    spec.pop("manifest_path", None)
    return codec


def load_normalization_stats(path, codec=None):
    """Load frozen four-axis statistics and validate their optional codebook binding."""
    stats = json.loads(Path(path).expanduser().read_text())
    if stats.get("action_order", ACTION_ORDER) != ACTION_ORDER:
        raise ValueError("Normalization action order differs from codec")
    for name in ("action_min", "action_max", "action_mean", "action_std"):
        values = np.asarray(stats[name], dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid frozen {name}")
    low, high = (np.asarray(stats[name], dtype=np.float64) for name in ("action_min", "action_max"))
    if not (high > low).all():
        raise ValueError("Normalization bounds must increase on every action dimension")
    if codec is not None:
        if not np.array_equal(low, codec.low) or not np.array_equal(high, codec.high):
            raise ValueError("Dataset normalization bounds differ from action codebook")
        if stats.get("action_codec_sha256", codec.sha256) != codec.sha256:
            raise ValueError("Normalization statistics/codec checksum mismatch")
    return stats


def prepare_action_codec_config(config):
    """Resolve selected assets and pin their content before model initialization."""
    data = config.datasets.vla_data
    stats_path = data.get("normalization_stats_path")
    if stats_path:
        data.normalization_stats_path = str(Path(stats_path).expanduser().resolve())
    if config.framework.get("action_codec") is not None:
        load_codec_from_config(config)
    elif stats_path:
        load_normalization_stats(data.normalization_stats_path)
