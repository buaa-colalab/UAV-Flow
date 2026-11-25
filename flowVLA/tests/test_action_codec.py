"""Action-codebook, normalization, dataset-label and checkpoint contracts."""

import copy
import io
import unittest
import numpy as np
import torch
from torch import nn
from starVLA.model.action_codec.action_codebook import ActionCodebook, make_manifest, manifest_digest, ACTION_ORDER
import contextlib
import json
from pathlib import Path
import tempfile
from omegaconf import OmegaConf
from PIL import Image
from starVLA.dataloader.uav_datasets import SimpleVLADataset, make_uav_datasets


STATS = {
    "action_min": [-28.63044126414136, -26.312990424491574, -14.00134, -0.11008031471619936],
    "action_max": [46.29771115923434, 24.970052427293453, 14.1422, 0.08339738765052823],
}


class ActionCodecTests(unittest.TestCase):
    def setUp(self):
        self.codec = ActionCodebook(make_manifest(STATS))

    def test_zero_uses_nearest_existing_value_without_inserting_a_bin(self):
        ids = self.codec.encode_physical(torch.zeros(100, 4, dtype=torch.float64))
        expected_ids = np.argmin(abs(self.codec.centers), axis=1)
        np.testing.assert_array_equal(ids.numpy(), np.tile(expected_ids, (100, 1)))
        values = self.codec.decode_physical(ids).numpy()
        expected_values = self.codec.centers[np.arange(4), expected_ids]
        np.testing.assert_array_equal(values, np.tile(expected_values, (100, 1)))
        self.assertTrue((abs(values) <= np.array(self.codec.manifest["step"]) / 2 + 1e-12).all())

    def test_all_classes_round_trip_including_normalized_policy_api(self):
        ids = torch.arange(256)[:, None].expand(-1, 4)
        self.assertTrue(torch.equal(self.codec.encode_physical(self.codec.decode_physical(ids)), ids))
        self.assertTrue(torch.equal(self.codec.encode_normalized(self.codec.decode_normalized(ids)), ids))
        self.assertTrue(torch.equal(torch.from_numpy(self.codec.encode_numpy(self.codec.centers.T)), ids))

    def test_nearest_values_numpy_torch_and_fuzz(self):
        rng = np.random.default_rng(42)
        values = rng.uniform(self.codec.low * 1.5, self.codec.high * 1.5, (25000, 4))
        observed = self.codec.encode_numpy(values)
        # Independent nearest-distance reference, chunked to bound memory.
        expected = np.stack(
            [np.argmin(abs(values[:, d, None] - self.codec.centers[d]), axis=1) for d in range(4)], axis=-1
        )
        np.testing.assert_array_equal(observed, expected)
        np.testing.assert_array_equal(self.codec.encode_physical(torch.tensor(values)).numpy(), expected)
        decoded = self.codec.decode_physical(torch.tensor(observed)).numpy()
        clipped = np.clip(values, self.codec.low, self.codec.high)
        self.assertTrue((abs(decoded - clipped) <= np.array(self.codec.manifest["step"]) / 2 + 1e-12).all())

    def test_endpoints_clipping_and_midpoint_ties(self):
        np.testing.assert_array_equal(self.codec.encode_numpy(self.codec.low * 100), [0] * 4)
        np.testing.assert_array_equal(self.codec.encode_numpy(self.codec.high * 100), [255] * 4)
        for i in [0, 126, 127, 254]:
            boundary = self.codec.boundaries[:, i]
            np.testing.assert_array_equal(self.codec.encode_numpy(boundary), [i] * 4)
            np.testing.assert_array_equal(self.codec.encode_physical(torch.tensor(boundary)).numpy(), [i] * 4)
        ids = torch.tensor([[0] * 4, [255] * 4])
        np.testing.assert_array_equal(self.codec.decode_physical(ids).numpy(), [self.codec.low, self.codec.high])

    def test_bfloat16_conversion_cannot_modify_codec(self):
        before = self.codec.get_extra_state()
        self.codec.to(dtype=torch.bfloat16)
        self.assertEqual(before, self.codec.get_extra_state())
        ids = torch.arange(256)[:, None].expand(-1, 4)
        self.assertEqual(self.codec.decode_normalized(ids).dtype, torch.float32)
        self.assertEqual(self.codec.decode_physical(ids).dtype, torch.float64)
        self.assertTrue(torch.equal(self.codec.encode_normalized(self.codec.decode_normalized(ids)), ids))

    def test_checkpoint_round_trip_and_fail_closed(self):
        stream = io.BytesIO()
        torch.save(self.codec.state_dict(), stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        self.codec.load_state_dict(state, strict=True)
        with self.assertRaisesRegex(RuntimeError, "missing versioned"):
            self.codec.load_state_dict({}, strict=False)
        other = ActionCodebook(make_manifest(dict(STATS, action_max=[v * 1.01 for v in STATS["action_max"]])))
        with self.assertRaisesRegex(RuntimeError, "differs"):
            other.load_state_dict(state, strict=False)
        corrupt = copy.deepcopy(state)
        corrupt["_extra_state"]["manifest"]["physical_centers"][0][1] += 1
        with self.assertRaisesRegex(RuntimeError, "corrupted"):
            self.codec.load_state_dict(corrupt)

    def test_malformed_inputs_rejected(self):
        for x in [np.zeros((3,)), np.full((4,), np.nan), np.full((4,), np.inf)]:
            with self.assertRaises(ValueError):
                self.codec.encode_numpy(x)
            with self.assertRaises(ValueError):
                self.codec.encode_physical(torch.tensor(x))
        for x in [torch.zeros(4), torch.tensor([-1, 0, 0, 0]), torch.tensor([256, 0, 0, 0])]:
            with self.assertRaises(ValueError):
                self.codec.decode_physical(x)
        for low in [STATS["action_max"], [100] * 4, [float("nan")] * 4]:
            with self.assertRaises(ValueError):
                make_manifest(dict(STATS, action_min=low))
        with self.assertRaises(ValueError):
            ActionCodebook(make_manifest(STATS), "0" * 64)
        with self.assertRaisesRegex(ValueError, "retain FP32"):
            self.codec.encode_normalized(torch.zeros(4, dtype=torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_cpu_parity(self):
        values = torch.tensor(np.random.default_rng(8).uniform(self.codec.low, self.codec.high, (4096, 4)))
        cpu = self.codec.encode_physical(values)
        gpu = self.codec.encode_physical(values.cuda())
        self.assertTrue(torch.equal(cpu, gpu.cpu()))
        self.assertTrue(torch.equal(self.codec.decode_physical(cpu), self.codec.decode_physical(gpu).cpu()))


class FrameworkCodecRegression(unittest.TestCase):
    def test_legacy_and_versioned_checkpoints_cannot_mix_even_non_strict(self):
        from starVLA.model.framework.QwenOFT_discrete import Qwenvl_OFT

        legacy = Qwenvl_OFT.__new__(Qwenvl_OFT)
        nn.Module.__init__(legacy)
        legacy.action_codec = None
        versioned = Qwenvl_OFT.__new__(Qwenvl_OFT)
        nn.Module.__init__(versioned)
        versioned.action_codec = ActionCodebook(make_manifest(STATS))
        with self.assertRaisesRegex(RuntimeError, "legacy floor decoder"):
            legacy.load_state_dict(versioned.state_dict(), strict=False)
        with self.assertRaisesRegex(RuntimeError, "missing versioned"):
            versioned.load_state_dict(legacy.state_dict(), strict=False)

    def test_framework_zero_and_class_mapping_use_the_same_codec(self):
        from starVLA.model.framework.QwenOFT_discrete import Qwenvl_OFT

        model = Qwenvl_OFT.__new__(Qwenvl_OFT)
        nn.Module.__init__(model)
        model.num_action_bins = 256
        model.action_codec = ActionCodebook(make_manifest(STATS))
        ids = torch.arange(256)[:, None].expand(-1, 4)
        self.assertTrue(torch.equal(model.continuous_to_discrete(model.discrete_to_continuous(ids)), ids))
        norm_zero = torch.tensor(model.action_codec.normalize_numpy(np.zeros((10, 4))), dtype=torch.float32)
        labels = model.continuous_to_discrete(norm_zero)
        np.testing.assert_array_equal(labels.numpy(), model.action_codec.encode_numpy(np.zeros((10, 4))))


class CodecDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        episode = self.data / "one"
        episode.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (16, 16), (i * 80, 30, 40)).save(episode / f"{i:06d}.jpg")
        # Integer input logs exercise preservation of radians and local rotation.
        self.log = {
            "raw_logs": [[0, 0, 0, 0, 45, 0], [1, 1, 0, 0, 45, 0], [1, 1, 0, 0, 45, 0]],
            "preprocessed_logs": [[0, 0, 0, 0, 0, 0], [1.414, 0, 0, 0, 0, 0], [1.414, 0, 0, 0, 0, 0]],
            "instruction": "reach the tree",
        }
        (episode / "log.json").write_text(json.dumps(self.log))
        self.stats = dict(STATS, action_mean=[0] * 4, action_std=[1] * 4, action_order=ACTION_ORDER)
        self.stats_path = self.root / "stats.json"
        self.stats_path.write_text(json.dumps(self.stats))
        self.codec = ActionCodebook(make_manifest(self.stats))
        self.stats.update(action_min=self.codec.low.tolist(), action_max=self.codec.high.tolist())
        self.stats_path.write_text(json.dumps(self.stats))

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self):
        return SimpleVLADataset(self.data, action_codec=self.codec, stats_path=self.stats_path)

    def test_raw_targets_and_terminal_padding_use_nearest_grid_value(self):
        dataset = self.dataset()
        np.testing.assert_allclose(dataset.episodes[0]["actions"][0], [np.sqrt(2), 0, 0, 0], atol=1e-14)
        example = dataset[0]
        np.testing.assert_array_equal(example["action_bins"], self.codec.encode_numpy(dataset.samples[0]["action"]))
        np.testing.assert_array_equal(
            example["action_bins"][1:], np.tile(self.codec.encode_numpy(np.zeros(4)), (9, 1))
        )
        sample = next(i for i, s in enumerate(dataset.samples) if s["image_path"][1].name == "000002.jpg")
        np.testing.assert_array_equal(
            dataset[sample]["action_bins"], np.tile(self.codec.encode_numpy(np.zeros(4)), (10, 1))
        )

    def test_stats_mismatch_is_fatal(self):
        bad = dict(self.stats, action_min=[-28.5, *self.stats["action_min"][1:]])
        self.stats_path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "differ"):
            self.dataset()

    def test_invalid_episode_is_not_silently_skipped(self):
        self.log["raw_logs"].pop()
        (self.data / "one/log.json").write_text(json.dumps(self.log))
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            self.dataset()

    def test_factory_workers_preserve_labels(self):
        cfg = OmegaConf.create(
            {
                "seed": 42,
                "framework": {
                    "action_codec": {
                        "version": self.codec.manifest["version"],
                        "sha256": self.codec.sha256,
                    },
                    "action_model": {"action_dim": 4, "num_action_bins": 256, "action_head_type": "discrete"},
                },
                "datasets": {
                    "vla_data": {
                        "data_root_dir": str(self.data),
                        "normalization_stats_path": str(self.stats_path),
                        "per_device_batch_size": 4,
                        "num_workers": 2,
                    }
                },
            }
        )
        loader = make_uav_datasets(cfg)
        for batch in loader:
            self.assertTrue(all(x["action_bins"].shape == (10, 4) for x in batch))


class Uniform256Layout(unittest.TestCase):
    def test_original_asymmetric_percentiles_and_uniform256(self):
        low = np.array([-28.63, -26.31, -14.0, -0.11])
        high = np.array([46.30, 24.97, 14.14, 0.083])
        codec = ActionCodebook(make_manifest({"action_min": low.tolist(), "action_max": high.tolist()}))
        self.assertEqual(codec.num_bins, 256)
        np.testing.assert_array_equal(codec.low, low)
        np.testing.assert_array_equal(codec.high, high)
        np.testing.assert_array_equal(codec.centers[:, 0], low)
        np.testing.assert_array_equal(codec.centers[:, -1], high)
        expected = np.stack([np.linspace(a, b, 256) for a, b in zip(low, high)])
        np.testing.assert_array_equal(codec.centers, expected)
        np.testing.assert_allclose(
            np.diff(codec.centers, axis=1),
            np.repeat(((high - low) / 255)[:, None], 255, axis=1),
            rtol=1e-12,
            atol=1e-15,
        )


class AssetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stats = dict(STATS, action_order=ACTION_ORDER, action_mean=[0] * 4, action_std=[1] * 4)
        self.stats_path = self.root / "statistics.json"
        self.stats_path.write_text(json.dumps(self.stats))
        self.manifest = make_manifest(self.stats)
        self.cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "starVLA/config/training/qwen35_discrete.yaml")
        self.cfg.datasets.vla_data.normalization_stats_path = str(self.stats_path)
        self.cfg.framework.action_codec.sha256 = None

    def test_existing_uniform256_checkpoint_fingerprint_is_preserved(self):
        from starVLA.model.action_codec.action_codebook import load_codec_from_config

        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "starVLA/config/training/qwen35_discrete.yaml")
        codec = load_codec_from_config(cfg)
        self.assertEqual(codec.sha256, "4e0d50490eb8ae86f74decdafa2eb6c66f5079aebb9538ca5c473861c1157afb")

    def test_legacy_manifest_path_is_unneeded_for_strict_checkpoint_loading(self):
        from starVLA.model.action_codec.action_codebook import load_codec_from_config

        self.cfg.framework.action_codec.manifest_path = "/nonexistent/old-codebook.json"
        self.cfg.framework.action_codec.sha256 = manifest_digest(self.manifest)
        derived = load_codec_from_config(self.cfg)
        derived.load_state_dict(ActionCodebook(self.manifest).state_dict(), strict=True)
        self.assertEqual(derived.manifest, self.manifest)
        self.assertNotIn("manifest_path", self.cfg.framework.action_codec)

    def test_statistics_derive_and_pin_the_exact_uniform_table(self):
        from starVLA.model.action_codec.action_codebook import prepare_action_codec_config, load_codec_from_config

        prepare_action_codec_config(self.cfg)
        codec = load_codec_from_config(self.cfg)
        self.assertEqual(self.cfg.framework.action_codec.sha256, manifest_digest(self.manifest))
        self.assertEqual(codec.manifest, self.manifest)
        for name in ["action_min", "action_max"]:
            self.assertEqual(getattr(codec, "low" if name == "action_min" else "high").tolist(), self.stats[name])
        self.assertNotIn("manifest_path", self.cfg.framework.action_codec)
        self.stats_path.write_text(json.dumps(dict(self.stats, action_max=[80, 40, 20, 0.3])))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_codec_from_config(self.cfg)

    def test_mismatched_stats_hash_order_and_range_are_rejected(self):
        from starVLA.model.action_codec.action_codebook import prepare_action_codec_config

        for change in [dict(action_order=list(reversed(ACTION_ORDER))), dict(action_min=[100] * 4), dict(action_codec_sha256="0" * 64)]:
            with self.subTest(change=change):
                self.stats_path.write_text(json.dumps(dict(self.stats, **change)))
                with self.assertRaises(ValueError):
                    prepare_action_codec_config(self.cfg)

    def test_server_rebuilds_codec_from_stats_without_a_manifest_file(self):
        from unittest.mock import patch
        from starVLA.model.action_codec.action_codebook import prepare_action_codec_config
        from starVLA.model.framework.ServerQwen35OFT import Qwen35OFTServer

        prepare_action_codec_config(self.cfg)
        OmegaConf.save(self.cfg, self.root / "config.yaml")
        checkpoint = self.root / "weights.pt"
        checkpoint.touch()
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "STATS_PATH"):
                Qwen35OFTServer(self.root, checkpoint)
            self.assertFalse((self.root / "uav_action_statistics.json").exists())
            # No action_codec.json or other manifest file is supplied.
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                Qwen35OFTServer(self.root, checkpoint, stats_path=self.stats_path)
            self.stats_path.write_text(json.dumps(dict(self.stats, action_max=[80, 40, 20, 0.3])))
            with self.assertRaisesRegex(ValueError, "checksum"):
                Qwen35OFTServer(self.root, checkpoint, stats_path=self.stats_path)

    def test_frozen_statistics_are_available_without_versioned_codec(self):
        from unittest.mock import patch
        from starVLA.training.train_starvla_vla import prepare_data
        from types import SimpleNamespace

        episode = dict(image_paths=[Path("first.png")], actions=np.zeros((1, 4)), proprio=np.zeros((1, 4)), instruction="go")
        with patch.object(SimpleVLADataset, "_load_data", return_value=[episode]):
            dataset = SimpleVLADataset(self.root, stats_path=self.stats_path)
        self.assertIsNone(dataset.action_codec)
        np.testing.assert_array_equal(dataset.action_min, self.stats["action_min"])
        np.testing.assert_array_equal(dataset._normalize_action(np.zeros(4)), 2 * (0 - np.array(self.stats["action_min"])) / (np.array(self.stats["action_max"]) - self.stats["action_min"]) - 1)
        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "starVLA/config/training/qwen3vl_discrete.yaml")
        accelerator = SimpleNamespace(is_main_process=True, dataloader_config=SimpleNamespace())
        with patch("starVLA.training.train_starvla_vla.build_dataloader", return_value=SimpleNamespace(dataset=dataset)):
            prepare_data(cfg, accelerator, self.root)
        saved = json.loads((self.root / "uav_action_statistics.json").read_text())
        self.assertEqual(saved["action_min"], self.stats["action_min"])
        self.assertEqual(saved["action_max"], self.stats["action_max"])
        self.assertFalse((self.root / "action_codec.json").exists())


if __name__ == "__main__":
    unittest.main()
