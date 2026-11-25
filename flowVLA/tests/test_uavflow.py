"""UAV data loading, training configuration, model compatibility and HTTP regression checks.

Optional snapshot comparisons use FLOWVLA_REFERENCE_SOURCE and FLOWVLA_REDUNDANCY_REFERENCE."""

import base64
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
from starVLA.model.framework.ServerQwen35OFT import Qwen35OFTServer, integrate_actions, load_action_range
from types import SimpleNamespace
from unittest.mock import patch
import warnings
import torch
from omegaconf import OmegaConf
import importlib.util
import os
import subprocess
import sys
from torch import nn
import contextlib
from collections import Counter
import io
import math
from torch.utils.data import DataLoader
from accelerate.data_loader import prepare_data_loader
from starVLA.dataloader.uav_datasets import SimpleVLADataset, collate_fn, make_uav_datasets
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups, TrainerUtils
import ast
from transformers import get_scheduler
from starVLA.training import train_starvla_vla as trainer


def encoded(color):
    stream = BytesIO()
    Image.new("RGB", (256, 256), color).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


class FakeModel:
    def predict_action(self, batch_images, instructions, state):
        self.first_pixel = batch_images[0][0].getpixel((0, 0))
        self.instruction = instructions[0]
        return {"normalized_actions": np.zeros((1, 10, 4))}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.server = Qwen35OFTServer.__new__(Qwen35OFTServer)
        self.server.model = FakeModel()
        self.server.info = {"image_size": 256}
        self.server.action_min = -np.ones(4)
        self.server.action_max = np.ones(4)
        self.client = self.server.create_app().test_client()
        self.payload = dict(image=encoded("red"), instr="reach the tree", proprio=[1, 2, 3, 90])

    def test_sessions_reset_and_stateless_first_frame(self):
        self.assertEqual(self.client.post("/predict", json=self.payload).status_code, 200)
        blue = dict(self.payload, image=encoded("blue"))
        self.client.post("/predict", json=blue)
        self.assertEqual(self.server.model.first_pixel, (255, 0, 0))
        self.client.post("/predict", json=dict(blue, session_id="second"))
        self.assertEqual(self.server.model.first_pixel, (0, 0, 255))
        self.client.post("/predict", json=dict(blue, initial_image=encoded("green")))
        self.assertEqual(self.server.model.first_pixel, (0, 128, 0))
        self.client.post("/predict", json=blue)
        self.assertEqual(self.server.model.first_pixel, (255, 0, 0))
        self.client.post("/reset", json={})
        result = self.client.post("/predict", json=blue).get_json()
        self.assertEqual(self.server.model.first_pixel, (0, 0, 255))
        np.testing.assert_allclose(result["action"], np.tile([1, 2, 3, np.pi / 2], (10, 1)))
        self.assertIn("Current State: 1.0,2.0,3.0,90.0", self.server.model.instruction)

    def test_invalid_inputs_do_not_start_an_episode(self):
        for change in (
            {"image": "bad"},
            {"proprio": [1, 2]},
            {"proprio": [0, 0, float("nan"), 0]},
            {"instr": ""},
            {"session_id": []},
        ):
            self.assertEqual(self.client.post("/predict", json=dict(self.payload, **change)).status_code, 400)
        self.assertEqual(self.client.post("/predict", json=[]).status_code, 400)
        self.assertEqual(self.client.post("/reset", json=[1]).status_code, 400)
        self.assertEqual(self.client.post("/reset", json=[]).status_code, 400)

    def test_sequential_yaw_changes_translation(self):
        local = np.array([[1, 0, 0, np.pi / 2], [1, 0, 0, 0]])
        result = integrate_actions(local, [10, 20, 30, 0])
        np.testing.assert_allclose(result, [[11, 20, 30, np.pi / 2], [11, 21, 30, np.pi / 2]])
        np.testing.assert_allclose(local, [[1, 0, 0, np.pi / 2], [1, 0, 0, 0]])

    def test_reject_incompatible_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            path.write_text(json.dumps({"action_min": [0] * 4, "action_max": [1, 1, 1, 0]}))
            with self.assertRaises(ValueError):
                load_action_range(path)

    def test_versioned_actions_use_exact_physical_lookup(self):
        from starVLA.model.action_codec.action_codebook import ActionCodebook, make_manifest

        codec = ActionCodebook(
            make_manifest({"action_min": [-28.6, -26.3, -14, -0.11], "action_max": [46.3, 25, 14.1, 0.083]})
        )
        import torch

        ids = torch.tensor(codec.encode_numpy(np.zeros((1, 10, 4))))

        class ZeroModel:
            def predict_action(self, **kwargs):
                return {
                    "normalized_actions": codec.decode_normalized(ids).numpy(),
                    "local_actions": codec.decode_physical(ids).numpy(),
                }

        self.server.model = ZeroModel()
        self.server.info["action_codec_version"] = codec.manifest["version"]
        result = self.client.post("/predict", json=self.payload).get_json()
        self.assertEqual(result["status"], "success")
        np.testing.assert_array_equal(result["local_action"], codec.decode_physical(ids)[0].numpy())
        np.testing.assert_array_equal(
            result["action"], integrate_actions(codec.decode_physical(ids)[0].numpy(), self.payload["proprio"])
        )


RECIPES = {
    "qwen3vl_fast": ("QwenFast", False),
    "qwen3vl_continuous": ("QwenOFT-L1", False),
    "qwen3vl_discrete": ("QwenOFT", False),
    "qwen3vl_discrete_alllayer": ("QwenOFT", True),
}


class RetainedVariants(unittest.TestCase):
    def test_qwen3_recipes_have_explicit_head_semantics(self):
        from starVLA.training.train_starvla_vla import validate_training_config

        observed = []
        for path in Path("starVLA/config/training").glob("*.yaml"):
            cfg = OmegaConf.load(path)
            if "Qwen3-VL" not in str(OmegaConf.select(cfg, "framework.qwenvl.base_vlm")):
                continue
            observed.append(path.stem)
            name = observed[-1]
            self.assertIn(name, RECIPES)
            framework, all_layer = RECIPES[name]
            self.assertEqual(cfg.framework.name, framework)
            self.assertEqual(bool(cfg.framework.get("use_all_layer_hidden", False)), all_layer)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                validate_training_config(cfg)
        self.assertEqual(set(observed), set(RECIPES))

    def test_continuous_route_rejects_all_layer_configuration(self):
        from starVLA.model.framework import QwenOFT_continuous as module

        cfg = OmegaConf.load("starVLA/config/training/qwen3vl_continuous.yaml")
        cfg.framework.use_all_layer_hidden = True
        with self.assertRaisesRegex(ValueError, "Continuous all-layer is unsupported"):
            trainer.validate_training_config(cfg)
        with patch.object(module, "get_vlm_model") as load_backbone:
            with self.assertRaisesRegex(ValueError, "Continuous all-layer is unsupported"):
                module.Qwenvl_OFT(cfg)
            load_backbone.assert_not_called()

    def test_sim_server_requires_explicit_matching_stats(self):
        from starVLA.model.framework.ServerQwenOFT_dis import QwenOFTServer

        server = QwenOFTServer.__new__(QwenOFTServer)
        with self.assertRaises(ValueError):
            server.load_action_stats({})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stats.json"
            path.write_text(json.dumps({"action_min": [-4, -3, -2, -1], "action_max": [1, 2, 3, 4]}))
            low, high = server.load_action_stats({"stats_path": str(path)})
            np.testing.assert_array_equal(low, [-4, -3, -2, -1])
            np.testing.assert_array_equal(high, [1, 2, 3, 4])


class FactoryContracts(unittest.TestCase):
    def test_package_import_does_not_discover_frameworks(self):
        code = """
import sys
import starVLA.model.framework
names = ('M1', 'QwenDual', 'QwenFM', 'QwenFast', 'QwenGR00T',
         'QwenOFT_discrete', 'QwenOFT_continuous', 'QwenPI', 'ServerQwen35OFT')
assert not any('starVLA.model.framework.' + n in sys.modules for n in names)
assert 'diffusers' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_builtin_routes_and_legacy_config(self):
        import starVLA.model.framework as factory

        expected = {
            "InternVLA-M1": ("M1", "InternVLA_M1"),
            "QwenOFT": ("QwenOFT_discrete", "Qwenvl_OFT"),
            "QwenOFT-L1": ("QwenOFT_continuous", "Qwenvl_OFT"),
            "QwenFast": ("QwenFast", "Qwenvl_Fast"),
            "QwenFM": ("QwenFM", "Qwenvl_FMHead"),
            "QwenPI": ("QwenPI", "Qwen_PI"),
            "Qwen-Dual": ("QwenDual", "Qwen_Dual"),
            "QwenGR00T": ("QwenGR00T", "Qwen_GR00T"),
        }
        for name, (module_name, class_name) in expected.items():
            for legacy in (False, True):
                with self.subTest(name=name, legacy=legacy):
                    cfg = OmegaConf.create({"framework": {"framework_py" if legacy else "name": name}})
                    marker = object()
                    target = SimpleNamespace(**{class_name: lambda config: (marker, config)})
                    with patch.object(factory, "import_module", return_value=target) as importer:
                        self.assertEqual(factory.build_framework(cfg), (marker, cfg))
                    importer.assert_called_once()
                    args, kwargs = importer.call_args
                    requested = args[0] if args else kwargs["name"]
                    package = kwargs.get("package", args[1] if len(args) > 1 else None)
                    self.assertEqual(
                        importlib.util.resolve_name(requested, package),
                        "starVLA.model.framework." + module_name,
                    )

    def test_registered_extension_and_unknown_name(self):
        import starVLA.model.framework as factory

        key = "__refactor_extension__"
        cfg = OmegaConf.create({"framework": {"name": key}})
        try:
            factory.FRAMEWORK_REGISTRY.register(key)(lambda config: config)
            self.assertIs(factory.build_framework(cfg), cfg)
        finally:
            factory.FRAMEWORK_REGISTRY._registry.pop(key, None)
        with self.assertRaises(NotImplementedError):
            factory.build_framework(cfg)


class SharedConstructionContracts(unittest.TestCase):
    def test_instruction_text_is_byte_identical(self):
        from starVLA.dataloader.uav_prompts import format_uav_instruction

        for values in ([0.0, -0.0, 1.25, -1.25], [12, -13, 0.05, 359.95], np.array([0.15, -0.15, 1.0001, -179.95])):
            for task in ("reach the tree", " 中文 punctuation!? "):
                state = ",".join(str(round(x, 1)) for x in values)
                self.assertEqual(
                    format_uav_instruction(values, task),
                    f"Current State: {state}, What action should the uav take to {task}?",
                )

    def test_lora_preserves_original_state_paths_and_initialization(self):
        from peft import LoraConfig, get_peft_model
        from starVLA.model.adapters import apply_backbone_lora

        def model():
            result = nn.Module()
            result.qwen_vl_interface = nn.Module()
            result.qwen_vl_interface.model = nn.Module()
            core = nn.Module()
            core.q_proj = nn.Linear(8, 8)
            result.qwen_vl_interface.model.model = core
            result.qwen_vl_interface.action_query_embeddings = nn.Parameter(torch.randn(10, 8))
            result.action_model = nn.Linear(8, 4)
            return result

        cfg = OmegaConf.create({"r": 2, "lora_alpha": 4, "target_modules": ["q_proj"], "lora_dropout": 0.05})
        torch.manual_seed(52)
        old = model()
        torch.manual_seed(52)
        new = model()
        torch.manual_seed(53)
        old.qwen_vl_interface.model.model = get_peft_model(
            old.qwen_vl_interface.model.model,
            LoraConfig(
                r=cfg.r,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.target_modules,
                lora_dropout=cfg.lora_dropout,
                init_lora_weights="gaussian",
            ),
        )
        torch.manual_seed(53)
        self.assertIs(apply_backbone_lora(new, cfg), new)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[name]), name)
        self.assertEqual(
            {n for n, p in old.named_parameters() if p.requires_grad},
            {n for n, p in new.named_parameters() if p.requires_grad},
        )


class TinyBackbone(nn.Module):
    def __init__(self, learned):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.projection = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.15)
        self.learned = learned
        self.calls = []
        if learned:
            self.action_query_embeddings = nn.Parameter(torch.randn(10, 8))

    def build_qwenvl_inputs(self, images, instructions):
        self.instructions = list(instructions)
        batch = len(instructions)
        ids = torch.tensor([[1, 2, 3] + ([] if self.learned else [7] * 10)]).expand(batch, -1)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.ones(batch, 2),
            "image_grid_thw": torch.ones(batch, 3, dtype=torch.long),
            "mm_token_type_ids": torch.zeros_like(ids),
        }

    def forward(self, input_ids, **kwargs):
        self.calls.append({k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)})
        x = self.embedding(input_ids)
        if self.learned:
            x = torch.cat((x, self.action_query_embeddings[None].expand(len(input_ids), -1, -1)), 1)
        x = self.dropout(x)
        y = torch.tanh(self.projection(x))
        return SimpleNamespace(hidden_states=(x, y, y + x * 0.25))


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 4 * 256)
        self.dropout = nn.Dropout(0.1)

    def predict_action(self, features):
        return self.projection(self.dropout(features)).reshape(*features.shape[:2], 4, 256)


def make_framework(cls, learned, all_layers, versioned):
    from starVLA.model.action_codec import ActionCodebook
    from starVLA.model.action_codec.action_codebook import make_manifest

    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = OmegaConf.create({"datasets": {"vla_data": {}}})
    model.qwen_vl_interface = TinyBackbone(learned)
    model.action_model = TinyHead()
    model.future_action_window_size = 9
    model.past_action_window_size = 0
    model.chunk_len = 10
    model.num_action_bins = 256
    model.action_token = None if learned else "Q"
    model.action_token_id = None if learned else 7
    model.use_learned_action_queries = learned
    model.use_all_layer_hidden = all_layers
    if all_layers:
        model.layer_weights = nn.Parameter(torch.tensor([-0.3, 0.1, 0.5]))
    model.action_codec = (
        ActionCodebook(make_manifest({"action_min": [-2.0, -1.0, -3.0, -0.1], "action_max": [3.0, 4.0, 2.0, 0.2]}))
        if versioned
        else None
    )
    model.ce_loss = nn.CrossEntropyLoss()
    model.dr_enabled = False
    return model


@unittest.skipUnless(os.environ.get("FLOWVLA_REFERENCE_SOURCE"), "optional original framework snapshot")
class SnapshotEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from starVLA.model.framework.QwenOFT_discrete import Qwenvl_OFT

        spec = importlib.util.spec_from_file_location(
            "_flowvla_original_framework", Path(os.environ["FLOWVLA_REFERENCE_SOURCE"])
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.original, cls.current = module.Qwenvl_OFT, Qwenvl_OFT

    def compare(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            if isinstance(left[key], torch.Tensor):
                self.assertTrue(torch.equal(left[key], right[key]), key)
            else:
                np.testing.assert_array_equal(left[key], right[key], err_msg=key)

    def test_full_outputs_gradients_sampling_and_policy(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'.*")
            for learned in (False, True):
                for layers in (False, True):
                    for versioned in (False, True):
                        with self.subTest(learned=learned, all_layers=layers, versioned=versioned):
                            torch.manual_seed(101)
                            old = make_framework(self.original, learned, layers, versioned)
                            new = make_framework(self.current, learned, layers, versioned)
                            new.load_state_dict(old.state_dict(), strict=True)
                            self.assertEqual(set(old.state_dict()), set(new.state_dict()))
                            action = np.linspace(-0.9, 0.9, 40).reshape(10, 4)
                            examples = [
                                {"image": [None, None], "lang": "move forward", "action": action} for _ in range(2)
                            ]
                            if versioned:
                                for example in examples:
                                    example["action_bins"] = old.action_codec.encode_numpy(action)
                            torch.manual_seed(102)
                            a = old(examples)
                            a["action_loss"].backward()
                            torch.manual_seed(102)
                            b = new(examples)
                            b["action_loss"].backward()
                            self.compare(a, b)
                            self.assertEqual(old.qwen_vl_interface.instructions, new.qwen_vl_interface.instructions)
                            for (name, p), (other, q) in zip(old.named_parameters(), new.named_parameters()):
                                self.assertEqual(name, other)
                                self.assertEqual(p.grad is None, q.grad is None, name)
                                if p.grad is not None:
                                    self.assertTrue(torch.equal(p.grad, q.grad), name)
                            old.eval()
                            new.eval()
                            self.compare(
                                old.predict_action([[None, None]], ["move forward"]),
                                new.predict_action([[None, None]], ["move forward"]),
                            )
                            self.assertEqual(old.qwen_vl_interface.instructions, new.qwen_vl_interface.instructions)
                            inputs = old.qwen_vl_interface.build_qwenvl_inputs([[None, None]], ["move forward"])
                            torch.manual_seed(103)
                            a = old.sample_action_from_inputs(**inputs, temperature=1.3)
                            torch.manual_seed(103)
                            b = new.sample_action_from_inputs(**inputs, temperature=1.3)
                            self.compare(a, b)
                            self.compare(
                                old(mode="logp", **inputs, actions=a["sampled_actions"], temperature=1.3),
                                new(mode="logp", **inputs, actions=b["sampled_actions"], temperature=1.3),
                            )
                            self.assertEqual(old.qwen_vl_interface.calls, new.qwen_vl_interface.calls)


class IndexedSamples(SimpleVLADataset):
    image_path = None

    def __init__(self, data_path, **kwargs):
        self.action_min = np.full(4, -1.0)
        self.action_max = np.full(4, 1.0)
        self.samples = [
            dict(
                image_path=[self.image_path, self.image_path],
                action=np.zeros((10, 4)),
                proprio=np.zeros(4),
                instruction=f"sample_{i}",
            )
            for i in range(int(data_path))
        ]


def ids(batches):
    return [int(s["lang"].rsplit("sample_", 1)[1].rstrip("?")) for b in batches for s in b]


def make_test_loader(n, workers, batch_size=3, seed=42):
    cfg = OmegaConf.create(
        {
            "seed": seed,
            "datasets": {
                "vla_data": {"data_root_dir": str(n), "per_device_batch_size": batch_size, "num_workers": workers}
            },
        }
    )
    with patch("starVLA.dataloader.uav_datasets.SimpleVLADataset", IndexedSamples):
        return make_uav_datasets(cfg)


class UavDataLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        IndexedSamples.image_path = cls.root / "frame.jpg"
        Image.new("RGB", (16, 16), (120, 80, 40)).save(IndexedSamples.image_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_worker_coverage(self):
        for n in (1, 3, 35):
            for workers in (0, 1, 2, 4):
                with self.subTest(n=n, workers=workers):
                    loader = DataLoader(
                        IndexedSamples(str(n)), batch_size=3, num_workers=workers, collate_fn=collate_fn
                    )
                    observed = ids(loader)
                    self.assertEqual(Counter(observed), Counter(range(n)))

    def test_factory_shuffle_is_reproducible_across_workers(self):
        first = ids(make_test_loader(35, 0))
        self.assertNotEqual(first, list(range(35)))
        self.assertEqual(first, ids(make_test_loader(35, 4)))
        self.assertEqual(first, ids(make_test_loader(35, 0)))
        self.assertNotEqual(first, ids(make_test_loader(35, 0, seed=43)))
        loader = make_test_loader(35, 2)
        epoch0, epoch1 = ids(loader), ids(loader)
        self.assertNotEqual(epoch0, epoch1)
        self.assertEqual(Counter(epoch1), Counter(range(35)))
        self.assertEqual(loader.num_workers, 2)

    def test_accelerate_rank_sharding_and_tail_padding(self):
        for n in (1, 131, 256):
            with self.subTest(n=n):
                loaders = [
                    prepare_data_loader(
                        make_test_loader(n, 2, 16),
                        num_processes=8,
                        process_index=rank,
                        split_batches=False,
                        dispatch_batches=False,
                        even_batches=True,
                        rng_types=[],
                    )
                    for rank in range(8)
                ]
                expected_steps = math.ceil(n / 128)
                previous = None
                for epoch in range(2):
                    ranks = []
                    for loader in loaders:
                        batches = list(loader)
                        self.assertEqual(len(batches), expected_steps)
                        self.assertTrue(all(len(b) == 16 for b in batches))
                        ranks.append(ids(batches))
                    merged = [i for rank in ranks for i in rank]
                    self.assertEqual(len(merged), expected_steps * 128)
                    self.assertEqual(set(merged), set(range(n)))
                    if n % 128 == 0:
                        self.assertEqual(Counter(merged), Counter(range(n)))
                        for i in range(8):
                            for j in range(i):
                                self.assertFalse(set(ranks[i]) & set(ranks[j]))
                        if previous is not None:
                            self.assertNotEqual(previous, ranks)
                        previous = ranks
                    else:
                        self.assertLess(len(merged) - n, 128)

    def test_real_episode_weighting_and_stable_order(self):
        root = self.root / "episodes"
        root.mkdir(exist_ok=True)
        raw = [[float(v) for v in row] for row in [[0, 0, 0, 0, 0, 0], [1, 2, 3, 0, 5, 0], [3, 6, 9, 0, 15, 0]]]
        for name in ("b", "a"):
            episode = root / name
            episode.mkdir(exist_ok=True)
            for i in range(3):
                Image.new("RGB", (16, 16), (i * 30, 20, 40)).save(episode / f"{i:06d}.jpg")
            (episode / "log.json").write_text(
                json.dumps({"raw_logs": raw, "preprocessed_logs": raw, "instruction": name})
            )
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = SimpleVLADataset(root)
        self.assertEqual([e["instruction"] for e in dataset.episodes], ["a", "b"])
        self.assertEqual(len(dataset), 36)
        counts = Counter((s["instruction"], s["image_path"][1].name) for s in dataset.samples)
        self.assertTrue(all(count == 6 for count in counts.values()))
        self.assertEqual(dataset[0]["action"].shape, (10, 4))
        self.assertTrue(np.isfinite(dataset[0]["action"]).all())


class ParameterConfigTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Linear(2, 2)
        self.model.head = torch.nn.Linear(2, 1)

    def test_invalid_lr_path_is_rejected(self):
        cfg = OmegaConf.create({"trainer": {"learning_rate": {"base": 0.01, "typo": 0.02}}})
        with self.assertRaises(ValueError):
            build_param_lr_groups(self.model, cfg)

    def test_reserved_lora_rate_and_frozen_parameters(self):
        cfg = OmegaConf.create({"trainer": {"learning_rate": {"base": 0.01, "backbone": 0.02, "lora": 0.03}}})
        self.model.backbone.bias.requires_grad_(False)
        groups = build_param_lr_groups(self.model, cfg)
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in self.model.parameters() if p.requires_grad})
        self.assertEqual([g["name"] for g in groups], ["backbone", "base"])
        self.assertEqual([g["lr"] for g in groups], [0.02, 0.01])

    def test_overlapping_lr_paths_are_rejected(self):
        cfg = OmegaConf.create(
            {"trainer": {"learning_rate": {"base": 0.01, "backbone": 0.02, "backbone.weight": 0.03}}}
        )
        # A tensor is not a submodule; both misspelled/overlapping paths fail clearly.
        with self.assertRaises(ValueError):
            build_param_lr_groups(self.model, cfg)

    def test_invalid_freeze_path_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainerUtils.freeze_backbones(self.model, "typo")

    def test_weights_only_load_works_without_distributed_initialization(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pt"
            torch.save(self.model.state_dict(), path)
            self.assertIs(TrainerUtils.load_pretrained_backbones(self.model, path), self.model)

    def test_freezing_is_applied_before_optimizer_selection(self):
        from starVLA.training.train_starvla_vla import prepare_model_parameters, setup_optimizer_and_scheduler_lora

        cfg = OmegaConf.load("starVLA/config/training/qwen35_discrete.yaml")
        cfg.trainer.freeze_modules = "backbone"
        prepare_model_parameters(self.model, cfg)
        opt, _ = setup_optimizer_and_scheduler_lora(self.model, cfg)
        self.assertEqual(
            {id(p) for g in opt.param_groups for p in g["params"]}, {id(p) for p in self.model.head.parameters()}
        )


REFERENCE = os.environ.get("FLOWVLA_REDUNDANCY_REFERENCE")


class RedundancyEquivalence(unittest.TestCase):
    def assertNestedEqual(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assertNestedEqual(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assertNestedEqual(x, y)
        else:
            self.assertEqual(a, b)

    def test_boundary_repeats_padding_and_independent_actions(self):
        classes = [SimpleVLADataset]
        if REFERENCE:
            path = Path(REFERENCE) / "starVLA/dataloader/uav_datasets.py"
            spec = importlib.util.spec_from_file_location("uav_reference_redundancy", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            classes.insert(0, module.SimpleVLADataset)
        for frames in (1, 3, 6, 12):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as folder:
                episode = Path(folder) / "episode"
                episode.mkdir()
                poses = [[float(i), float(i * i), float(i % 3), 0.0, float(i * 7), 0.0] for i in range(frames)]
                for i in range(frames):
                    Image.new("RGB", (16, 16), (i, 20, 40)).save(episode / f"{i:06d}.png")
                (episode / "log.json").write_text(
                    json.dumps({"raw_logs": poses, "preprocessed_logs": poses, "instruction": "reach"})
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    datasets = [cls(folder) for cls in classes]
                for dataset in datasets:
                    expected = sum(6 if i == 0 or i >= frames - 5 else 1 for i in range(frames))
                    self.assertEqual(len(dataset), expected)
                    arrays = [sample["action"] for sample in dataset.samples]
                    self.assertEqual(len({id(a) for a in arrays}), len(arrays))
                    self.assertTrue(all(a.dtype == np.float64 and a.shape == (10, 4) for a in arrays))
                if len(datasets) == 2:
                    self.assertNestedEqual(datasets[0].samples, datasets[1].samples)
                    for key in ("action_min", "action_max", "action_mean", "action_std"):
                        self.assertNestedEqual(getattr(datasets[0], key), getattr(datasets[1], key))

    @unittest.skipUnless(REFERENCE, "optional pre-consolidation source snapshot")
    def test_optimizer_groups_schedule_and_updates_are_exact(self):
        source = (Path(REFERENCE) / "starVLA/training/train_starvla_vla.py").read_text()
        names = ("setup_optimizer_and_scheduler_lora", "setup_optimizer_and_scheduler")
        definitions = [
            n for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name in (*names, "_build_optimizer_and_scheduler")
        ]
        scope = dict(
            torch=torch,
            dist=torch.distributed,
            get_scheduler=get_scheduler,
            build_param_lr_groups=build_param_lr_groups,
            Tuple=tuple,
            logger=SimpleNamespace(info=lambda *_: None),
        )
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "<reference-optimizers>", "exec"), scope)
        for mode in names:
            states = []
            for factory in (scope[mode], getattr(trainer, mode)):
                torch.manual_seed(42)
                model = torch.nn.Module()
                model.qwen_vl_interface = torch.nn.Linear(4, 4)
                model.action_model = torch.nn.Linear(4, 1)
                cfg = OmegaConf.load("starVLA/config/training/qwen35_discrete.yaml")
                opt, schedule = factory(model, cfg)
                for i in range(4):
                    loss = (
                        model.action_model(model.qwen_vl_interface(torch.arange(8, dtype=torch.float32).reshape(2, 4)))
                        .square()
                        .mean()
                    )
                    loss.backward()
                    opt.step()
                    schedule.step()
                    opt.zero_grad()
                states.append((model.state_dict(), opt.state_dict(), schedule.state_dict()))
            self.assertNestedEqual(*states)


REPO = Path(__file__).resolve().parents[1]


def recipe():
    return OmegaConf.load(REPO / "starVLA/config/training/qwen35_discrete.yaml")


class ConfigurationContracts(unittest.TestCase):
    def test_import_does_not_initialize_accelerator(self):
        code = """from accelerate.state import AcceleratorState
import starVLA.training.train_starvla_vla
assert not AcceleratorState._shared_state, 'trainer import initialized distributed runtime'
"""
        run = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout)

    def test_deepspeed_uses_training_config_values(self):
        from starVLA.training import train_starvla_vla as training

        cfg = recipe()
        cfg.trainer.gradient_accumulation_steps = 3
        cfg.trainer.gradient_clipping = 0.25
        with patch.object(training, "Accelerator") as factory:
            training.create_accelerator(cfg)
        kw = factory.call_args.kwargs
        plugin = kw["deepspeed_plugin"]
        self.assertEqual(plugin.deepspeed_config["gradient_accumulation_steps"], 3)
        self.assertEqual(plugin.deepspeed_config["gradient_clipping"], 0.25)
        self.assertEqual(kw["gradient_accumulation_plugin"].num_steps, 3)
        self.assertFalse(kw["gradient_accumulation_plugin"].sync_with_dataloader)
        cfg.trainer.gradient_clipping = None
        with patch.object(training, "Accelerator") as factory:
            training.create_accelerator(cfg)
        self.assertEqual(factory.call_args.kwargs["deepspeed_plugin"].deepspeed_config["gradient_clipping"], 0.0)

    def test_incompatible_settings_fail_before_model_loading(self):
        from starVLA.training.train_starvla_vla import validate_training_config

        validate_training_config(recipe())
        changes = {
            "framework.action_model.past_action_window_size": 1,
            "framework.action_model.future_action_window_size": 4,
            "framework.action_model.action_dim": 7,
            "framework.action_model.action_head_type": "gaussian",
            "datasets.vla_data.image_size": [256, 256],
            "trainer.gradient_accumulation_steps": 0,
            "trainer.gradient_clipping": -1,
            "datasets.vla_data.num_workers": 1.5,
            "datasets.vla_data.per_device_batch_size": False,
            "trainer.optimizer.name": "SGD",
            "trainer.is_resume": True,
            "trainer.resume_step": 2,
            "trainer.num_warmup_steps": 100001,
            "trackers": ["typo"],
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                cfg = recipe()
                OmegaConf.update(cfg, key, value)
                with self.assertRaises(ValueError):
                    validate_training_config(cfg)

    def test_trackers_control_wandb_and_jsonl(self):
        from starVLA.training.train_starvla_vla import VLAMTrainer
        from types import SimpleNamespace

        trainer = VLAMTrainer.__new__(VLAMTrainer)
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer.completed_steps = 10
        trainer.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [0.0002])
        with tempfile.TemporaryDirectory() as folder:
            trainer.config = recipe()
            trainer.config.output_dir = folder
            trainer.config.trackers = ["jsonl"]
            with (
                patch("starVLA.training.train_starvla_vla.wandb") as wandb,
                patch("starVLA.training.train_starvla_vla.logger"),
            ):
                trainer._init_wandb()
                trainer._log_metrics({"action_loss": 1.25})
                wandb.init.assert_not_called()
                wandb.log.assert_not_called()
            row = json.loads((Path(folder) / "metrics.jsonl").read_text())
            self.assertEqual(row["step"], 10)
            self.assertEqual(row["learning_rate"], 0.0002)
            self.assertEqual(row["action_loss"], 1.25)

    def test_launcher_preserves_yaml_unless_env_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / "python"
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake.chmod(0o755)
            env = os.environ.copy()
            for key in ["BASE_VLM", "DATA_ROOT_DIR", "BATCH_SIZE", "RUN_ROOT_DIR"]:
                env.pop(key, None)
            env.update(PYTHON_BIN=str(fake), CONFIG_YAML="custom.yaml")
            launcher = str(REPO / "scripts/run_scripts/uav_train_qwen35_discrete.sh")
            args = subprocess.check_output(["bash", launcher], env=env, text=True).splitlines()
            self.assertIn("custom.yaml", args)
            self.assertNotIn("--datasets.vla_data.per_device_batch_size", args)
            self.assertNotIn("--framework.qwenvl.base_vlm", args)
            env.update(BATCH_SIZE="3", BASE_VLM="/custom/model")
            args = subprocess.check_output(["bash", launcher], env=env, text=True).splitlines()
            self.assertEqual(args[args.index("--datasets.vla_data.per_device_batch_size") + 1], "3")
            self.assertEqual(args[args.index("--framework.qwenvl.base_vlm") + 1], "/custom/model")

    def test_debug_reaches_uav_dataloader(self):
        from starVLA.dataloader import build_dataloader

        cfg = recipe()
        cfg.is_debug = True
        with patch("starVLA.dataloader.uav_datasets.make_uav_datasets", return_value="loader") as make:
            self.assertEqual(build_dataloader(cfg, "uav_datasets"), "loader")
        make.assert_called_once_with(cfg, is_debug=True)

    def test_qwen3_attention_setting_is_honored(self):
        from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
        from types import SimpleNamespace

        cfg = recipe()
        cfg.framework.qwenvl.model_type = "qwen3_vl"
        cfg.framework.qwenvl.attn_implementation = "sdpa"
        fake = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048)))
        with (
            patch(
                "starVLA.model.modules.vlm.QWen3.Qwen3VLForConditionalGeneration.from_pretrained", return_value=fake
            ) as load,
            patch("starVLA.model.modules.vlm.QWen3.AutoProcessor.from_pretrained"),
        ):
            _QWen3_VL_Interface(cfg)
        self.assertEqual(load.call_args.kwargs["attn_implementation"], "sdpa")

    def test_qwen3_gradient_checkpointing_switch_is_honored(self):
        from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
        from types import SimpleNamespace
        from unittest.mock import Mock

        for enabled in (False, True):
            cfg = recipe()
            cfg.trainer.enable_gradient_checkpointing = enabled
            fake = Mock(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048)))
            with (
                patch(
                    "starVLA.model.modules.vlm.QWen3.Qwen3VLForConditionalGeneration.from_pretrained",
                    return_value=fake,
                ),
                patch("starVLA.model.modules.vlm.QWen3.AutoProcessor.from_pretrained"),
            ):
                _QWen3_VL_Interface(cfg)
            self.assertEqual(fake.gradient_checkpointing_enable.called, enabled)
            self.assertEqual(fake.enable_input_require_grads.called, enabled)

    def test_explicit_backbone_type_wins_over_directory_name(self):
        from starVLA.model.modules.vlm import get_vlm_model

        cfg = recipe()
        cfg.framework.qwenvl.model_type = "qwen3_vl"
        # Moving old weights under a directory named Qwen3.5 must not change architecture.
        with patch("starVLA.model.modules.vlm.QWen3._QWen3_VL_Interface", return_value="qwen3"):
            self.assertEqual(get_vlm_model(cfg), "qwen3")
        cfg.framework.qwenvl.model_type = "typo"
        with self.assertRaises(NotImplementedError):
            get_vlm_model(cfg)

    def test_train_logs_one_average_per_optimizer_update(self):
        from starVLA.training.train_starvla_vla import VLAMTrainer
        from types import SimpleNamespace
        from unittest.mock import Mock

        trainer = VLAMTrainer.__new__(VLAMTrainer)
        trainer.config = recipe()
        trainer.config.trainer.max_train_steps = 2
        trainer.accelerator = SimpleNamespace(sync_gradients=False, is_local_main_process=False)
        trainer.completed_steps = 0
        trainer._log_training_config = Mock()
        trainer._create_data_iterators = Mock()
        trainer._get_next_batch = Mock(return_value=(None, None))
        trainer._finalize_training = Mock()
        trainer._save_checkpoint = Mock()
        trainer._log_metrics = Mock()
        values = iter([1.0, 3.0, 5.0, 7.0])

        def step(*_):
            value = next(values)
            trainer.accelerator.sync_gradients = value in (3.0, 7.0)
            return {"action_loss": value}

        trainer._train_step = step
        trainer.train()
        self.assertEqual([call.args[0]["action_loss"] for call in trainer._log_metrics.call_args_list], [2.0, 6.0])
        trainer._save_checkpoint.assert_not_called()


class AccumulationContract(unittest.TestCase):
    def test_two_microbatches_equal_full_batch_and_one_scheduler_step(self):
        # Fresh process isolates Accelerator's singleton from other test modules.
        code = """import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from omegaconf import OmegaConf
from starVLA.training.train_starvla_vla import VLAMTrainer
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
    def forward(self, batch):
        x,y=batch
        return {'action_loss': (self.weight*x-y).square().mean()}
a=Accelerator(cpu=True, gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=2,sync_with_dataloader=False))
m=Tiny(); ref=Tiny()
o=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=0)
r=torch.optim.AdamW(ref.parameters(),lr=.01,weight_decay=0)
s=torch.optim.lr_scheduler.LambdaLR(o, lambda step: 1/(step+1))
rs=torch.optim.lr_scheduler.LambdaLR(r, lambda step: 1/(step+1))
m,o=a.prepare(m,o)
cfg=OmegaConf.create({'datasets':{'vla_data':{'per_device_batch_size':1}},'trainer':{'gradient_clipping':None}})
t=VLAMTrainer(cfg,m,None,None,o,s,a)
for offset in [0.,1.]:
    for x,y in [(torch.tensor([1.+offset]),torch.tensor([2.])),(torch.tensor([3.+offset]),torch.tensor([-1.]))]:
        t._train_step((x,y),None)
    r.zero_grad()
    ref((torch.tensor([1.+offset,3.+offset]),torch.tensor([2.,-1.])))['action_loss'].backward()
    r.step(); rs.step()
    torch.testing.assert_close(a.unwrap_model(m).weight,ref.weight,rtol=0,atol=1e-7)
    assert s.last_epoch==rs.last_epoch, (s.last_epoch,rs.last_epoch)
print('ACCUMULATION_OK')
"""
        run = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout)


class FastProcessorCompatibility(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("FLOWVLA_FAST_TOKENIZER"), "external FAST tokenizer fixture")
    def test_transformers5_load_preserves_fast_encoding(self):
        from transformers import AutoTokenizer
        from starVLA.model.modules.action_model.fast_ActionHeader import Fast_Action_Tokenizer

        folder = os.environ["FLOWVLA_FAST_TOKENIZER"]
        wrapped = Fast_Action_Tokenizer(folder).fast_tokenizer
        config = json.loads((Path(folder) / "processor_config.json").read_text())
        reference = type(wrapped)(
            AutoTokenizer.from_pretrained(folder),
            **{key: config[key] for key in ("scale", "vocab_size", "min_token", "time_horizon", "action_dim")},
        )
        actions = np.random.default_rng(123).uniform(-1, 1, (2, 10, 4))
        actual = wrapped(actions)
        expected = reference(actions)
        self.assertEqual(actual, expected)
        self.assertTrue(all(len(row) > 0 for row in actual))
        np.testing.assert_array_equal(wrapped.decode(actual), reference.decode(expected))


class TrainingDataStatistics(unittest.TestCase):
    def test_constant_action_axis_is_rejected_before_training(self):
        from starVLA.dataloader.uav_datasets import make_uav_datasets
        with tempfile.TemporaryDirectory() as folder:
            episode = Path(folder) / "episode"
            episode.mkdir()
            poses = [[0, 0, 0, 0, 0, 0], [1, 2, 0, 0, 10, 0], [3, 4, 0, 0, 20, 0]]
            for i in range(3):
                Image.new("RGB", (16, 16)).save(episode / f"{i:06d}.png")
            (episode / "log.json").write_text(json.dumps({"raw_logs": poses, "preprocessed_logs": poses}))
            cfg = OmegaConf.create({"datasets": {"vla_data": {"data_root_dir": folder, "per_device_batch_size": 1, "num_workers": 0}}})
            with self.assertRaisesRegex(ValueError, "normalization"):
                make_uav_datasets(cfg)


class Qwen3HiddenStateCompatibility(unittest.TestCase):
    def test_legacy_decoder_features_with_deepstack_and_checkpointing(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface

        class TextLM(nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = decoder

            def forward(self, **kwargs):
                result = self.model.language_model(**kwargs)
                return CausalLMOutputWithPast(logits=result.last_hidden_state, hidden_states=result.hidden_states)

        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                torch.manual_seed(42)
                cfg = Qwen3VLTextConfig(
                    vocab_size=32, hidden_size=16, intermediate_size=24,
                    num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                    rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
                )
                cfg._attn_implementation = "eager"
                decoder = Qwen3VLTextModel(cfg).train()
                if checkpointing:
                    decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                wrapped = _QWen3_VL_Interface.__new__(_QWen3_VL_Interface)
                nn.Module.__init__(wrapped)
                wrapped.model = TextLM(decoder)
                expected = {}
                handles = []
                def save_input(key):
                    def capture(module, args):
                        expected[key] = args[0]
                    return capture
                for index in (1, 2, 3):
                    handles.append(decoder.layers[index].register_forward_pre_hook(save_input(index)))
                handles.append(decoder.norm.register_forward_pre_hook(save_input("last")))
                visual = [torch.randn(2, 16, requires_grad=True) for _ in range(3)]
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'.*")
                        outputs = wrapped(
                            inputs_embeds=torch.randn(1, 5, 16, requires_grad=True),
                            visual_pos_masks=torch.tensor([[False, True, True, False, False]]),
                            deepstack_visual_embeds=visual, use_cache=False, output_hidden_states=True,
                        )
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(len(outputs.hidden_states), 5)
                self.assertTrue(torch.equal(outputs.hidden_states[-1], expected["last"]))
                self.assertFalse(torch.equal(outputs.hidden_states[-1], outputs.logits))
                for index in (1, 2, 3):
                    self.assertTrue(torch.equal(outputs.hidden_states[index], expected[index]))
                outputs.hidden_states[-1].square().mean().backward()
                self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in visual))
                counts = [(len(layer._forward_pre_hooks), len(layer._forward_hooks)) for layer in decoder.layers]
                with patch.object(wrapped.model, "forward", side_effect=RuntimeError("test failure")):
                    with self.assertRaisesRegex(RuntimeError, "test failure"):
                        wrapped(output_hidden_states=True)
                self.assertEqual(counts, [(len(layer._forward_pre_hooks), len(layer._forward_hooks)) for layer in decoder.layers])


class DiscreteAllLayerRestoration(unittest.TestCase):
    @staticmethod
    def interface(**kwargs):
        module = nn.Module()
        module.model = nn.Module()
        module.model.config = SimpleNamespace(hidden_size=8, text_config=SimpleNamespace(num_hidden_layers=2))
        module.processor = SimpleNamespace(tokenizer=lambda *args, **kwargs: {"input_ids": [1]})
        return module

    def test_server_rejects_wrong_layer_configuration(self):
        from starVLA.model.framework import QwenOFT_discrete as framework
        from starVLA.model.framework.ServerQwenOFT_dis import QwenOFTServer

        cfg = OmegaConf.load(REPO / "starVLA/config/training/qwen3vl_discrete.yaml")
        cfg.trainer.use_lora = False
        with tempfile.TemporaryDirectory() as folder, patch.object(framework, "get_vlm_model", side_effect=self.interface), patch("torch.cuda.is_available", return_value=False):
            folder = Path(folder)
            stats = folder / "stats.json"
            stats.write_text(json.dumps({"action_min": [-1] * 4, "action_max": [1] * 4}))
            checkpoints = {}
            for layered in (False, True):
                cfg.framework.use_all_layer_hidden = layered
                model = framework.Qwenvl_OFT(cfg)
                path = folder / f"weights_{layered}.pt"
                torch.save(model.state_dict(), path)
                checkpoints[layered] = path
            for layered in (False, True):
                cfg.framework.use_all_layer_hidden = layered
                config = folder / f"config_{layered}.yaml"
                OmegaConf.save(cfg, config)
                args = dict(config_yaml=str(config), base_vlm="unused", checkpoint_path=str(checkpoints[not layered]), stats_path=str(stats))
                with self.assertRaisesRegex(RuntimeError, "layer_weights"):
                    QwenOFTServer(args)
                args["checkpoint_path"] = str(checkpoints[layered])
                server = QwenOFTServer(args)
                self.assertEqual(server.model.use_all_layer_hidden, layered)


    def test_restored_head_shapes_losses_and_continuous_route(self):
        from starVLA.model.framework import QwenOFT_discrete as framework
        from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model, DiscreteActionHead, L1RegressionActionHead

        for layered in (False, True):
            name = "qwen3vl_discrete" + ("_alllayer" if layered else "") + ".yaml"
            cfg = OmegaConf.load(REPO / "starVLA/config/training" / name)
            self.assertEqual(cfg.framework.action_model.action_head_type, "discrete")
            self.assertEqual(cfg.framework.action_model.num_action_bins, 256)
            self.assertEqual(cfg.framework.use_all_layer_hidden, layered)
            trainer.validate_training_config(cfg)
            with patch.object(framework, "get_vlm_model", side_effect=self.interface):
                model = framework.Qwenvl_OFT(cfg)
            self.assertIsInstance(model.action_model, DiscreteActionHead)
            logits = model.action_model.predict_action(torch.randn(2, 10, 8))
            self.assertEqual(tuple(logits.shape), (2, 10, 4, 256))
            loss = model.ce_loss(logits.reshape(-1, 256), torch.zeros(80, dtype=torch.long))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.action_model.parameters()))
        cfg = OmegaConf.load(REPO / "starVLA/config/training/qwen3vl_continuous.yaml")
        cfg.framework.action_model.action_hidden_dim = 8
        head = get_action_model(cfg)
        self.assertIsInstance(head, L1RegressionActionHead)
        self.assertEqual(tuple(head.predict_action(torch.randn(2, 10, 8)).shape), (2, 10, 4))
        cfg.framework.action_model.action_head_type = "discrete"
        with self.assertRaisesRegex(ValueError, "Continuous"):
            trainer.validate_training_config(cfg)

    def test_alllayer_launcher_preserves_baseline_warm_start(self):
        import argparse
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / "python"
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake.chmod(0o755)
            env = os.environ.copy()
            for key in ["CONFIG_YAML", "BATCH_SIZE"]:
                env.pop(key, None)
            env.update(PYTHON_BIN=str(fake), PRETRAINED_CHECKPOINT="/baseline/model.pt", BASE_VLM="/custom/model", DATA_ROOT_DIR="/custom/data", NUM_PROCESSES="2")
            args = subprocess.check_output(["bash", str(REPO / "scripts/run_scripts/uav_train_qwen3vl_discrete_alllayer.sh")], env=env, text=True).splitlines()
            self.assertEqual(args[args.index("--num_processes") + 1], "2")
            parser = argparse.ArgumentParser()
            parser.add_argument("--config_yaml")
            parsed, extra = parser.parse_known_args(args[args.index("starVLA/training/train_starvla_vla.py") + 1:])
            cfg = OmegaConf.merge(OmegaConf.load(REPO / parsed.config_yaml), OmegaConf.from_dotlist(trainer.normalize_dotlist_args(extra)))
            self.assertTrue(cfg.framework.use_all_layer_hidden)
            self.assertEqual(cfg.framework.action_model.action_head_type, "discrete")
            self.assertEqual(cfg.trainer.pretrained_checkpoint, "/baseline/model.pt")
            self.assertEqual(cfg.trainer.reload_modules, "qwen_vl_interface,action_model")
            self.assertEqual(cfg.framework.qwenvl.base_vlm, "/custom/model")
            trainer.validate_training_config(cfg)

    def test_server_rejects_wrong_head_and_expected_layer_flag(self):
        from starVLA.model.framework.ServerQwenOFT_dis import QwenOFTServer
        cfg = OmegaConf.load(REPO / "starVLA/config/training/qwen3vl_discrete.yaml")
        with tempfile.TemporaryDirectory() as folder, patch("torch.cuda.is_available", return_value=False):
            path = Path(folder) / "config.yaml"
            OmegaConf.save(cfg, path)
            args = dict(config_yaml=str(path), base_vlm="unused", checkpoint_path="unused", expected_all_layers=True)
            with self.assertRaisesRegex(ValueError, "use_all_layer_hidden"):
                QwenOFTServer(args)
            cfg.framework.action_model.action_head_type = "continuous"
            OmegaConf.save(cfg, path)
            with self.assertRaisesRegex(ValueError, "discrete"):
                QwenOFTServer(args)
            cfg.framework.action_model.action_head_type = "discrete"
            cfg.framework.qwenvl.model_type = "qwen3_5"
            OmegaConf.save(cfg, path)
            with self.assertRaisesRegex(ValueError, "Qwen3.5 server"):
                QwenOFTServer(args)


class GenericEntrypointTests(unittest.TestCase):
    def test_all_training_launchers_preserve_cli_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / "python"
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake.chmod(0o755)
            for name in [*RECIPES, "qwen35_discrete"]:
                with self.subTest(name=name):
                    env = os.environ.copy()
                    for key in ["CONFIG_YAML"]:
                        env.pop(key, None)
                    env.update(PYTHON_BIN=str(fake), PRETRAINED_CHECKPOINT="/baseline/model.pt", DATA_ROOT_DIR="/custom/episodes", BASE_VLM="/custom/model", BATCH_SIZE="3", NUM_PROCESSES="2", STATS_PATH="/custom/stats.json")
                    args = subprocess.check_output(["bash", str(REPO / f"scripts/run_scripts/uav_train_{name}.sh"), "--datasets.vla_data.per_device_batch_size", "5"], env=env, text=True).splitlines()
                    self.assertEqual(args[args.index("--num_processes") + 1], "2")
                    i = args.index("--config_yaml")
                    cfg = OmegaConf.merge(OmegaConf.load(REPO / args[i + 1]), OmegaConf.from_dotlist(trainer.normalize_dotlist_args(args[i + 2:])))
                    self.assertEqual(cfg.datasets.vla_data.per_device_batch_size, 5)
                    self.assertEqual(cfg.datasets.vla_data.data_root_dir, "/custom/episodes")
                    self.assertEqual(cfg.datasets.vla_data.normalization_stats_path, "/custom/stats.json")
                    if name == "qwen35_discrete":
                        self.assertNotIn("manifest_path", cfg.framework.action_codec)
                        self.assertIsNone(cfg.framework.action_codec.sha256)
                    if name.endswith("alllayer"):
                        self.assertEqual(cfg.trainer.reload_modules, "qwen_vl_interface,action_model")

    def test_server_launchers_use_explicit_stats_and_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            run = folder / "old-run"
            (run / "checkpoints").mkdir(parents=True)
            for name in ["config.yaml", "checkpoints/steps_20_pytorch_model.pt", "checkpoints/steps_100_pytorch_model.pt"]:
                (run / name).touch()
            stats = folder / "dataset-normalization.json"
            stats.touch()
            fake = folder / "python"
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$SERVER_MODULE" "$CONFIG_YAML" "$STATS_PATH" "$CHECKPOINT_PATH" "${EXPECTED_ALL_LAYERS:-unset}"\n')
            fake.chmod(0o755)
            for name, module in [("qwen3vl_discrete", "ServerQwenOFT_dis"), ("qwen3vl_discrete_alllayer", "ServerQwenOFT_dis"), ("qwen3vl_continuous", "ServerQwenOFT_l1"), ("qwen3vl_fast", "ServerQwenFast"), ("qwen35_discrete", "ServerQwen35OFT")]:
                env = os.environ.copy()
                for key in ["CONFIG_YAML", "STATS_PATH", "CHECKPOINT_PATH", "EXPECTED_ALL_LAYERS"]:
                    env.pop(key, None)
                env.update(RUN_DIR=str(run), PYTHON_BIN=str(fake), STATS_PATH=str(stats))
                result = subprocess.check_output(["bash", str(REPO / f"scripts/run_scripts/start_{name}_server.sh")], env=env, text=True).splitlines()
                self.assertTrue(result[0].endswith(module))
                self.assertEqual(result[1], str(run / "config.yaml"))
                self.assertEqual(result[2], str(stats))
                self.assertEqual(result[3], str(run / "checkpoints/steps_100_pytorch_model.pt"))
                self.assertFalse((run / "uav_action_statistics.json").exists())
                if name.startswith("qwen3vl_discrete"):
                    self.assertEqual(result[4], "1" if name.endswith("alllayer") else "0")
                env.pop("STATS_PATH")
                missing = subprocess.run(["bash", str(REPO / f"scripts/run_scripts/start_{name}_server.sh")], env=env, text=True, capture_output=True)
                self.assertNotEqual(missing.returncode, 0)
                self.assertIn("Set STATS_PATH", missing.stderr)


class FastStatisticsRegression(unittest.TestCase):
    def test_selected_statistics_control_fast_denormalization(self):
        from flask import Flask
        from starVLA.model.framework.ServerQwenFast import QwenFastServer

        class Model:
            def predict_action(self, *args):
                return {"normalized_actions": np.zeros((1, 10, 4), dtype=np.float32)}

        server = QwenFastServer.__new__(QwenFastServer)
        server.model = Model()
        server.action_min = np.array([-1, -2, -3, -0.1], dtype=np.float32)
        server.action_max = np.array([3, 6, 9, 0.1], dtype=np.float32)
        server.init_img = None
        server.app = Flask(__name__)
        server._routes()
        result = server.app.test_client().post("/predict", json=dict(image=encoded("red"), instr="go", proprio=[0, 0, 0, 0]))
        self.assertEqual(result.status_code, 200)
        expected = np.arange(1, 11)[:, None] * np.array([1, 2, 3, 0])[None, :]
        np.testing.assert_array_equal(result.get_json()["action"], expected)

    def test_legacy_server_stats_reject_nonfinite_values(self):
        from starVLA.model.framework.ServerQwenOFT_l1 import _extract_action_range

        with self.assertRaisesRegex(ValueError, "finite"):
            _extract_action_range(dict(action_min=[float("nan")] * 4, action_max=[1] * 4))


class PortableInferenceConfigTests(unittest.TestCase):
    def test_grounding_catalog_accepts_external_roots(self):
        path = REPO / "starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py"
        spec = importlib.util.spec_from_file_location("_portable_grounding_config", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"COCO_DATA_ROOT": "/example/coco", "LLAVA_FORMAT_ROOT": "/example/annotations"}):
            spec.loader.exec_module(module)
        self.assertEqual(module.asv2_conversation_en["data_path"], "/example/coco")
        self.assertEqual(module.asv2_conversation_en["annotation_path"], "/example/annotations/asv2_conversation_en.jsonl")

    def test_continuous_requires_explicit_stats_and_accepts_custom_values(self):
        from starVLA.model.framework.ServerQwenOFT_l1 import QwenOFTL1Server

        server = QwenOFTL1Server.__new__(QwenOFTL1Server)
        with self.assertRaisesRegex(ValueError, "STATS_PATH"):
            server._load_action_stats(None)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "custom.json"
            path.write_text(json.dumps(dict(action_min=[-2] * 4, action_max=[3] * 4)))
            low, high = server._load_action_stats(str(path))
            np.testing.assert_array_equal(low, [-2] * 4)
            np.testing.assert_array_equal(high, [3] * 4)

    def test_fast_requires_stats_before_model_initialization(self):
        from starVLA.model.framework.ServerQwenFast import QwenFastServer

        with self.assertRaisesRegex(ValueError, "STATS_PATH"):
            QwenFastServer({})

    def test_continuous_runtime_does_not_select_an_old_experiment(self):
        from starVLA.model.framework.ServerQwenOFT_l1 import build_runtime_config

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(KeyError):
                build_runtime_config()
        with patch.dict(os.environ, {"RUN_DIR": "/example/run", "STATS_PATH": "/example/stats.json"}, clear=True):
            cfg = build_runtime_config()
        self.assertEqual(cfg["run_dir"], Path("/example/run"))
        self.assertEqual(cfg["config_yaml"], "/example/run/config.yaml")
        self.assertIsNone(cfg["base_vlm"])

    def test_continuous_base_override_preserves_hub_identifiers(self):
        from starVLA.model.framework.ServerQwenOFT_l1 import QwenOFTL1Server

        server = QwenOFTL1Server.__new__(QwenOFTL1Server)
        cfg = server._load_config(dict(
            config_yaml=str(REPO / "starVLA/config/training/qwen3vl_continuous.yaml"),
            base_vlm="Qwen/Qwen3-VL-2B-Instruct",
        ))
        self.assertEqual(cfg.framework.qwenvl.base_vlm, "Qwen/Qwen3-VL-2B-Instruct")

    def test_discrete_runtime_does_not_override_the_training_dataset(self):
        from starVLA.model.framework import ServerQwenOFT_dis as module

        with patch.dict(os.environ, {"CHECKPOINT_PATH": "/example/model.pt"}, clear=True), patch.object(module, "QwenOFTServer") as server:
            module.main()
        self.assertIsNone(server.call_args.args[0]["train_data_root"])

    def test_qwen35_requires_an_explicit_checkpoint(self):
        from starVLA.model.framework import ServerQwen35OFT as module

        with patch.dict(os.environ, {"RUN_DIR": "/example/run"}, clear=True), patch.object(module, "Qwen35OFTServer") as server:
            with self.assertRaisesRegex(ValueError, "CHECKPOINT_PATH"):
                module.main()
        server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
