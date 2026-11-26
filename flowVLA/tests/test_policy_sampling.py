"""Action sampling honors the deterministic validation switch without a GPU."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from starVLA.model.framework.QwenOFT_discrete import Qwenvl_OFT


class PolicySamplingTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.linspace(-2, 2, 256).repeat(2, 10, 4, 1)
        self.policy = SimpleNamespace(
            action_model=SimpleNamespace(predict_action=lambda queries: self.logits),
            _encode_action_queries=lambda inputs, policy: None,
            discrete_to_continuous=lambda bins: bins.float() / 255 * 2 - 1,
        )
        self.inputs = {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "pixel_values": torch.zeros(2, 3),
        }

    def sample(self, **kwargs):
        # Exercise the actual model method; only GPU autocast and the backbone are stubbed.
        with patch("torch.autocast", return_value=nullcontext()):
            return Qwenvl_OFT.sample_action_from_inputs(self.policy, **self.inputs, **kwargs)

    def test_greedy_never_calls_multinomial_and_is_seed_independent(self):
        with patch("torch.multinomial", side_effect=AssertionError("greedy path sampled")):
            torch.manual_seed(1)
            first = self.sample(do_sample=False, temperature=1.6)
            torch.manual_seed(999)
            second = self.sample(do_sample=False, temperature=0.5)
        expected = self.policy.discrete_to_continuous(self.logits.argmax(dim=-1))
        self.assertTrue(torch.equal(first["sampled_actions"], expected))
        self.assertTrue(torch.equal(second["sampled_actions"], expected))
        self.assertEqual(first["sampled_actions"].shape, (2, 10, 4))
        self.assertTrue(torch.equal(first["action_logits"], self.logits))

    def test_default_and_explicit_sampling_keep_previous_temperature_behavior(self):
        probabilities = F.softmax((self.logits / 1.6).view(-1, 256), dim=-1)
        torch.manual_seed(42)
        expected_bins = torch.multinomial(probabilities, 1).view(2, 10, 4)
        expected = self.policy.discrete_to_continuous(expected_bins)
        for kwargs in ({}, {"do_sample": True}):
            torch.manual_seed(42)
            actual = self.sample(temperature=1.6, **kwargs)
            self.assertTrue(torch.equal(actual["sampled_actions"], expected))
            self.assertTrue(torch.equal(actual["action_logits"], self.logits))

    def test_random_sampling_remains_seed_dependent(self):
        torch.manual_seed(1)
        first = self.sample(do_sample=True)["sampled_actions"]
        torch.manual_seed(999)
        second = self.sample(do_sample=True)["sampled_actions"]
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
