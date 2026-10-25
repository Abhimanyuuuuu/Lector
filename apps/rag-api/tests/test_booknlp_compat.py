"""Focused compatibility tests for BookNLP's legacy BERT checkpoint key."""

from __future__ import annotations

import unittest

import torch

from lector_rag_api.booknlp_compat import (
    LEGACY_POSITION_IDS_KEY,
    booknlp_state_dict_compatibility,
)


class _TargetModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


class _EntityTagger(_TargetModel):
    pass


class _CoreferenceTagger(_TargetModel):
    pass


class _SpeakerTagger(_TargetModel):
    pass


class BookNLPStateDictCompatibilityTests(unittest.TestCase):
    def test_base_module_compatibility_covers_all_booknlp_component_shapes(self):
        original_loader = torch.nn.Module.load_state_dict
        checkpoint = {
            "weight": torch.tensor([2.0]),
            LEGACY_POSITION_IDS_KEY: torch.arange(512).reshape(1, 512),
        }

        with booknlp_state_dict_compatibility():
            for component in (_EntityTagger, _CoreferenceTagger, _SpeakerTagger):
                with self.subTest(component=component.__name__):
                    result = component().load_state_dict(checkpoint)
                    self.assertEqual(result.missing_keys, [])
                    self.assertEqual(result.unexpected_keys, [])

        self.assertIs(torch.nn.Module.load_state_dict, original_loader)

    def test_only_legacy_position_ids_key_is_removed_and_strict_load_succeeds(self):
        model = _TargetModel()
        checkpoint = {
            "weight": torch.tensor([2.0]),
            LEGACY_POSITION_IDS_KEY: torch.arange(512).reshape(1, 512),
        }
        original_keys = set(checkpoint)

        with booknlp_state_dict_compatibility(_TargetModel):
            result = model.load_state_dict(checkpoint)

        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(model.weight.item(), 2.0)
        self.assertEqual(set(checkpoint), original_keys)
        self.assertNotIn("load_state_dict", _TargetModel.__dict__)

    def test_unrelated_unexpected_key_still_fails_strict_loading(self):
        model = _TargetModel()
        checkpoint = {
            "weight": torch.tensor([2.0]),
            LEGACY_POSITION_IDS_KEY: torch.arange(512),
            "unexpected.weight": torch.tensor([3.0]),
        }

        with booknlp_state_dict_compatibility(_TargetModel):
            with self.assertRaisesRegex(RuntimeError, "unexpected.weight"):
                model.load_state_dict(checkpoint)

    def test_missing_required_key_still_fails_strict_loading(self):
        model = _TargetModel()
        checkpoint = {LEGACY_POSITION_IDS_KEY: torch.arange(512)}

        with booknlp_state_dict_compatibility(_TargetModel):
            with self.assertRaisesRegex(RuntimeError, "weight"):
                model.load_state_dict(checkpoint)


if __name__ == "__main__":
    unittest.main()
