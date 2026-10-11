from __future__ import annotations

import unittest

import torch

from specfit.runtime.numeric_modes import (
    NUMERIC_MODES,
    apply_numeric_mode,
    current_numeric_settings,
    numeric_mode,
)


class NumericModesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_settings = current_numeric_settings()

    def tearDown(self) -> None:
        torch.backends.cuda.enable_flash_sdp(
            self.initial_settings["flash_sdp_enabled"]
        )
        torch.backends.cuda.enable_mem_efficient_sdp(
            self.initial_settings["memory_efficient_sdp_enabled"]
        )
        torch.backends.cuda.enable_math_sdp(
            self.initial_settings["math_sdp_enabled"]
        )
        torch.backends.cuda.enable_cudnn_sdp(
            self.initial_settings["cudnn_sdp_enabled"]
        )
        torch.backends.cuda.matmul.fp32_precision = self.initial_settings[
            "cuda_matmul_fp32_precision"
        ]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            self.initial_settings[
                "allow_bf16_reduced_precision_reduction"
            ]
        )
        torch.use_deterministic_algorithms(
            self.initial_settings["deterministic_algorithms"],
            warn_only=self.initial_settings["deterministic_warn_only"],
        )

    def test_supported_modes_match_diagnostic_contract(self) -> None:
        self.assertEqual(
            NUMERIC_MODES,
            (
                "default",
                "math_sdpa",
                "bf16_full_reduction",
                "deterministic",
                "math_ieee",
                "math_bf16_full_reduction",
            ),
        )

    def test_each_temporary_mode_records_settings_and_restores_state(
        self,
    ) -> None:
        for mode in NUMERIC_MODES:
            with self.subTest(mode=mode):
                with numeric_mode(mode) as actual:
                    expected = dict(self.initial_settings)
                    if mode in {
                        "math_sdpa",
                        "math_ieee",
                        "math_bf16_full_reduction",
                    }:
                        expected.update(
                            {
                                "forced_sdpa_backend": "math",
                                "flash_sdp_enabled": False,
                                "memory_efficient_sdp_enabled": False,
                                "math_sdp_enabled": True,
                                "cudnn_sdp_enabled": False,
                            }
                        )
                    if mode == "math_ieee":
                        expected["cuda_matmul_fp32_precision"] = "ieee"
                    if mode in {
                        "bf16_full_reduction",
                        "math_bf16_full_reduction",
                    }:
                        expected[
                            "allow_bf16_reduced_precision_reduction"
                        ] = False
                    if mode == "deterministic":
                        expected["deterministic_algorithms"] = True
                        expected["deterministic_warn_only"] = False

                    self.assertEqual(actual, expected)

                self.assertEqual(
                    current_numeric_settings(),
                    self.initial_settings,
                )

    def test_temporary_mode_restores_state_after_exception(self) -> None:
        for mode in NUMERIC_MODES:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    with numeric_mode(mode):
                        raise RuntimeError("fixture failure")

                self.assertEqual(
                    current_numeric_settings(),
                    self.initial_settings,
                )

    def test_process_handle_keeps_mode_until_explicit_restore(self) -> None:
        handle = apply_numeric_mode("bf16_full_reduction")
        self.assertFalse(handle.restored)
        self.assertFalse(handle.state_restored)
        self.assertFalse(
            current_numeric_settings()[
                "allow_bf16_reduced_precision_reduction"
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "must be restored"):
            handle.artifact(scope="unit-test")

        settings_after = handle.restore()
        self.assertEqual(settings_after, self.initial_settings)
        self.assertTrue(handle.restored)
        self.assertTrue(handle.state_restored)
        self.assertEqual(handle.restore(), self.initial_settings)
        self.assertEqual(
            handle.artifact(scope="unit-test"),
            {
                "name": "bf16_full_reduction",
                "scope": "unit-test",
                "settings_before": self.initial_settings,
                "actual_settings": {
                    **self.initial_settings,
                    "allow_bf16_reduced_precision_reduction": False,
                },
                "settings_after": self.initial_settings,
                "state_restored": True,
            },
        )

    def test_unsupported_mode_does_not_change_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported numeric mode"):
            apply_numeric_mode("unsupported")
        self.assertEqual(current_numeric_settings(), self.initial_settings)


if __name__ == "__main__":
    unittest.main()
