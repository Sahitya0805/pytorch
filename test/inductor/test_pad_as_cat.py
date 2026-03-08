# Owner(s): ["module: inductor"]
"""
Tests for pad-as-cat and cat multi-consumer optimizations.

When cat/pad inputs have multiple consumers, pointwise_cat inlines the input
computation causing duplicate work. These tests verify that ConcatKernel's
realize-into-slice zero-copy mechanism is used instead.

References:
  - pytorch#125075: cat + to_fp16 fusion
  - vllm#24917 / vllm#25477: mul + pad + addmm fusion
"""

import re
import unittest

import torch
from torch._dynamo.utils import counters
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


def _count_triton_kernels(code: str) -> int:
    """Count the number of triton kernel definitions in generated code."""
    return len(re.findall(r"def triton_\w+\(", code))


class TestPadAsCat(TestCase):
    """Tests for pad→ConcatKernel and cat multi-consumer optimizations."""

    # ─── Pattern 1: mul + pad + addmm (vllm#24917) ────────────────────

    @requires_gpu()
    def test_mul_pad_addmm(self):
        """F.pad with multi-consumer input should use ConcatKernel zero-copy."""
        counters.clear()

        def fn(x, scale, bias, weight):
            mul_result = x * scale
            padded = torch.nn.functional.pad(mul_result, [0, 192])
            mm_result = torch.addmm(bias, mul_result, weight)
            return padded, mm_result

        x = torch.randn(4096, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        scale = torch.randn(4096, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        bias = torch.randn(1024, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(2880, 1024, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x, scale, bias, weight)
        ref = fn(x, scale, bias, weight)

        # Correctness
        self.assertTrue(torch.allclose(result[0], ref[0]))
        self.assertTrue(torch.allclose(result[1], ref[1], atol=1e-2, rtol=1e-2))
        # ConcatKernel uses reinterpret_tensor for the slice — sign of zero-copy
        self.assertIn("reinterpret_tensor", code)
        # pad_as_cat counter should fire
        self.assertGreater(counters["inductor"]["pad_as_cat"], 0)

    # ─── Pattern 2: cat + to_fp16 (pytorch#125075) ────────────────────

    @requires_gpu()
    def test_cat_to_fp16(self):
        """cat with multi-consumer input should fuse, not duplicate computation."""

        def fn(x):
            z = torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)
            y = x.to(torch.float16)
            return z, y

        x = torch.randn(30522, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x)
        ref = fn(x)

        # Correctness
        self.assertTrue(torch.allclose(result[0], ref[0]))
        self.assertTrue(torch.allclose(result[1], ref[1]))
        # With ConcatKernel, x's copy and to_fp16 should fuse into 1 kernel
        # plus a tiny fill kernel for the 6*768 zeros
        kernel_count = _count_triton_kernels(code)
        self.assertLessEqual(
            kernel_count,
            2,
            f"Expected at most 2 kernels (fused + fill), got {kernel_count}.",
        )

    # ─── Single consumer unchanged ─────────────────────────────────────

    @requires_gpu()
    def test_single_consumer_pad_unchanged(self):
        """Single-consumer F.pad should NOT use _pad_as_cat (default is optimal)."""
        counters.clear()

        def fn(x, scale):
            return torch.nn.functional.pad(x * scale, [0, 192])

        x = torch.randn(4096, 2880, device=GPU_TYPE)
        scale = torch.randn(4096, 2880, device=GPU_TYPE)

        compiled = torch.compile(fn)
        result = compiled(x, scale)
        ref = fn(x, scale)

        self.assertTrue(torch.allclose(result, ref))
        self.assertEqual(counters["inductor"]["pad_as_cat"], 0)

    @requires_gpu()
    def test_single_consumer_cat_unchanged(self):
        """Single-consumer cat should use pointwise_cat (no behavior change)."""

        def fn(x):
            return torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)

        x = torch.randn(30522, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        result = compiled(x)
        ref = fn(x)

        self.assertTrue(torch.allclose(result, ref))


if __name__ == "__main__":
    unittest.main()
