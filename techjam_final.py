#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every output element:
    abs(user - ref) <= atol
    OR
    abs(user - ref) <= rtol * abs(ref)

Final two-file runner: python techjam_final.py --all --output-dir results/RUN.
The default thresholds are atol=0.002 and rtol=0.02 (2%).

Shapes 1-13 use the supplied baseline, random generator, accuracy checker and
CUDA-event timing method. Shape 14 uses a separately labelled, independent
query-tiled reference at the FULL published dimensions. It does NOT have an
original-baseline speedup: that reference would need a 20.48 TB score tensor.

Optimizations: V4/V5 packed QKV + static compile for ordinary unpadded shapes;
fused-only SDPA with batch microbatching for long sequences. Inference-only
caches track tensor versions; padded/gradient-enabled calls use a correct
reference path. No fixed-input output caching or precision substitution.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class BaselineSelfAttention(nn.Module):
    """Explicit multi-head self-attention implemented with native PyTorch ops."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            # Mask invalid key positions. Shape: [B, 1, 1, S].
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        # Computing softmax in fp32 provides a stable reference for fp16/bf16 tests.
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


def tiled_reference_attention(attention, x, valid_token_mask, causal, query_block):
    """Original attention formula, scheduled by query tile, NOT using SDPA.

    Separate original Q/K/V projections and fp32 softmax are intentional:
    this reference must not reuse the candidate's packed projections or SDPA.
    Causal query indices are GLOBAL, including at nonzero tile offsets.
    """
    batch, seq_len, _ = x.shape
    q = attention._split_heads(attention.q_proj(x))
    k = attention._split_heads(attention.k_proj(x))
    v = attention._split_heads(attention.v_proj(x))
    context = torch.empty_like(q)
    for start in range(0, seq_len, query_block):
        stop = min(start + query_block, seq_len)
        key_stop = stop if causal else seq_len
        scores = torch.matmul(q[:, :, start:stop], k[:, :, :key_stop].transpose(-2, -1))
        scores.mul_(attention.scale)
        if causal:
            future = torch.arange(key_stop, device=x.device)[None, :] > torch.arange(
                start, stop, device=x.device
            )[:, None]
            scores.masked_fill_(future, float("-inf"))
        if valid_token_mask is not None:
            scores.masked_fill_(~valid_token_mask[:, None, None, :key_stop], float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context[:, :, start:stop] = torch.matmul(probs, v[:, :, :key_stop])
        del scores, probs
    context = context.transpose(1, 2).contiguous().view(batch, seq_len, attention.d_model)
    output = attention.out_proj(context)
    if valid_token_mask is not None:
        output = output.masked_fill(~valid_token_mask[..., None], 0)
    return output


class TiledReferenceAttention(BaselineSelfAttention):
    def __init__(self, d_model, num_heads, query_block):
        super().__init__(d_model, num_heads)
        self.query_block = query_block

    def forward(self, x, valid_token_mask=None, causal=False):
        return tiled_reference_attention(self, x, valid_token_mask, causal, self.query_block)


class TiledReferenceTransformer(BaselineTransformer):
    """Independent full-shape reference. NOT the unmodified official baseline."""
    def __init__(self, config, query_block=256):
        super().__init__(config)
        if query_block <= 0:
            raise ValueError("query_block must be positive")
        for layer in self.layers:
            layer.attention = TiledReferenceAttention(config.d_model, config.num_heads, query_block)
        self.progress = False

    def forward(self, x, valid_token_mask=None):
        output = torch.empty_like(x)
        for sample in range(x.size(0)):
            mask = None if valid_token_mask is None else valid_token_mask[sample:sample + 1]
            output[sample:sample + 1].copy_(super().forward(x[sample:sample + 1], mask))
            if self.progress:
                if x.is_cuda:
                    torch.cuda.synchronize(x.device)
                print(f"  tiled reference: sample {sample + 1}/{x.size(0)} completed", flush=True)
        return output


def tensor_version(tensor):
    # Inference tensors do not have version counters. Never cache their values
    # by identity alone: in-place mutation would otherwise be invisible.
    try:
        return tensor._version
    except RuntimeError:
        return None


class UserOptimizedTransformer(BaselineTransformer):
    """Measured V5 strategy with cache/masking safety fixes for final validation.

    Parameter names, input signature and output shape match the reference.
    Long sequences remain eager; only their fused SDPA kernel is optimized.
    The reference implementation above is validation/support code, not used in
    the measured all-valid optimized path.
    """
    def __init__(self, config, compile_mode="reduce-overhead", fullgraph=False,
                 long_seq_threshold=32768, long_seq_microbatch=1, query_block=256):
        super().__init__(config)
        if min(long_seq_threshold, long_seq_microbatch, query_block) <= 0:
            raise ValueError("threshold, microbatch and query block must be positive")
        self.compile_mode = compile_mode
        self.long_seq_threshold = long_seq_threshold
        self.long_seq_microbatch = long_seq_microbatch
        self.query_block = query_block
        self._packed_key = None
        self._mask_tensor = None
        self._mask_version = None
        self._mask_all_valid = False
        self._fast_forward = self._unpadded_forward if compile_mode == "off" else torch.compile(
            self._unpadded_forward, mode=compile_mode, dynamic=False, fullgraph=fullgraph
        )

    def _ensure_packed_qkv(self):
        parameters = [p for layer in self.layers
                      for proj in (layer.attention.q_proj, layer.attention.k_proj, layer.attention.v_proj)
                      for p in (proj.weight, proj.bias)]
        versions = [tensor_version(p) for p in parameters]
        key = tuple((id(p), version) for p, version in zip(parameters, versions))
        if all(v is not None for v in versions) and key == self._packed_key:
            return
        with torch.inference_mode(False), torch.no_grad():
            for layer in self.layers:
                a = layer.attention
                weights = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
                biases = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias])
                if hasattr(a, "qkv_weight_packed"):
                    # Preserve buffer storage when possible: compiled graphs may
                    # refer to these buffers after a same-device weight reload.
                    a.qkv_weight_packed.copy_(weights)
                    a.qkv_bias_packed.copy_(biases)
                else:
                    a.register_buffer("qkv_weight_packed", weights, persistent=False)
                    a.register_buffer("qkv_bias_packed", biases, persistent=False)
        self._packed_key = key if all(v is not None for v in versions) else None

    def _apply(self, fn, recurse=True):
        # Module.to()/dtype conversions can replace storage without increasing
        # parameter version counters, so explicitly invalidate derived buffers.
        self._packed_key = None
        self._mask_tensor = None
        self._mask_version = None
        return super()._apply(fn, recurse=recurse)

    def _all_tokens_valid(self, mask):
        if mask is None:
            return True
        version = tensor_version(mask)
        if version is not None and mask is self._mask_tensor and version == self._mask_version:
            return self._mask_all_valid
        result = bool(mask.all())
        if version is not None:
            self._mask_tensor, self._mask_version, self._mask_all_valid = mask, version, result
        return result

    @staticmethod
    def _sdpa_attention(attention, x, causal):
        batch, seq_len, _ = x.shape
        qkv = F.linear(x, attention.qkv_weight_packed, attention.qkv_bias_packed)
        q, k, v = qkv.view(batch, seq_len, 3, attention.num_heads, attention.head_dim).permute(
            2, 0, 3, 1, 4
        ).unbind(0)
        context = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=causal, scale=attention.scale
        )
        context = context.transpose(1, 2).reshape(batch, seq_len, attention.d_model)
        return attention.out_proj(context)

    def _unpadded_forward(self, x):
        for layer in self.layers:
            x = x + self._sdpa_attention(layer.attention, layer.norm1(x), self.config.causal)
            x = x + layer.ffn_out(F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none"))
        return self.final_norm(x)

    def _long_forward(self, x):
        # No math fallback: even B=1 would require 640 GB for S=100000 scores.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                          SDPBackend.CUDNN_ATTENTION]):
            if x.size(0) <= self.long_seq_microbatch:
                return self._unpadded_forward(x)
            output = torch.empty_like(x)
            for start in range(0, x.size(0), self.long_seq_microbatch):
                stop = min(start + self.long_seq_microbatch, x.size(0))
                output[start:stop].copy_(self._unpadded_forward(x[start:stop]))
            return output

    def _long_reference_fallback(self, x, mask):
        # Correct (but slower) handling of padding or gradient-enabled calls.
        output = torch.empty_like(x)
        for sample in range(x.size(0)):
            piece = x[sample:sample + 1]
            piece_mask = None if mask is None else mask[sample:sample + 1]
            for layer in self.layers:
                piece = piece + tiled_reference_attention(
                    layer.attention, layer.norm1(piece), piece_mask, self.config.causal, self.query_block
                )
                piece = piece + layer.ffn_out(F.gelu(layer.ffn_in(layer.norm2(piece)), approximate="none"))
                if piece_mask is not None:
                    piece = piece.masked_fill(~piece_mask[..., None], 0)
            piece = self.final_norm(piece)
            if piece_mask is not None:
                piece = piece.masked_fill(~piece_mask[..., None], 0)
            output[sample:sample + 1].copy_(piece)
        return output

    def forward(self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        long_sequence = x.size(1) >= self.long_seq_threshold
        if torch.is_grad_enabled() or not self._all_tokens_valid(valid_token_mask):
            if long_sequence:
                return self._long_reference_fallback(x, valid_token_mask)
            return super().forward(x, valid_token_mask)
        self._ensure_packed_qkv()
        return self._long_forward(x) if long_sequence else self._fast_forward(x)


def copy_model_weights(
    baseline: nn.Module, optimized: nn.Module, strict: bool = True
) -> None:
    """Copy identical weights into both implementations for a fair comparison."""
    state_dict = copy.deepcopy(baseline.state_dict())
    incompatible = optimized.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[warning] missing optimized keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warning] unexpected optimized keys: {incompatible.unexpected_keys}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )
    positions = torch.arange(config.seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask


@dataclass
class AccuracyResult:
    passed: bool
    total_elements: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    mean_abs_error: float
    failed_feature_dims: List[int]
    worst_index: Tuple[int, ...]
    reference_at_worst: float
    optimized_at_worst: float


def compare_outputs(
    reference: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float,
    atol: float,
) -> AccuracyResult:
    if reference.shape != optimized.shape:
        raise AssertionError(
            f"shape mismatch: baseline={tuple(reference.shape)}, "
            f"optimized={tuple(optimized.shape)}"
        )
    if reference.dtype != optimized.dtype:
        print(
            f"[warning] dtype mismatch: baseline={reference.dtype}, "
            f"optimized={optimized.dtype}"
        )

    ref = reference.detach().float()
    opt = optimized.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()

    # Exact interpretation of the requested OR condition. torch.isclose uses
    # atol + rtol * abs(ref), which is slightly more permissive and is not used.
    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed_mask = ~passed_mask
    failed_elements = int(failed_mask.sum().item())
    total_elements = reference.numel()

    flat_worst = int(abs_error.reshape(-1).argmax().item())
    worst_index_list = []
    remaining = flat_worst
    for size in reversed(reference.shape):
        worst_index_list.append(remaining % size)
        remaining //= size
    worst_index = tuple(reversed(worst_index_list))

    denominator = ref.abs().clamp_min(1e-12)
    relative_error = abs_error / denominator

    # Summarize failures by the last/output-feature dimension.
    if reference.ndim == 0:
        failed_feature_dims = [0] if failed_elements else []
    elif reference.ndim == 1:
        failed_feature_dims = torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
    else:
        reduce_dims = tuple(range(reference.ndim - 1))
        failed_by_feature = failed_mask.any(dim=reduce_dims)
        failed_feature_dims = (
            torch.nonzero(failed_by_feature, as_tuple=False).flatten().tolist()
        )

    return AccuracyResult(
        passed=failed_elements == 0,
        total_elements=total_elements,
        failed_elements=failed_elements,
        max_abs_error=float(abs_error.max().item()),
        max_relative_error=float(relative_error.max().item()),
        mean_abs_error=float(abs_error.mean().item()),
        failed_feature_dims=failed_feature_dims,
        worst_index=worst_index,
        reference_at_worst=float(ref[worst_index].item()),
        optimized_at_worst=float(opt[worst_index].item()),
    )


def run_accuracy_tests(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
) -> bool:
    print("\n=== Accuracy check ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")

    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=padding_ratio,
                input_scale=input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)

            all_passed &= result.passed
            global_max_abs = max(global_max_abs, result.max_abs_error)
            global_max_rel = max(global_max_rel, result.max_relative_error)
            total_failed += result.failed_elements
            total_elements += result.total_elements

            status = "PASS" if result.passed else "FAIL"
            print(
                f"trial {trial + 1:02d}/{trials}: {status} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"max_rel={result.max_relative_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

            if not result.passed:
                preview = result.failed_feature_dims[:16]
                suffix = "..." if len(result.failed_feature_dims) > len(preview) else ""
                print(
                    f"  worst_index={result.worst_index}, "
                    f"baseline={result.reference_at_worst:.8g}, "
                    f"optimized={result.optimized_at_worst:.8g}"
                )
                print(f"  failed output feature dims={preview}{suffix}")

    print(
        f"summary: {'PASS' if all_passed else 'FAIL'} | "
        f"max_abs={global_max_abs:.6g} | max_rel={global_max_rel:.6g} | "
        f"failed={total_failed}/{total_elements}"
    )
    return all_passed


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class TimingResult:
    samples_ms: List[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> List[float]:
    samples_ms: List[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

            torch.cuda.synchronize(device)
            for index in range(iterations):
                starts[index].record()
                model(x, valid_mask)
                ends[index].record()
            torch.cuda.synchronize(device)

            samples_ms.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
        else:
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, valid_mask)
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

    return samples_ms


def benchmark_models(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
    reference_kind: str = "original",
) -> dict:
    print("\n=== Performance benchmark ===")
    print("timing excludes random-data generation and uses a fixed input")
    if device.type == "cuda":
        print("CUDA latency is measured with torch.cuda.Event on the current stream")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    print(f"reference_kind={reference_kind}; warmup={warmup}, repeats={repeats}, rounds={rounds}", flush=True)
    # Warm up both models before collecting any timing data.
    print("Warming reference...", flush=True)
    warmup_model(baseline, x, valid_mask, warmup, device)
    print("Warming optimized...", flush=True)
    warmup_model(optimized, x, valid_mask, warmup, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []
    peaks = {"reference": 0, "optimized": 0}

    def collect(model, label):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        print(f"Timing {label}: {repeats} full forwards...", flush=True)
        samples = benchmark_once(model, x, valid_mask, repeats, device)
        if device.type == "cuda":
            peaks[label] = max(peaks[label], torch.cuda.max_memory_allocated(device))
        print(f"  {label} round median_ms={statistics.median(samples):.4f}", flush=True)
        return samples

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                collect(baseline, "reference")
            )
            optimized_samples.extend(
                collect(optimized, "optimized")
            )
        else:
            optimized_samples.extend(
                collect(optimized, "optimized")
            )
            baseline_samples.extend(
                collect(baseline, "reference")
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens_per_call = config.batch_size * config.seq_len
    baseline_tokens_per_second = tokens_per_call * 1000.0 / baseline_result.median_ms
    optimized_tokens_per_second = tokens_per_call * 1000.0 / optimized_result.median_ms

    reference_label = "baseline" if reference_kind == "original" else "tiled_reference"
    print(
        f"{reference_label} : median={baseline_result.median_ms:.4f} ms | "
        f"mean={baseline_result.mean_ms:.4f} ms | "
        f"p90={baseline_result.p90_ms:.4f} ms | "
        f"min={baseline_result.min_ms:.4f} ms | "
        f"throughput={baseline_tokens_per_second:.2f} token/s"
    )
    print(
        f"optimized: median={optimized_result.median_ms:.4f} ms | "
        f"mean={optimized_result.mean_ms:.4f} ms | "
        f"p90={optimized_result.p90_ms:.4f} ms | "
        f"min={optimized_result.min_ms:.4f} ms | "
        f"throughput={optimized_tokens_per_second:.2f} token/s"
    )
    if reference_kind == "original":
        print(f"speedup  : {speedup:.3f}x versus supplied eager baseline")
    else:
        print("original_baseline_speedup: N/A (20.48 TB explicit score tensor)")
        print(f"tiled_reference_speedup: {speedup:.3f}x (experimental, NOT official-baseline speedup)")
    print(f"optimized_phase_peak_allocated_gib: {peaks['optimized'] / 1024**3:.3f}")
    print("Peak allocation includes input, output, both resident models, and active model caches.")
    return dict(
        reference_kind=reference_kind,
        baseline_speedup=speedup if reference_kind == "original" else None,
        tiled_reference_speedup=speedup if reference_kind == "tiled" else None,
        reference_median_ms=baseline_result.median_ms,
        optimized_median_ms=optimized_result.median_ms,
        optimized_mean_ms=optimized_result.mean_ms,
        optimized_p90_ms=optimized_result.p90_ms,
        optimized_tokens_per_second=optimized_tokens_per_second,
        reference_samples_ms=baseline_samples,
        optimized_samples_ms=optimized_samples,
        reference_phase_peak_allocated_gib=peaks["reference"] / 1024**3,
        optimized_phase_peak_allocated_gib=peaks["optimized"] / 1024**3,
    )


def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and optimized PyTorch Transformer"
    )
    parser.add_argument("--all", action="store_true", help="run all 14 shapes in isolated subprocesses")
    parser.add_argument("--shape", type=int, choices=range(1, 15))
    parser.add_argument("--self-test", action="store_true", help="small reference/cache/accuracy regression tests")
    parser.add_argument("--output-dir", default="results/final-run")
    parser.add_argument("--result-json", help=argparse.SUPPRESS)
    parser.add_argument("--modes", choices=("both", "compiled", "eager"), default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument(
        "--user-compile-mode",
        choices=("off", "default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--reference-query-block", type=int, default=256)
    parser.add_argument("--shape14-accuracy-trials", type=int, default=1)
    parser.add_argument("--shape14-warmup", type=int, default=1)
    parser.add_argument("--shape14-repeats", type=int, default=3)
    parser.add_argument("--shape14-rounds", type=int, default=1)
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable TF32 on CUDA for both implementations",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if min(args.microbatch, args.reference_query_block, args.shape14_accuracy_trials,
           args.shape14_repeats, args.shape14_rounds) <= 0 or args.shape14_warmup < 0:
        raise ValueError("invalid shape-14 trial/timing/microbatch/query-block setting")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")


# B, S, D, H, FFN, layers. All published shapes use causal attention.
SHAPES = {
    1: (64, 128, 128, 4, 128, 4), 2: (1, 128, 128, 4, 128, 4),
    3: (4, 128, 128, 4, 128, 4), 4: (16, 128, 128, 4, 128, 4),
    5: (128, 128, 128, 4, 128, 4), 6: (10000, 128, 128, 4, 128, 4),
    7: (64, 128, 32, 4, 32, 4), 8: (64, 128, 1024, 4, 1024, 4),
    9: (64, 128, 128, 1, 128, 4), 10: (64, 128, 128, 2, 128, 4),
    11: (64, 128, 128, 16, 128, 4), 12: (64, 32, 128, 4, 128, 4),
    13: (64, 1024, 128, 4, 128, 4), 14: (32, 100000, 1024, 16, 1024, 2),
}


def published_config(shape):
    return TransformerConfig(*SHAPES[shape], causal=True)


def compare_streamed_outputs(reference, candidate, rtol, atol, tokens_per_block=4096):
    """Apply the UNCHANGED supplied checker to every element, in memory-safe slices."""
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise AssertionError("expected identical [batch, sequence, feature] output shapes")
    total, failed, max_abs, max_rel = 0, 0, 0.0, 0.0
    for batch in range(reference.size(0)):
        for start in range(0, reference.size(1), tokens_per_block):
            stop = min(start + tokens_per_block, reference.size(1))
            result = compare_outputs(reference[batch:batch + 1, start:stop],
                                     candidate[batch:batch + 1, start:stop], rtol=rtol, atol=atol)
            total += result.total_elements
            failed += result.failed_elements
            # NaN must never accidentally produce a finite max-error report.
            max_abs = max(max_abs, result.max_abs_error) if math.isfinite(result.max_abs_error) else float("inf")
            max_rel = max(max_rel, result.max_relative_error) if math.isfinite(result.max_relative_error) else float("inf")
        print(f"  checked output sample {batch + 1}/{reference.size(0)}; cumulative failed={failed}", flush=True)
    assert total == reference.numel(), "checker did not visit every element"
    print(f"full_shape_accuracy: {'PASS' if failed == 0 else 'FAIL'} | "
          f"failed={failed}/{total} | max_abs={max_abs:.6g} | max_rel={max_rel:.6g}", flush=True)
    return dict(passed=failed == 0, failed_elements=failed, total_elements=total,
                max_abs_error=max_abs, max_relative_error=max_rel)


def require_close(reference, candidate, label, rtol=0.02, atol=0.002):
    result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)
    if not result.passed:
        raise AssertionError(f"{label}: failed={result.failed_elements}, max_abs={result.max_abs_error}")
    print(f"self-test PASS: {label}; max_abs={result.max_abs_error:.6g}")


def run_self_tests(device):
    """Small CPU/CUDA regressions; not a substitute for full-shape GPU tests."""
    torch.manual_seed(9876)
    for causal in (False, True):
        cfg = TransformerConfig(3, 17, 32, 4, 48, 2, causal)
        baseline = BaselineTransformer(cfg).to(device).eval()
        tiled = TiledReferenceTransformer(cfg, query_block=5).to(device).eval()
        optimized = UserOptimizedTransformer(cfg, compile_mode="off", query_block=5).to(device).eval()
        new_baseline = BaselineTransformer(cfg).to(device).eval()
        tiled.load_state_dict(baseline.state_dict(), strict=True)
        optimized.load_state_dict(baseline.state_dict(), strict=True)
        x = torch.randn(3, 17, 32, device=device)
        mask = torch.ones(3, 17, dtype=torch.bool, device=device)
        with torch.inference_mode():
            require_close(baseline(x, mask), tiled(x, mask), f"tiled reference causal={causal}")
            require_close(baseline(x, mask), optimized(x, mask), f"packed SDPA causal={causal}")
            # Cache invalidation on in-place mask mutation, then padded-key support.
            mask[:, 5] = False
            mask[:, -2:] = False
            require_close(baseline(x, mask), optimized(x, mask), f"mutated mask causal={causal}")
            require_close(baseline(x, mask), tiled(x, mask), f"padded tiled reference causal={causal}")
            optimized.long_seq_threshold = 1
            require_close(baseline(x, mask), optimized(x, mask), f"long padded fallback causal={causal}")
            mask.fill_(True)
            optimized.long_seq_microbatch = 2  # exercises a non-divisible B=3 tail
            require_close(baseline(x, mask), optimized(x, mask), f"long microbatch tail causal={causal}")
            # Test inference tensors which do not expose a version counter.
            inference_mask = torch.ones_like(mask)
            optimized(x, inference_mask)
            inference_mask[:, -1] = False
            require_close(baseline(x, inference_mask), optimized(x, inference_mask), "inference-mask mutation")
            optimized.long_seq_threshold = 32768
            baseline.layers[0].attention.q_proj.weight.add_(0.01)
            optimized.layers[0].attention.q_proj.weight.add_(0.01)
            require_close(baseline(x, mask), optimized(x, mask), "weight mutation repacks QKV")
            optimized.load_state_dict(new_baseline.state_dict(), strict=True)
            require_close(new_baseline(x, mask), optimized(x, mask), "weight reload after first forward")
            assert set(optimized.state_dict()) == set(new_baseline.state_dict())
        # Gradient-enabled calls must not silently use detached packed weights.
        reference_grad_input = x.detach().clone().requires_grad_(True)
        optimized_grad_input = x.detach().clone().requires_grad_(True)
        new_baseline(reference_grad_input, mask).square().sum().backward()
        optimized(optimized_grad_input, mask).square().sum().backward()
        require_close(reference_grad_input.grad, optimized_grad_input.grad, "gradient-enabled fallback")
    # This pair passes atol+rtol*|ref|, but FAILS the organizer's OR rule.
    strict_ref = torch.tensor([[[0.1]]], device=device)
    strict_candidate = torch.tensor([[[0.103]]], device=device)
    assert not compare_outputs(strict_ref, strict_candidate, rtol=0.02, atol=0.002).passed
    assert not compare_outputs(strict_ref, torch.full_like(strict_ref, float("nan")), rtol=.02, atol=.002).passed
    assert not compare_outputs(torch.full_like(strict_ref, float("inf")),
                               torch.zeros_like(strict_ref), rtol=.02, atol=.002).passed
    print("SELF_TESTS: PASS (including strict OR and nonfinite rejection)", flush=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_full_accuracy(reference, optimized, config, device, dtype, args):
    print("=== Full shape-14 correctness versus independent tiled reference ===", flush=True)
    print("Every batch sample and every token are checked. This is NOT proxy validation.", flush=True)
    results = []
    with torch.inference_mode():
        for trial in range(args.shape14_accuracy_trials):
            x, mask = generate_random_case(config, device, dtype, args.seed + trial,
                                           args.padding_ratio, args.input_scale)
            print(f"Full trial {trial + 1}: executing optimized B={config.batch_size}, S={config.seq_len}...", flush=True)
            candidate = optimized(x, mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print("Optimized output ready; computing independent reference...", flush=True)
            reference.progress = True
            try:
                expected = reference(x, mask)
            finally:
                reference.progress = False
            result = compare_streamed_outputs(expected, candidate, args.rtol, args.atol)
            results.append(result)
            del x, mask, expected, candidate
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return all(r["passed"] for r in results), results


def configure_runtime(args, device):
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32


def run_case(args):
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    validate_args(args, device, dtype)
    configure_runtime(args, device)
    cfg = published_config(args.shape) if args.shape else TransformerConfig(
        args.batch_size, args.seq_len, args.d_model, args.heads, args.ffn_dim, args.layers, args.causal
    )
    cfg.validate()
    is14 = cfg == published_config(14)
    score_bytes = cfg.batch_size * cfg.num_heads * cfg.seq_len**2 * torch.empty((), dtype=dtype).element_size()
    if not is14 and score_bytes > 256 * 1024**3:
        raise ValueError("Custom shape's explicit reference exceeds 256 GiB; use --shape 14 for the published long test")
    if is14:
        if device.type != "cuda" or dtype != torch.float32:
            raise ValueError("Full shape 14 runner requires CUDA and float32")
        if args.padding_ratio != 0:
            raise ValueError("Published shape-14 performance uses all-valid inputs; padding is covered by self-tests")
        # A small sanity check validates the independent tiling implementation
        # before the MUCH larger full input is allocated.
        run_self_tests(device)
        configure_runtime(args, device)
    mode = "long-fused-eager" if is14 else args.user_compile_mode
    print("=== Configuration ===", flush=True)
    print(cfg)
    print(f"torch={torch.__version__}, CUDA build={torch.version.cuda}, dtype={dtype}, mode={mode}")
    print(f"matmul_precision={args.matmul_precision}, allow_tf32={args.allow_tf32}, "
          f"rtol={args.rtol}, atol={args.atol}, padding_ratio={args.padding_ratio}")
    print(f"microbatch={args.microbatch}, reference_query_block={args.reference_query_block}")
    print(f"source_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}; memory_gib="
              f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.3f}")
    if is14:
        print("Original baseline not attempted: 20.48 TB score tensor; official-baseline speedup=N/A")
        print("Tiled-reference ratio is experimental, not an organizer-approved score.")
    reference = TiledReferenceTransformer(cfg, args.reference_query_block) if is14 else BaselineTransformer(cfg)
    optimized = UserOptimizedTransformer(cfg, compile_mode="off" if is14 else args.user_compile_mode,
                                         fullgraph=args.compile_fullgraph,
                                         long_seq_microbatch=args.microbatch,
                                         query_block=args.reference_query_block)
    copy_model_weights(reference, optimized, strict=True)
    reference = reference.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()
    outcome = dict(shape=14 if is14 else args.shape, config=asdict(cfg), mode=mode, status="FAIL",
                   torch_version=torch.__version__, cuda_build=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   settings=vars(args).copy(), reference_kind="tiled" if is14 else "original")
    if is14:
        passed, trials = run_full_accuracy(reference, optimized, cfg, device, dtype, args)
        outcome["full_accuracy_trials"] = trials
        outcome["accuracy_scope"] = "every output element of the full published shape versus independent tiled reference"
    else:
        passed = run_accuracy_tests(reference, optimized, cfg, device, dtype, args.accuracy_trials,
                                    args.seed, args.padding_ratio, args.input_scale, args.rtol, args.atol)
        outcome["accuracy_scope"] = "full shape versus original supplied baseline"
    outcome["accuracy_passed"] = passed
    if not passed:
        print("Accuracy FAILED. Performance and speedup are NOT reported.", flush=True)
        return outcome
    outcome.update(benchmark_models(reference, optimized, cfg, device, dtype, args.seed,
                                    args.padding_ratio, args.input_scale,
                                    args.shape14_warmup if is14 else args.warmup,
                                    args.shape14_repeats if is14 else args.repeats,
                                    args.shape14_rounds if is14 else args.benchmark_rounds,
                                    reference_kind="tiled" if is14 else "original"))
    outcome["status"] = "PASS"
    print(f"FINAL_STATUS: PASS shape={14 if is14 else args.shape} mode={mode}", flush=True)
    return outcome


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_summary(root, records):
    fields = ["shape", "mode", "status", "accuracy_passed", "reference_kind",
              "reference_median_ms", "optimized_median_ms", "baseline_speedup",
              "tiled_reference_speedup", "optimized_tokens_per_second", "optimized_phase_peak_allocated_gib"]
    with (root / "summary.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({k: "N/A" if row.get(k) is None else row[k] for k in fields})
    aggregates = {}
    for mode in ("off", "reduce-overhead"):
        rows = [r for r in records if r.get("mode") == mode and r.get("shape") in range(1, 14)]
        complete = len(rows) == 13 and all(r.get("status") == "PASS" and r.get("baseline_speedup", 0) > 0 for r in rows)
        aggregates[mode] = dict(complete_13_shapes=complete,
            geometric_mean_speedup=math.exp(statistics.fmean(math.log(r["baseline_speedup"]) for r in rows)) if complete else None)
    write_json(root / "summary.json", dict(results=records, aggregates_1_to_13=aggregates,
               note="Shape 14 is excluded from original-baseline aggregates; its tiled ratio is experimental."))
    return aggregates


def run_suite(args):
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Reusing a completed directory would otherwise mix different versions.
    manifest = root / "manifest.json"
    if manifest.exists():
        raise ValueError(f"Choose a NEW --output-dir; {manifest} already exists")
    write_json(manifest, dict(arguments=vars(args), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             python=sys.version, torch=torch.__version__, started=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    modes = ["off", "reduce-overhead"] if args.modes == "both" else ["off" if args.modes == "eager" else "reduce-overhead"]
    schedule = [(shape, mode) for shape in range(1, 14) for mode in modes] + [(14, "long-fused-eager")]
    records = [dict(shape=shape, mode=mode, status="PENDING",
                    reference_kind="tiled" if shape == 14 else "original") for shape, mode in schedule]
    write_summary(root, records)
    settings = ["accuracy_trials", "warmup", "repeats", "benchmark_rounds", "seed", "dtype",
                "device", "padding_ratio", "input_scale", "rtol", "atol", "matmul_precision",
                "microbatch", "reference_query_block", "shape14_accuracy_trials", "shape14_warmup",
                "shape14_repeats", "shape14_rounds"]
    common = [item for key in settings for item in ("--" + key.replace("_", "-"), str(getattr(args, key)))]
    common += ["--allow-tf32" if args.allow_tf32 else "--no-allow-tf32"]
    if args.compile_fullgraph:
        common.append("--compile-fullgraph")
    for index, (shape, mode) in enumerate(schedule):
        stem = f"shape{shape:02d}-{mode}"
        result_path = root / f"{stem}.json"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--shape", str(shape),
                   "--user-compile-mode", "off" if shape == 14 else mode,
                   "--result-json", str(result_path)] + common
        print(f"\n{'=' * 70}\nSHAPE {shape} MODE {mode}; log={root / (stem + '.log')}\n{'=' * 70}", flush=True)
        records[index]["status"] = "RUNNING"
        write_summary(root, records)
        with (root / f"{stem}.log").open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + " ".join(command) + "\n")
            log.flush()
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
            except KeyboardInterrupt:
                child.terminate()
                child.wait()
                records[index]["status"] = "INTERRUPTED"
                write_summary(root, records)
                raise
            finally:
                child.stdout.close()
        if result_path.exists():
            records[index] = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            records[index]["status"] = "ERROR"
        if code != 0:
            records[index]["status"] = "ERROR" if records[index].get("status") == "PASS" else records[index]["status"]
        records[index]["exit_code"] = code
        write_summary(root, records)
    aggregates = write_summary(root, records)
    print("\n=== FINAL SUMMARY (shape 14 uses a DIFFERENT reference) ===")
    print((root / "summary.tsv").read_text(encoding="utf-8"))
    print("Shapes 1-13 geometric means:", json.dumps(aggregates))
    print(f"All evidence saved in {root}")
    return 0 if all(row.get("status") == "PASS" for row in records) else 2


def main():
    args = parse_args()
    if args.self_test:
        device = resolve_device(args.device)
        configure_runtime(args, device)
        run_self_tests(device)
        return 0
    if args.all:
        return run_suite(args)
    result_path = Path(args.result_json) if args.result_json else Path(args.output_dir) / (
        f"shape{args.shape or 'custom'}-{args.user_compile_mode}-{time.time_ns()}.json"
    )
    try:
        outcome = run_case(args)
    except Exception as exc:
        traceback.print_exc()
        outcome = dict(shape=args.shape, mode="long-fused-eager" if args.shape == 14 else args.user_compile_mode,
                       status="ERROR", error_type=type(exc).__name__, error=str(exc))
    write_json(result_path, outcome)
    print(f"result_json={result_path}", flush=True)
    return 0 if outcome["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())



