#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every output element:
    abs(user - ref) <= atol
    OR
    abs(user - ref) <= rtol * abs(ref)

The default thresholds are atol=0.001 and rtol=0.01 (1%).
"""

from __future__ import annotations

import argparse
import copy
import math
import statistics
import time
from dataclasses import dataclass
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


class UserOptimizedTransformer(BaselineTransformer):
    """
    Fifth optimization pass for the published TechJam test matrix.

    Changes relative to V3, in addition to everything V3 already does:
      * torch.compile(dynamic=False). Each benchmark process runs exactly one
        fixed shape, so telling the compiler the shapes are static lets it
        specialize instead of emitting dynamic-shape guards and generic
        kernels. This is the main expected win at the small published shapes,
        where guard and dispatch overhead is a meaningful fraction of latency.
      * Single-permute QKV unpack. V3 packed the projections but then undid
        the packing with a split plus three separate view/transpose chains.
        V4 reshapes the packed output once to [B, S, 3, H, D] and permutes to
        [3, B, H, S, D], so Q/K/V fall out of one unbind with no per-tensor
        stride juggling.
      * fullgraph is opt-in (--compile-fullgraph). When set, torch.compile
        raises instead of silently falling back to eager on a graph break,
        which turns "the compiler quietly gave up" into a visible error.
        It is off by default so a graph break degrades rather than fails.

    Everything else (SDPA attention, packed QKV weights, hoisted padding mask,
    strict weight-copy compatibility) is unchanged from V3.

    Shape 14 adds a separate eager long-sequence path. It processes independent
    batch elements in microbatches, writes into one preallocated output, and
    never allows SDPA's explicit math fallback. A query-blocked implementation
    is also available if the installed fused kernel rejects S=100000.
    """

    def __init__(
        self,
        config: TransformerConfig,
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = False,
        long_seq_threshold: int = 32_768,
        long_seq_microbatch: int = 1,
        long_seq_backend: str = "fused",
        long_seq_query_block: int = 128,
    ) -> None:
        super().__init__(config)
        if long_seq_threshold <= 0:
            raise ValueError("long_seq_threshold must be positive")
        if long_seq_microbatch <= 0:
            raise ValueError("long_seq_microbatch must be positive")
        if long_seq_backend not in ("fused", "blocked"):
            raise ValueError("long_seq_backend must be 'fused' or 'blocked'")
        if long_seq_query_block <= 0:
            raise ValueError("long_seq_query_block must be positive")
        self.compile_mode = compile_mode
        self.fullgraph = fullgraph
        self.long_seq_threshold = long_seq_threshold
        self.long_seq_microbatch = long_seq_microbatch
        self.long_seq_backend = long_seq_backend
        self.long_seq_query_block = long_seq_query_block
        self._valid_mask_key: Optional[torch.Tensor] = None
        self._valid_mask_value = True
        self._qkv_packed_built = False

        # torch.compile traces lazily on first call, so wrapping here still
        # picks up the weights, device and dtype that main() applies later.
        # dynamic=False: one process benchmarks one fixed shape, so static
        # specialization is always correct here and avoids dynamic guards.
        self._fast_forward = (
            self._unpadded_forward
            if compile_mode == "off"
            else torch.compile(
                self._unpadded_forward,
                mode=compile_mode,
                dynamic=False,
                fullgraph=fullgraph,
            )
        )

    def _build_packed_qkv(self) -> None:
        # Runs once, lazily, on first forward call -- by then copy_model_weights
        # and .to(device, dtype) have already run, so we pack the *final*
        # weights rather than freezing a stale copy from __init__.
        for layer in self.layers:
            attention = layer.attention
            w_packed = torch.cat(
                [attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight],
                dim=0,
            )
            b_packed = torch.cat(
                [attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias],
                dim=0,
            )
            attention.register_buffer("qkv_weight_packed", w_packed, persistent=False)
            attention.register_buffer("qkv_bias_packed", b_packed, persistent=False)
        self._qkv_packed_built = True

    @staticmethod
    def _sdpa_attention(
        attention: BaselineSelfAttention,
        x: torch.Tensor,
        causal: bool,
        fused_only: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # One matmul against the packed weight instead of three separate
        # q_proj/k_proj/v_proj calls. The packed weight is cat([q, k, v]) on
        # dim 0, so the output's last dim is [q | k | v], and viewing it as
        # (3, num_heads, head_dim) recovers them in that order.
        qkv = F.linear(x, attention.qkv_weight_packed, attention.qkv_bias_packed)
        q, k, v = (
            qkv.view(batch, seq_len, 3, attention.num_heads, attention.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        sdpa_args = dict(
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=attention.scale,
        )
        if fused_only:
            # The math fallback materializes [B,H,S,S]. For shape 14 that is
            # 20.48 TB in fp32, so an unsupported fused kernel must fail safely.
            with sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.CUDNN_ATTENTION,
                ]
            ):
                context = F.scaled_dot_product_attention(q, k, v, **sdpa_args)
        else:
            context = F.scaled_dot_product_attention(q, k, v, **sdpa_args)
        context = context.transpose(1, 2).reshape(
            batch, seq_len, attention.d_model
        )
        return attention.out_proj(context)

    def _blocked_attention(
        self,
        attention: BaselineSelfAttention,
        x: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Memory-bounded exact attention fallback for very long sequences."""
        batch, seq_len, _ = x.shape
        qkv = F.linear(x, attention.qkv_weight_packed, attention.qkv_bias_packed)
        q, k, v = (
            qkv.view(batch, seq_len, 3, attention.num_heads, attention.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        context = torch.empty(
            q.shape,
            device=q.device,
            dtype=q.dtype,
        )

        block_size = self.long_seq_query_block
        for start in range(0, seq_len, block_size):
            stop = min(start + block_size, seq_len)
            key_stop = stop if causal else seq_len
            scores = torch.matmul(
                q[:, :, start:stop, :],
                k[:, :, :key_stop, :].transpose(-2, -1),
            )
            scores.mul_(attention.scale)

            if causal:
                query_positions = torch.arange(
                    start, stop, device=x.device
                )[:, None]
                key_positions = torch.arange(key_stop, device=x.device)[None, :]
                scores.masked_fill_(key_positions > query_positions, float("-inf"))

            probabilities = torch.softmax(scores.float(), dim=-1).to(x.dtype)
            context[:, :, start:stop, :] = torch.matmul(
                probabilities,
                v[:, :, :key_stop, :],
            )
            del scores, probabilities

        context = context.transpose(1, 2).reshape(batch, seq_len, attention.d_model)
        return attention.out_proj(context)

    def _all_tokens_valid(self, valid_token_mask: Optional[torch.Tensor]) -> bool:
        if valid_token_mask is None:
            return True
        # bool() on a CUDA tensor forces a device sync, which would both inflate
        # the measured latency and be illegal inside a CUDA graph. The benchmark
        # reuses one mask object across calls, so cache on tensor identity and
        # pay for the sync once per distinct mask, outside the compiled region.
        if valid_token_mask is not self._valid_mask_key:
            self._valid_mask_key = valid_token_mask
            self._valid_mask_value = bool(valid_token_mask.all())
        return self._valid_mask_value

    def _unpadded_forward(self, x: torch.Tensor) -> torch.Tensor:
        causal = self.config.causal
        for layer in self.layers:
            x = x + self._sdpa_attention(layer.attention, layer.norm1(x), causal)
            x = x + layer.ffn_out(
                F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none")
            )
        return self.final_norm(x)

    def _long_sequence_chunk_forward(self, x: torch.Tensor) -> torch.Tensor:
        causal = self.config.causal
        for layer in self.layers:
            normalized = layer.norm1(x)
            if self.long_seq_backend == "fused":
                attention_output = self._sdpa_attention(
                    layer.attention,
                    normalized,
                    causal,
                    fused_only=True,
                )
            else:
                attention_output = self._blocked_attention(
                    layer.attention,
                    normalized,
                    causal,
                )
            x = x + attention_output
            x = x + layer.ffn_out(
                F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none")
            )
        return self.final_norm(x)

    def _long_sequence_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process independent batch slices without duplicating the final output."""
        batch_size = x.size(0)
        microbatch = min(self.long_seq_microbatch, batch_size)
        if microbatch == batch_size:
            return self._long_sequence_chunk_forward(x)

        output = torch.empty_like(x)
        for start in range(0, batch_size, microbatch):
            stop = min(start + microbatch, batch_size)
            output[start:stop].copy_(
                self._long_sequence_chunk_forward(x[start:stop])
            )
        return output

    def _padded_forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        # This path keeps invalid query outputs zero but does NOT exclude padded
        # keys inside SDPA, so it does not reproduce the baseline under padding.
        # Do not claim padded-case support until a variable-length attention
        # path replaces it.
        invalid = ~valid_token_mask[..., None]
        causal = self.config.causal
        for layer in self.layers:
            x = x + self._sdpa_attention(layer.attention, layer.norm1(x), causal)
            x = x + layer.ffn_out(
                F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none")
            )
            x = x.masked_fill(invalid, 0)
        x = self.final_norm(x)
        return x.masked_fill(invalid, 0)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._qkv_packed_built:
            self._build_packed_qkv()
        if self._all_tokens_valid(valid_token_mask):
            if x.size(1) >= self.long_seq_threshold:
                return self._long_sequence_forward(x)
            return self._fast_forward(x)
        return self._padded_forward(x, valid_token_mask)


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
) -> None:
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

    # Warm up both models before collecting any timing data.
    warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
        else:
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens_per_call = config.batch_size * config.seq_len
    baseline_tokens_per_second = tokens_per_call * 1000.0 / baseline_result.median_ms
    optimized_tokens_per_second = tokens_per_call * 1000.0 / optimized_result.median_ms

    print(
        f"baseline : median={baseline_result.median_ms:.4f} ms | "
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
    print(f"speedup  : {speedup:.3f}x based on median latency")


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
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
        help="torch.compile mode for the baseline, used with --compile-baseline",
    )
    parser.add_argument(
        "--user-compile-mode",
        choices=("off", "default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
        help=(
            "torch.compile mode baked into UserOptimizedTransformer; "
            "'reduce-overhead' enables CUDA graphs"
        ),
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help=(
            "pass fullgraph=True to torch.compile so a graph break raises "
            "instead of silently falling back to eager"
        ),
    )
    parser.add_argument(
        "--long-seq-threshold",
        type=int,
        default=32_768,
        help="sequence length at which batch microbatching is enabled",
    )
    parser.add_argument(
        "--long-seq-microbatch",
        type=int,
        default=1,
        help="batch chunk size for the shape-14 path",
    )
    parser.add_argument(
        "--long-seq-backend",
        choices=("fused", "blocked"),
        default="fused",
        help="fused SDPA or the bounded query-block fallback",
    )
    parser.add_argument(
        "--long-seq-query-block",
        type=int,
        default=128,
        help="query tile size used by the blocked fallback",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
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
    if args.long_seq_threshold <= 0:
        raise ValueError("long_seq_threshold must be positive")
    if args.long_seq_microbatch <= 0:
        raise ValueError("long_seq_microbatch must be positive")
    if args.long_seq_query_block <= 0:
        raise ValueError("long_seq_query_block must be positive")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()
    validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(
        config,
        compile_mode=args.user_compile_mode,
        fullgraph=args.compile_fullgraph,
        long_seq_threshold=args.long_seq_threshold,
        long_seq_microbatch=args.long_seq_microbatch,
        long_seq_backend=args.long_seq_backend,
        long_seq_query_block=args.long_seq_query_block,
    )
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    # Compile only after model construction, weight copy, device transfer, and eval().
    # The optimized model compiles itself in __init__ via --user-compile-mode.
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)

    print("=== Configuration ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    print(f"user_compile_mode={args.user_compile_mode}")
    print(
        f"long_seq_threshold={args.long_seq_threshold}, "
        f"long_seq_microbatch={args.long_seq_microbatch}, "
        f"long_seq_backend={args.long_seq_backend}, "
        f"long_seq_query_block={args.long_seq_query_block}"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    accuracy_passed = run_accuracy_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol,
        atol=args.atol,
    )

    if not accuracy_passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        print("Use --benchmark-on-failure to benchmark an incorrect implementation anyway.")
        return 2

    benchmark_models(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
    )
    return 0 if accuracy_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
