# SPDX-License-Identifier: Apache-2.0
"""trellis-dense-checkpoint/1 accepts mcg K3..K6 and keeps lut_e4m3 K2 unchanged."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from b12x.moe.checkpoints import independent
from b12x.moe.checkpoints.independent import read_independent_layer

HIDDEN, INTER, EXPERTS, LAYER = 128, 256, 2, 3
MCG_SEED = 0xCBAC1FED


def _h128() -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / 128**0.5).float()


def _write(root, *, codebook="mcg", bits=3, seed=MCG_SEED, words=None, with_lut=None):
    """A tiny checkpoint: per-matrix tile words are arange-coded so slicing is checkable."""
    words = 16 * bits if words is None else words
    tables = {"hadamard_128": _h128()}
    if with_lut if with_lut is not None else codebook == "lut_e4m3":
        from b12x._lib.quant.lut_e4m3 import lut_e4m3_direct_table_cpu

        tables["lut_e4m3_k2"] = lut_e4m3_direct_table_cpu()[:65536].cpu()
    save_file(tables, str(root / "tables.safetensors"))
    tensors, quantized = {}, []
    for e in range(EXPERTS):
        for p in ("w1", "w3", "w2"):
            out_f, in_f = (INTER, HIDDEN) if p != "w2" else (HIDDEN, INTER)
            stem = f"layers.{LAYER}.ffn.experts.{e}.{p}"
            n = (in_f // 16) * (out_f // 16) * words
            base = (e * 3 + ("w1", "w3", "w2").index(p)) * 1000
            tensors[stem + ".trellis"] = (
                (torch.arange(n, dtype=torch.int32) + base) % 32000
            ).to(torch.int16).reshape(in_f // 16, out_f // 16, words)
            tensors[stem + ".suh"] = torch.linspace(0.5, 1.5, in_f).half()
            tensors[stem + ".svh"] = torch.linspace(1.5, 0.5, out_f).half()
            quantized.append(
                {"name": stem + ".weight", "logical_shape": [out_f, in_f],
                 "packed_file": "experts-L03.safetensors"}
            )
    save_file(tensors, str(root / "experts-L03.safetensors"))
    manifest = {
        "schema": "trellis-dense-checkpoint/1",
        "codebook": codebook,
        "bits_per_quantized_coefficient": bits,
        "state_bits": 16,
        "tailbite_context": 128,
        "tile_shape": [16, 16],
        "hadamard_block": 128,
        "scale_dtype": "float16",
        "decoder_tables": "tables.safetensors",
        "decoder_tables_sha256": hashlib.sha256(
            (root / "tables.safetensors").read_bytes()
        ).hexdigest(),
        "quantized": quantized,
    }
    if seed is not None:
        manifest["codebook_seed"] = seed
    (root / "trellis-manifest.json").write_text(json.dumps(manifest))
    independent._manifest.cache_clear()
    return tensors


def _read(root, tp_rank=0, tp_size=1):
    return read_independent_layer(
        root, LAYER, num_experts=EXPERTS, hidden_size=HIDDEN,
        intermediate_size=INTER, tp_rank=tp_rank, tp_size=tp_size,
    )


@pytest.mark.parametrize("bits", [3, 4, 5, 6])
def test_mcg_uniform_k_reads(tmp_path, bits):
    _write(tmp_path, bits=bits)
    source, weights = _read(tmp_path)
    assert source.config.codebook.value == "mcg"
    assert source.uniform_bits == bits
    assert int(weights.rate) == 17 * bits
    slots, tiles = INTER // 32, HIDDEN // 16
    assert weights.codes.shape == (slots, EXPERTS * 3 * 2 * tiles * 16 * bits * 2)


def test_mcg_tp2_slices_fc1_columns_and_fc2_rows(tmp_path):
    tensors = _write(tmp_path, bits=3)
    _, weights = _read(tmp_path, tp_rank=1, tp_size=2)
    local, tiles, words = INTER // 2, HIDDEN // 16, 48
    codes = weights.codes.view(torch.int16).reshape(local // 32, EXPERTS, 3, 2, tiles, words)
    w1 = tensors[f"layers.{LAYER}.ffn.experts.1.w1.trellis"]
    want_w1 = w1[:, local // 16 : INTER // 16, :].reshape(tiles, local // 32, 2, words).permute(1, 2, 0, 3)
    assert torch.equal(codes[:, 1, 0], want_w1)
    w2 = tensors[f"layers.{LAYER}.ffn.experts.1.w2.trellis"]
    want_w2 = w2[local // 16 : INTER // 16, :, :].reshape(local // 32, 2, tiles, words)
    assert torch.equal(codes[:, 1, 2], want_w2)


def test_lut_e4m3_k2_unchanged(tmp_path):
    _write(tmp_path, codebook="lut_e4m3", bits=2, seed=None)
    source, weights = _read(tmp_path)
    assert source.config.codebook.value == "lut_e4m3"
    assert source.uniform_bits == 2
    assert int(weights.rate) == 0x22


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"bits": 2}, "mcg"),                                   # b12x has no mcg K2
        ({"bits": 7}, "bits_per_quantized_coefficient"),
        ({"seed": None}, "codebook_seed"),
        ({"seed": 12345}, "codebook_seed"),
        ({"codebook": "mul1"}, "codebook"),                     # exllamav3 default, b12x cannot decode
        ({"codebook": "lut_e4m3", "bits": 3, "seed": None}, "bits_per_quantized_coefficient"),
        ({"bits": 3, "words": 32}, "native tile shape"),        # K3 manifest, K2 words
    ],
)
def test_rejects(tmp_path, kwargs, match):
    _write(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        _read(tmp_path)
