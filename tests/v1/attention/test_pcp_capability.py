# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.attention.backend import AttentionBackend, AttentionType


@pytest.mark.parametrize(
    ("supports_pcp", "is_rejected"),
    [
        (True, False),
        (False, True),
    ],
)
def test_attention_backend_pcp_capability_gate(supports_pcp, is_rejected):
    impl_cls = SimpleNamespace(supports_pcp=supports_pcp)
    with patch.object(
        AttentionBackend,
        "get_impl_cls",
        return_value=impl_cls,
    ):
        invalid_reasons = AttentionBackend.validate_configuration(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=SimpleNamespace(),
            attn_type=AttentionType.DECODER,
            use_pcp=True,
        )

    assert ("PCP not supported" in invalid_reasons) is is_rejected
