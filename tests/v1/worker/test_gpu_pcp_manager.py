# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _make_pcp_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=False,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        lora_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )


def test_validate_config_accepts_backend_gated_gqa_pcp():
    PCPManager.validate_config(
        _make_pcp_config(),
        supports_mm_inputs=False,
    )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("pp", "PP"),
        ("encoder_decoder", "encoder-decoder"),
        ("mm", "MM inputs"),
        ("lora", "LoRA"),
        ("spec_decode", "speculative decoding"),
        ("full_graph", "PIECEWISE"),
    ],
)
def test_validate_config_preserves_common_pcp_restrictions(case, match):
    vllm_config = _make_pcp_config()
    supports_mm_inputs = case == "mm"
    if case == "pp":
        vllm_config.parallel_config.pipeline_parallel_size = 2
    elif case == "encoder_decoder":
        vllm_config.model_config.is_encoder_decoder = True
    elif case == "lora":
        vllm_config.lora_config = object()
    elif case == "spec_decode":
        vllm_config.speculative_config = object()
    elif case == "full_graph":
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL

    with pytest.raises(NotImplementedError, match=match):
        PCPManager.validate_config(
            vllm_config,
            supports_mm_inputs=supports_mm_inputs,
        )
