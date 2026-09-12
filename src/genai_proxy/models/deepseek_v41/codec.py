"""Hardcoded official prompt codec for DeepSeek V4.1.

The official prompt format ships with the repository (``official_encoding.py``)
instead of being downloaded. V4.1 changed three things relative to V4:

1. DSML tag names carry a leading space (``<｜DSML｜ calls>``,
   ``<｜DSML｜ invoke>``, ``<｜DSML｜ parameter>``).
2. Reasoning effort is a numeric budget rendered as
   ``Reasoning Effort: {budget} (range 1-100, ...)`` (low=50, high=75, max=100).
3. Mid-conversation system messages use ``<｜System｜>``.

The shared transport in :mod:`genai_proxy.models.deepseek_v4.codec` is reused
with the vendored encoder; the byte re-encode verification is skipped because
the vendored renderer cannot drift and V4.1's default effort prefix would be
rendered twice by a naive re-encode.
"""

import json

from genai_proxy.models.deepseek_v4.codec import (
    DEEPSEEK_V4_PRO_SPEC,
    official_transport_messages as _v4_transport_messages,
)
from genai_proxy.models.deepseek_v41 import official_encoding
from genai_proxy.models.hf_assets import TokenizerSpec

DEEPSEEK_V4_1_SPEC = TokenizerSpec(
    family="deepseek_v4_1",
    repository="deepseek-ai/DeepSeek-V4.1-Flash",
    revision="dba1be0a40aa",
    # The V4.1 tokenizer differs from V4 only in a handful of added tokens;
    # counting reuses the cached V4 artifact (see docs/models/deepseek-v4.md).
    tokenizer=DEEPSEEK_V4_PRO_SPEC.tokenizer,
)


def encoder_namespace() -> dict:
    """Expose the vendored encoder under the same names as the HF loader."""
    return {
        "bos_token": official_encoding.bos_token,
        "eos_token": official_encoding.eos_token,
        "USER_SP_TOKEN": official_encoding.USER_SP_TOKEN,
        "ASSISTANT_SP_TOKEN": official_encoding.ASSISTANT_SP_TOKEN,
        "SYSTEM_SP_TOKEN": official_encoding.SYSTEM_SP_TOKEN,
        "thinking_start_token": official_encoding.thinking_start_token,
        "thinking_end_token": official_encoding.thinking_end_token,
        "encode_messages": official_encoding.encode_messages,
        "render_message": official_encoding.render_message,
        "merge_tool_messages": official_encoding.merge_tool_messages,
        "sort_tool_results_by_call_order": (
            official_encoding.sort_tool_results_by_call_order
        ),
        "find_last_user_index": official_encoding.find_last_user_index,
        "render_tools": official_encoding.render_tools,
        "tools_from_openai_format": official_encoding.tools_from_openai_format,
    }


def official_tool_prompt(spec: TokenizerSpec, function_tools: list[dict]) -> str:
    return official_encoding.render_tools(
        official_encoding.tools_from_openai_format(function_tools)
    )


def official_reasoning_prefix(spec: TokenizerSpec, effort: str | None) -> str:
    if effort in (None, "none"):
        return ""
    return official_encoding.render_reasoning_effort(0, "thinking", effort)


def encode_prompt(
    messages,
    *,
    thinking: bool | None,
    reasoning_config: dict | None = None,
) -> str:
    """Render the full official prompt for token accounting."""
    effort = (reasoning_config or {}).get("effort")
    if thinking is False or effort == "none":
        effort = None
    return official_encoding.encode_messages(
        messages,
        thinking_mode="chat" if thinking is False else "thinking",
        drop_thinking=True,
        add_default_bos_token=True,
        reasoning_effort=effort,
    )


def official_transport_messages(
    spec: TokenizerSpec,
    messages,
    tools,
    *,
    reasoning_config: dict | None = None,
    tool_choice_suffix: str = "",
) -> list[dict]:
    return _v4_transport_messages(
        spec,
        messages,
        tools,
        reasoning_config=reasoning_config,
        tool_choice_suffix=tool_choice_suffix,
        encoder=encoder_namespace(),
        user_prefix_tokens=(
            official_encoding.USER_SP_TOKEN,
            official_encoding.SYSTEM_SP_TOKEN,
        ),
        verify=False,
        model_label="DeepSeek V4.1",
    )


def serialize_completion(
    message: dict,
    *,
    finish_reason: str = "stop",
    thinking: bool | None = None,
) -> str:
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    parts = (
        [str(content)]
        if thinking is False
        else [str(reasoning), "</think>", str(content)]
    )
    if tool_calls:
        parts.append(_tool_calls(tool_calls))
    if finish_reason != "length":
        parts.append(official_encoding.eos_token)
    return "".join(parts)


def _tool_calls(tool_calls) -> str:
    rendered = []
    for call in tool_calls:
        function = call.get("function") or {}
        rendered.append(
            official_encoding.tool_call_template.format(
                dsml_token=official_encoding.dsml_token,
                tool_call_tag_name=official_encoding.tool_call_tag_name,
                name=function.get("name", ""),
                arguments=official_encoding.encode_arguments_to_dsml(
                    {
                        "name": function.get("name", ""),
                        "arguments": _json_arguments(function.get("arguments")),
                    }
                ),
            )
        )
    return "\n\n" + official_encoding.tool_calls_template.format(
        dsml_token=official_encoding.dsml_token,
        tool_calls="\n".join(rendered),
        tc_block_name=official_encoding.tool_calls_block_name,
    )


def _json_arguments(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"arguments": str(value or "")}
    return parsed if isinstance(parsed, dict) else {"arguments": parsed}


__all__ = [
    "DEEPSEEK_V4_1_SPEC",
    "encode_prompt",
    "encoder_namespace",
    "official_reasoning_prefix",
    "official_tool_prompt",
    "official_transport_messages",
    "serialize_completion",
]
