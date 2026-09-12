"""DeepSeek V4.1 prompt and tool-call adapter (hardcoded official codec)."""

from genai_proxy.models.deepseek_v41.codec import (
    DEEPSEEK_V4_1_SPEC,
    encode_prompt,
    encoder_namespace,
    official_reasoning_prefix,
    official_tool_prompt,
    official_transport_messages,
    serialize_completion,
)

__all__ = [
    "DEEPSEEK_V4_1_SPEC",
    "encode_prompt",
    "encoder_namespace",
    "official_reasoning_prefix",
    "official_tool_prompt",
    "official_transport_messages",
    "serialize_completion",
]
