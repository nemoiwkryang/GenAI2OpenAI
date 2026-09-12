"""Routing, prompt injection, extraction and transport for the new upstream
model versions: GLM-5.3-Flash, Qwen-3.8 and DeepSeek-V4.1."""

import json

from genai_proxy.chat.tool_protocol import extract_tool_calls
from genai_proxy.models.deepseek_v41.codec import (
    DEEPSEEK_V4_1_SPEC,
    official_transport_messages,
    serialize_completion,
)
from genai_proxy.models.glm53.codec import GLM_5_3_SPEC
from genai_proxy.models.glm53.codec import official_tool_prompt as glm53_tool_prompt
from genai_proxy.models.qwen38.codec import official_tool_prompt as qwen38_tool_prompt
from genai_proxy.models.registry import (
    DEEPSEEK_V4_1_ADAPTER,
    DEEPSEEK_V4_FLASH_ADAPTER,
    DEEPSEEK_V4_PRO_ADAPTER,
    GLM_5_2_ADAPTER,
    GLM_5_3_ADAPTER,
    QWEN_3_5_ADAPTER,
    QWEN_3_8_ADAPTER,
    select_tool_adapter,
)
from genai_proxy.reasoning import normalize_reasoning_for_adapter

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


def _record(ai_type: str, ai_name: str) -> dict:
    return {
        "aiType": ai_type,
        "aiName": ai_name,
        "simpleName": ai_name.lower(),
        "rootAiType": "xinference",
        "rootModelName": "Xinference",
    }


def test_new_version_records_route_to_new_adapters():
    assert (
        select_tool_adapter("chatglm", _record("chatglm", "GLM-5.3-Flash"))
        == GLM_5_3_ADAPTER
    )
    assert (
        select_tool_adapter("qwen-instruct", _record("qwen-instruct", "Qwen-3.8"))
        == QWEN_3_8_ADAPTER
    )
    assert (
        select_tool_adapter("deepseek-pro", _record("deepseek-pro", "DeepSeek-V4.1"))
        == DEEPSEEK_V4_1_ADAPTER
    )
    assert (
        select_tool_adapter(
            "deepseek-chat", _record("deepseek-chat", "DeepSeek-V4-Flash")
        )
        == DEEPSEEK_V4_FLASH_ADAPTER
    )


def test_bare_model_keys_keep_legacy_routing():
    assert select_tool_adapter("chatglm", None) == GLM_5_2_ADAPTER
    assert select_tool_adapter("qwen-instruct", None) == QWEN_3_5_ADAPTER
    assert select_tool_adapter("deepseek-pro", None) == DEEPSEEK_V4_PRO_ADAPTER
    assert select_tool_adapter("deepseek-chat", None) == DEEPSEEK_V4_FLASH_ADAPTER


def test_v41_flash_record_routes_to_v41():
    assert (
        select_tool_adapter(
            "deepseek-chat", _record("deepseek-chat", "DeepSeek-V4.1-Flash")
        )
        == DEEPSEEK_V4_1_ADAPTER
    )


def test_glm53_official_tool_prompt_extracts_tool_section():
    prompt = glm53_tool_prompt(GLM_5_3_SPEC, [WEATHER_TOOL])
    assert "# Tools" in prompt
    assert "<tool_call>{function-name}<arg_key>" in prompt
    assert '"get_weather"' in prompt
    assert "Reasoning Effort" not in prompt
    assert "__GENAI2OPENAI" not in prompt


def test_qwen38_official_tool_prompt_extracts_tool_section_without_effort():
    prompt = qwen38_tool_prompt([WEATHER_TOOL])
    assert prompt.startswith("# Tools")
    assert "<function=" in prompt
    assert "Reasoning effort" not in prompt
    assert "__GENAI2OPENAI" not in prompt


def test_v41_spaced_dsml_tool_calls_are_recovered():
    content = (
        "Let me check the weather.\n\n"
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="get_weather">\n'
        '<｜DSML｜ parameter name="location" string="true">Shanghai</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls>"
    )
    calls, remaining = extract_tool_calls(
        content, adapter=DEEPSEEK_V4_1_ADAPTER, tools=[WEATHER_TOOL]
    )
    assert calls is not None
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"location": "Shanghai"}
    assert remaining == "Let me check the weather."


def test_v41_serialize_completion_round_trips_through_extractor():
    message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "thinking",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Shanghai"}',
                },
            }
        ],
    }
    serialized = serialize_completion(message, thinking=True)
    assert "<｜DSML｜ calls>" in serialized
    assert '<｜DSML｜ invoke name="get_weather">' in serialized
    calls, remaining = extract_tool_calls(
        serialized, adapter=DEEPSEEK_V4_1_ADAPTER, tools=[WEATHER_TOOL]
    )
    assert calls is not None
    assert calls[0]["function"]["name"] == "get_weather"
    assert remaining is not None


def test_v41_transport_returns_system_user_pair():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather in Shanghai?"},
    ]
    transported = official_transport_messages(
        DEEPSEEK_V4_1_SPEC,
        messages,
        [WEATHER_TOOL],
        reasoning_config={"effort": "high"},
    )
    assert [m["role"] for m in transported] == ["system", "user"]
    assert "## Tools" in transported[0]["content"]
    assert transported[1]["content"] == "Weather in Shanghai?"


def test_v41_transport_verification_accepts_every_effort():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather in Shanghai?"},
    ]
    for effort in ("low", "high", "max", None):
        transported = official_transport_messages(
            DEEPSEEK_V4_1_SPEC,
            messages,
            [WEATHER_TOOL],
            reasoning_config={"effort": effort} if effort else None,
        )
        assert len(transported) == 2


def test_v41_reasoning_prompt_injects_numeric_budget():
    from genai_proxy.models.deepseek_v4 import inject_deepseek_reasoning_prompt

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    out = inject_deepseek_reasoning_prompt(
        messages,
        {"effort": "max"},
        adapter=DEEPSEEK_V4_1_ADAPTER,
    )
    assert out[0]["content"].startswith(
        "Reasoning Effort: 100 (range 1-100, the higher the value, the more thorough the reasoning)"
    )
    out = inject_deepseek_reasoning_prompt(
        messages,
        {"effort": "low"},
        adapter=DEEPSEEK_V4_1_ADAPTER,
    )
    assert out[0]["content"].startswith("Reasoning Effort: 50")


def test_reasoning_normalization_for_new_adapters():
    assert normalize_reasoning_for_adapter(GLM_5_3_ADAPTER, {"effort": "high"}) == {
        "effort": "max"
    }
    assert normalize_reasoning_for_adapter(QWEN_3_8_ADAPTER, {"effort": "max"}) == {
        "effort": "xhigh"
    }
    assert normalize_reasoning_for_adapter(QWEN_3_8_ADAPTER, {"effort": "low"}) == {
        "effort": "low"
    }
    assert normalize_reasoning_for_adapter(
        DEEPSEEK_V4_1_ADAPTER, {"effort": "minimal"}
    ) == {"effort": "low"}
    assert normalize_reasoning_for_adapter(
        DEEPSEEK_V4_1_ADAPTER, {"effort": "low"}
    ) == {"effort": "low"}
    assert normalize_reasoning_for_adapter(
        DEEPSEEK_V4_1_ADAPTER, {"effort": "medium"}
    ) == {"effort": "high"}
    assert normalize_reasoning_for_adapter(
        DEEPSEEK_V4_1_ADAPTER, {"effort": "max"}
    ) == {"effort": "max"}
