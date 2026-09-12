"""Hardcoded adapters for GLM 5.3 Flash, Qwen 3.8 and DeepSeek V4.1."""

import json

from genai_proxy.models import (
    DEEPSEEK_V4_1_ADAPTER,
    DEEPSEEK_V4_FLASH_ADAPTER,
    DEEPSEEK_V4_PRO_ADAPTER,
    GLM_5_2_ADAPTER,
    GLM_5_3_ADAPTER,
    QWEN_3_5_ADAPTER,
    QWEN_3_8_ADAPTER,
    bridge_tool_prompt,
    extract_bridge_tool_calls,
    extract_deepseek_tool_calls,
    inject_bridge_tool_prompt,
    select_tool_adapter,
    tool_start_tags,
)
from genai_proxy.models.glm53.codec import GLM_5_3_SPEC
from genai_proxy.models.hf_assets import load_template
from genai_proxy.models.qwen38.codec import QWEN_3_8_SPEC
from genai_proxy.token_usage import (
    DEEPSEEK_V4_1_SPEC,
    _serialized_completion,
    official_deepseek_transport_messages,
    official_reasoning_prefix_for_adapter,
    official_tool_prompt_for_adapter,
    render_chat_prompt,
    tokenizer_family_for_model,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def _record(ai_type, ai_name, desc=""):
    return {
        "aiType": ai_type,
        "aiName": ai_name,
        "simpleName": ai_name,
        "descInfo": desc,
        "rootModelName": "Xinference",
    }


def test_new_model_records_route_to_hardcoded_adapters():
    assert (
        select_tool_adapter(
            "chatglm",
            _record("chatglm", "GLM-5.3-Flash", "本地部署的GLM 5.3 Flash 适合长任务执行"),
        )
        == GLM_5_3_ADAPTER
    )
    assert (
        select_tool_adapter(
            "qwen-instruct",
            _record("qwen-instruct", "Qwen-3.8", "千问最新版本模型"),
        )
        == QWEN_3_8_ADAPTER
    )
    assert (
        select_tool_adapter(
            "deepseek-pro",
            _record("deepseek-pro", "DeepSeek-V4.1", "本地部署最新DeepSeeK V4.1 552B参数模型"),
        )
        == DEEPSEEK_V4_1_ADAPTER
    )
    # Previous generations keep their own adapters.
    assert (
        select_tool_adapter("chatglm", _record("chatglm", "GLM 5.2"))
        == GLM_5_2_ADAPTER
    )
    assert (
        select_tool_adapter("qwen-instruct", _record("qwen-instruct", "Qwen-3.5"))
        == QWEN_3_5_ADAPTER
    )
    assert (
        select_tool_adapter(
            "deepseek-pro", _record("deepseek-pro", "DeepSeek-V4-Pro")
        )
        == DEEPSEEK_V4_PRO_ADAPTER
    )
    assert (
        select_tool_adapter(
            "deepseek-chat", _record("deepseek-chat", "DeepSeek-V4-Flash")
        )
        == DEEPSEEK_V4_FLASH_ADAPTER
    )


def test_new_model_families_resolve_for_token_counting():
    assert (
        tokenizer_family_for_model(
            "chatglm", _record("chatglm", "GLM-5.3-Flash"), GLM_5_3_ADAPTER
        )
        == GLM_5_3_SPEC.family
    )
    assert (
        tokenizer_family_for_model(
            "qwen-instruct", _record("qwen-instruct", "Qwen-3.8"), QWEN_3_8_ADAPTER
        )
        == QWEN_3_8_SPEC.family
    )
    assert (
        tokenizer_family_for_model(
            "deepseek-pro", _record("deepseek-pro", "DeepSeek-V4.1"), DEEPSEEK_V4_1_ADAPTER
        )
        == DEEPSEEK_V4_1_SPEC.family
    )


def test_inline_templates_compile_and_render_tool_prompts():
    # Inline templates are compiled without touching the artifact cache.
    assert load_template(GLM_5_3_SPEC) is not None
    assert load_template(QWEN_3_8_SPEC) is not None

    glm_prompt = official_tool_prompt_for_adapter(GLM_5_3_ADAPTER, [WEATHER_TOOL])
    assert "<tools>" in glm_prompt
    assert (
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>" in glm_prompt
    )

    qwen_prompt = official_tool_prompt_for_adapter(QWEN_3_8_ADAPTER, [WEATHER_TOOL])
    assert qwen_prompt.startswith("# Tools")
    assert "<function=example_function_name>" in qwen_prompt
    # The injected tool prompt carries no effort preamble; the effort directive
    # stays owned by the full-prompt/counting renderer.
    assert "Reasoning effort" not in qwen_prompt


def test_qwen38_counting_does_not_forward_reasoning_effort():
    # The 3.8 template raises for efforts outside xhigh/medium/low, so the
    # counting path must render it without an explicit reasoning_effort even
    # when a caller supplies one.
    prompt = render_chat_prompt(
        [{"role": "user", "content": "hi"}],
        QWEN_3_8_SPEC.family,
        add_generation_prompt=True,
        reasoning_config={"effort": "high"},
        thinking=None,
    )
    assert "Reasoning effort is set to xhigh." in prompt


def test_deepseek_v41_tool_prompt_uses_spaced_dsml_tags():
    prompt = official_tool_prompt_for_adapter(DEEPSEEK_V4_1_ADAPTER, [WEATHER_TOOL])
    assert "<｜DSML｜ calls>" in prompt
    assert '<｜DSML｜ invoke name="$TOOL_NAME">' in prompt
    assert "<｜DSML｜tool_calls>" not in prompt


def test_deepseek_v41_transport_splits_system_and_user_shells():
    transported = official_deepseek_transport_messages(
        DEEPSEEK_V4_1_ADAPTER,
        [{"role": "user", "content": "上海天气如何？"}],
        [WEATHER_TOOL],
    )
    assert [message["role"] for message in transported] == ["system", "user"]
    assert "<｜DSML｜ calls>" in transported[0]["content"]
    assert transported[1]["content"] == "上海天气如何？"


def test_deepseek_v41_reasoning_prefix_uses_numeric_budget():
    assert "Reasoning Effort: 100" in official_reasoning_prefix_for_adapter(
        DEEPSEEK_V4_1_ADAPTER, "max"
    )
    assert "Reasoning Effort: 75" in official_reasoning_prefix_for_adapter(
        DEEPSEEK_V4_1_ADAPTER, "high"
    )
    assert official_reasoning_prefix_for_adapter(DEEPSEEK_V4_1_ADAPTER, "none") == ""


def test_deepseek_v41_spaced_dsml_tool_calls_are_recovered():
    content = (
        "Let me check.\n"
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="get_weather">\n'
        '<｜DSML｜ parameter name="city" string="true">上海</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls>"
    )
    calls, remaining = extract_deepseek_tool_calls(
        content,
        tools=[WEATHER_TOOL],
        adapter=DEEPSEEK_V4_1_ADAPTER,
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "上海"}
    assert remaining == "Let me check."


def test_deepseek_v41_completion_serialization_uses_spaced_tags():
    rendered = _serialized_completion(
        {
            "content": "done",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "上海"}),
                    },
                }
            ],
        },
        DEEPSEEK_V4_1_SPEC.family,
        finish_reason="tool_calls",
        thinking=False,
    )
    assert "<｜DSML｜ calls>" in rendered
    assert '<｜DSML｜ parameter name="city" string="true">上海</｜DSML｜ parameter>' in rendered
    assert "</｜DSML｜ invoke>" in rendered


def test_tool_start_tags_use_the_bridge_marker_for_platform_hosted_models():
    from genai_proxy.models import BRIDGE_CALL_MARKER, BRIDGE_LINE_MARKER

    assert tool_start_tags(DEEPSEEK_V4_1_ADAPTER) == (
        BRIDGE_CALL_MARKER,
        BRIDGE_LINE_MARKER,
    )
    assert tool_start_tags(GLM_5_3_ADAPTER) == (
        BRIDGE_CALL_MARKER,
        BRIDGE_LINE_MARKER,
    )
    assert tool_start_tags(QWEN_3_8_ADAPTER) == (
        BRIDGE_CALL_MARKER,
        BRIDGE_LINE_MARKER,
    )
    # Previous generations keep their native syntax.
    assert "<｜DSML｜tool_calls>" in tool_start_tags(DEEPSEEK_V4_PRO_ADAPTER)


def test_bridge_injection_carries_tools_as_plain_text():
    messages = inject_bridge_tool_prompt(
        [{"role": "user", "content": "上海天气如何？"}],
        [WEATHER_TOOL],
    )
    assert messages[0]["role"] == "system"
    assert "CALLTOOL" in messages[0]["content"]
    assert "get_weather" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "上海天气如何？"}


def test_bridge_extraction_recovers_calls_and_strips_the_marker():
    content = (
        "Let me check the weather.\n"
        'CALLTOOL {"name": "get_weather", "arguments": {"city": "上海"}}\n'
        "done"
    )
    calls, remaining = extract_bridge_tool_calls(content, tools=[WEATHER_TOOL])
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "上海"}
    assert "CALLTOOL" not in (remaining or "")
    assert "Let me check the weather." in remaining


def test_bridge_extraction_accepts_the_line_shorthand():
    content = "I'll look around.\nRUNCMD cd /testbed && ls -la\n"
    calls, remaining = extract_bridge_tool_calls(content, tools=[BASH_TOOL])
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "command": "cd /testbed && ls -la"
    }
    assert "RUNCMD" not in (remaining or "")
    assert remaining == "I'll look around."


def test_bridge_line_shorthand_requires_an_unambiguous_tool():
    calls, remaining = extract_bridge_tool_calls(
        "RUNCMD ls", tools=[WEATHER_TOOL]
    )
    assert calls is None
    assert remaining == "RUNCMD ls"


def test_bridge_extraction_accepts_xml_json_shells():
    content = '<tool_call>{"name": "bash", "arguments": {"command": "pwd"}}</tool_call>'
    calls, remaining = extract_bridge_tool_calls(content, tools=[BASH_TOOL])
    assert calls is not None and len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "pwd"}
    assert remaining is None


def test_bridge_prompt_advertises_the_line_shorthand_for_shell_tools():
    prompt = bridge_tool_prompt([BASH_TOOL])
    assert "RUNCMD" in prompt
    assert "CALLTOOL" in prompt
    # A tool with several parameters keeps the JSON-only envelope.
    prompt = bridge_tool_prompt([WEATHER_TOOL])
    assert "RUNCMD" not in prompt


def test_bridge_extraction_ignores_unknown_functions():
    content = 'CALLTOOL {"name": "nope", "arguments": {}}'
    calls, remaining = extract_bridge_tool_calls(content, tools=[WEATHER_TOOL])
    assert calls is None
    assert remaining == content


def test_bridge_history_is_rendered_as_text():
    messages = inject_bridge_tool_prompt(
        [
            {"role": "user", "content": "上海天气如何？"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "上海"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "晴，26 度"},
            {"role": "user", "content": "谢谢"},
        ],
        [WEATHER_TOOL],
    )
    assert messages[0]["role"] == "system"
    assert "CALLTOOL" in messages[0]["content"]
    assistant = next(m for m in messages if m["role"] == "assistant")
    assert "CALLTOOL" in assistant["content"]
    results = [m for m in messages if m["role"] == "user" and "<tool_result>" in m["content"]]
    assert len(results) == 1
    assert "<tool_result>晴，26 度</tool_result>" in results[0]["content"]
    assert messages[-1] == {"role": "user", "content": "谢谢"}
