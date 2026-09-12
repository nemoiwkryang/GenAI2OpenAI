"""Plain-text tool bridge for models whose native tool syntax is intercepted.

The GenAI platform parses and drops native tool-call markers in both
directions: GLM/Qwen ``<tool_call>`` blocks and DeepSeek's DSML blocks are
stripped before the model sees them (or before the client receives them), so
those models answer with narration only when tools are requested. Markers the
platform does not recognize survive intact, so the bridge carries tool
definitions, calls and results as ordinary text.
"""

import json
import re
import uuid

BRIDGE_CALL_MARKER = "CALLTOOL"
BRIDGE_LINE_MARKER = "RUNCMD"

# Deliberately avoids a "## Tools" / JSON-schema framing: dumping raw function
# schemas makes the platform-hosted models fall back to their native tool
# syntax, which the platform then swallows (the model narrates the call and the
# stream ends). A prose listing plus concrete examples keeps them on this
# format. Models differ in which envelope they will follow, so two are offered
# and both are parsed.
BRIDGE_TOOL_PROMPT_TEMPLATE = """## Actions

You can call the following actions to help answer the user's request.

To call an action, reply with a single line:

CALLTOOL {{"name": "<action-name>", "arguments": {{<arguments-as-json>}}}}

Example: {example_call}
{line_format}
Available actions:
{action_list}

Rules:
1. Output one CALLTOOL line per action call; multiple lines call multiple actions.
2. "arguments" must be a JSON object using the parameter names and types listed above.
3. Never use XML tags, DSML, or any other call syntax.
4. Call an action only when it is needed to answer the current request; otherwise answer normally in plain text.
5. After a result is provided, either call more actions or give the final answer."""

BRIDGE_LINE_FORMAT_TEMPLATE = """
For the {name} action you can also reply with exactly one line:

{marker} <{value_hint}>

Example: {line_example}
"""

BRIDGE_REQUIRED_TOOL_SUFFIX = (
    "\n\nFor this turn, you must call at least one action with a CALLTOOL line."
)
BRIDGE_SPECIFIC_TOOL_SUFFIX = (
    '\n\nFor this turn, you must call the action named "{name}" with a CALLTOOL line.'
)
BRIDGE_NO_TOOL_SUFFIX = (
    "\n\nFor this turn, do not call any action and do not output a CALLTOOL line."
)

_CALL_START = re.compile(r"CALLTOOL\s*", re.IGNORECASE)
_LINE_START = re.compile(r"(?m)^[ \t]*RUNCMD[ \t]+(?P<value>.+?)[ \t]*$")
_XML_CALL = re.compile(
    r"<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

_SHELL_HINTS = ("bash", "shell", "run", "exec", "command", "terminal")


def _is_shell_like(tool: dict) -> bool:
    function = tool.get("function", {})
    text = " ".join(
        str(function.get(field) or "") for field in ("name", "description")
    ).lower()
    return any(hint in text for hint in _SHELL_HINTS)


def _line_tool(function_tools: list[dict]) -> tuple[dict, str] | None:
    """Pick the single shell-like tool that the RUNCMD shorthand can address."""
    candidates = []
    for tool in function_tools:
        if not _is_shell_like(tool):
            continue
        function = tool.get("function", {})
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = list(parameters.get("required") or properties)
        if len(properties) == 1 and len(required) == 1:
            key = required[0]
            if (properties.get(key) or {}).get("type") == "string":
                candidates.append((tool, key))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _function_tools(tools) -> list[dict]:
    return [
        tool
        for tool in tools or []
        if isinstance(tool, dict) and tool.get("type") == "function"
    ]


def _action_list(function_tools: list[dict]) -> str:
    lines = []
    for tool in function_tools:
        function = tool.get("function", {})
        name = function.get("name", "")
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or ())
        rendered = []
        for key, schema in properties.items():
            kind = (schema or {}).get("type", "any")
            suffix = ", required" if key in required else ""
            rendered.append(f"{key} ({kind}{suffix})")
        signature = ", ".join(rendered) if rendered else "no parameters"
        description = function.get("description") or ""
        lines.append(f"- {name}: {description} Parameters: {signature}.")
    return "\n".join(lines)


def _example_call(function_tools: list[dict]) -> str:
    function = function_tools[0].get("function", {})
    name = function.get("name", "<action-name>")
    parameters = function.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = list(parameters.get("required") or properties)
    arguments = {}
    for key in required[:2]:
        kind = (properties.get(key) or {}).get("type")
        arguments[key] = {
            "string": "<value>",
            "integer": 0,
            "number": 0,
            "boolean": False,
            "array": [],
            "object": {},
        }.get(kind, "<value>")
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)


def bridge_tool_prompt(tools, tool_choice=None) -> str:
    function_tools = _function_tools(tools)
    if not function_tools:
        return ""
    line_format = ""
    line = _line_tool(function_tools)
    if line is not None:
        tool, key = line
        line_format = BRIDGE_LINE_FORMAT_TEMPLATE.format(
            name=tool.get("function", {}).get("name", ""),
            marker=BRIDGE_LINE_MARKER,
            value_hint=key,
            line_example=f"{BRIDGE_LINE_MARKER} <the {key}>",
        )
    prompt = BRIDGE_TOOL_PROMPT_TEMPLATE.format(
        example_call=_example_call(function_tools),
        line_format=line_format,
        action_list=_action_list(function_tools),
    )
    return prompt + _tool_choice_suffix(tool_choice)


def inject_bridge_tool_prompt(
    messages,
    tools,
    tool_choice=None,
    adapter=None,
    reasoning_config=None,
):
    """Render tool definitions and prior tool history as plain text."""
    tool_prompt = bridge_tool_prompt(tools, tool_choice)
    new_messages = []
    has_system = False
    index = 0

    while index < len(messages):
        message = messages[index]
        role = message.get("role")

        if role == "system" and not has_system:
            content = message.get("content", "")
            new_messages.append(
                {
                    **message,
                    "content": tool_prompt + ("\n\n" + content if content else ""),
                }
            )
            has_system = True
            index += 1
            continue

        if role == "assistant" and message.get("tool_calls"):
            parts = []
            if message.get("content"):
                parts.append(str(message["content"]))
            parts.append(_render_tool_calls(message["tool_calls"]))
            new_messages.append(
                {"role": "assistant", "content": "\n".join(parts).strip()}
            )
            index += 1
            continue

        if role == "tool":
            results = []
            while index < len(messages) and messages[index].get("role") == "tool":
                result = _normalize_tool_content(messages[index].get("content"))
                results.append(f"<tool_result>{result}</tool_result>")
                index += 1
            new_messages.append(
                {"role": "user", "content": "\n".join(results)}
            )
            continue

        new_messages.append(message)
        index += 1

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": tool_prompt})
    return new_messages


def extract_bridge_tool_calls(content, tools=None, logger=None):
    if not content:
        return None, content
    lowered = content.lower()
    if (
        BRIDGE_CALL_MARKER.lower() not in lowered
        and BRIDGE_LINE_MARKER.lower() not in lowered
        and "<tool_call>" not in lowered
    ):
        return None, content

    function_tools = _function_tools(tools)
    tool_names = {
        str(tool.get("function", {}).get("name", "")).casefold()
        for tool in function_tools
    }
    found = []

    position = 0
    while True:
        match = _CALL_START.search(content, position)
        if not match:
            break
        parsed, end = _parse_json_object(content, match.end())
        if parsed is None:
            position = match.end()
            continue
        call = _validated_call(parsed, tool_names, logger)
        if call is None:
            position = match.end()
            continue
        found.append((match.start(), end, call))
        position = end

    line = _line_tool(function_tools)
    if line is not None:
        tool, key = line
        name = tool.get("function", {}).get("name", "")
        for match in _LINE_START.finditer(content):
            value = match.group("value").strip()
            if not value:
                continue
            found.append(
                (
                    match.start(),
                    match.end(),
                    _make_call(name, {key: value}),
                )
            )

    for match in _XML_CALL.finditer(content):
        try:
            parsed = json.loads(match.group("body"))
        except json.JSONDecodeError:
            continue
        call = _validated_call(parsed, tool_names, logger)
        if call is None:
            continue
        found.append((match.start(), match.end(), call))

    if not found:
        return None, content
    found.sort(key=lambda item: item[0])
    calls = [call for _, _, call in found]
    remaining = _remove_spans(content, [(start, end) for start, end, _ in found]).strip()
    if logger:
        logger.debug("Recovered %d bridge action call(s)", len(calls))
    return calls, remaining or None


def _validated_call(parsed, tool_names, logger):
    name = parsed.get("name")
    if not isinstance(name, str) or not name:
        return None
    if tool_names and name.casefold() not in tool_names:
        if logger:
            logger.warning("Ignoring action call for unknown name %r", name[:64])
        return None
    arguments = parsed.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {"arguments": arguments}
    return _make_call(name, arguments)


def _make_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _tool_choice_suffix(tool_choice) -> str:
    if tool_choice == "required":
        return BRIDGE_REQUIRED_TOOL_SUFFIX
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return BRIDGE_SPECIFIC_TOOL_SUFFIX.format(
            name=tool_choice["function"]["name"]
        )
    if tool_choice == "none" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
    ):
        return BRIDGE_NO_TOOL_SUFFIX
    return ""


def _render_tool_calls(tool_calls) -> str:
    lines = []
    for call in tool_calls:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, dict):
            try:
                parsed = json.loads(arguments or "{}")
            except (TypeError, json.JSONDecodeError):
                parsed = {"arguments": arguments}
            arguments = parsed if isinstance(parsed, dict) else {"arguments": parsed}
        lines.append(
            f"{BRIDGE_CALL_MARKER} "
            + json.dumps(
                {"name": function.get("name", ""), "arguments": arguments},
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _parse_json_object(text: str, start: int):
    """Return (object, end_index) for the first JSON object at or after start."""
    index = start
    while index < len(text) and text[index] in " \t\r\n`":
        index += 1
    if index >= len(text) or text[index] != "{":
        return None, index
    depth = 0
    in_string = False
    escaped = False
    for cursor in range(index, len(text)):
        char = text[cursor]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[index : cursor + 1]), cursor + 1
                except json.JSONDecodeError:
                    return None, cursor + 1
    return None, len(text)


def _normalize_tool_content(content) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _remove_spans(content: str, spans) -> str:
    if not spans:
        return content
    parts = []
    cursor = 0
    for start, end in sorted(spans):
        parts.append(content[cursor:start])
        cursor = end
    parts.append(content[cursor:])
    return "".join(parts)


__all__ = [
    "BRIDGE_CALL_MARKER",
    "BRIDGE_LINE_MARKER",
    "BRIDGE_TOOL_PROMPT_TEMPLATE",
    "bridge_tool_prompt",
    "extract_bridge_tool_calls",
    "inject_bridge_tool_prompt",
]
