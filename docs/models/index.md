# 模型适配总览

模型适配不是只按请求中的字符串选择。代理先从 GenAI 模型目录取得实际记录，再综合
`aiType`、名称和描述字段判断模型家族与版本。这样即使公开 ID 没有写明版本，仍可区分
GLM 5.2/5.3、Qwen 3.5/3.8、DeepSeek V4 Flash/Pro/V4.1 和旧模型。

## 当前维护范围

| GenAI ID | Adapter | Prompt/codec | 视觉 | 上游 thinking 开关 |
| --- | --- | --- | --- | --- |
| `chatglm`（GLM-5.3-Flash） | `glm_5_3` | 内联 GLM 5.3-Flash 官方模板 | 否 | 无独立字段 |
| `chatglm`（GLM 5.2） | `glm_5_2` | GLM 5.2 官方模板（HF 固定 revision） | 否 | 无独立字段 |
| `deepseek-chat` | `deepseek_v4_flash` | DeepSeek V4 Flash 官方 encoder | 否 | 支持 |
| `deepseek-pro`（DeepSeek-V4.1） | `deepseek_v4_1` | 内联官方 `encoding.py`（带空格 DSML、数字 effort） | 否 | 支持 |
| `deepseek-pro`（DeepSeek-V4-Pro） | `deepseek_v4_pro` | DeepSeek V4 Pro 官方 encoder | 否 | 支持 |
| `qwen-instruct`（Qwen-3.8） | `qwen_3_8` | 内联 Qwen 3.8 官方模板 | 是 | 无独立字段 |
| `qwen-instruct`（Qwen-3.5） | `qwen_3_5` | Qwen 3.5 官方模板（HF 固定 revision） | 是 | 无独立字段 |
| `kimi-k3` | `kimi_k3` | Kimi K3 官方 codec，工具使用通道桥接 | 是 | 无独立字段 |

模型目录记录明确显示 `rootModelName` 不是 Xinference 时，代理不会应用这些 adapter。
没有匹配官方公开 tokenizer 的模型仍可走通用代理路径，但 token 数是兼容估算。

## 内联官方资源（GLM 5.3-Flash / Qwen 3.8 / DeepSeek V4.1）

这三个版本的官方模板/编码器**直接内联在仓库内**，不再从 Hugging Face 下载，也不再
固定 revision（GenAI 的编码规则很少变化，内联可避免版本兼容问题）：

- `models/glm53/codec.py`：GLM-5.3-Flash `chat_template.jinja`（原始 sha256 记录在文件头）。
- `models/qwen38/codec.py`：Qwen 3.8 `chat_template.jinja`。
- `models/deepseek_v41/official_encoding.py`：DeepSeek V4.1 官方 `encoding.py` 逐字内联。

token 计数复用已缓存的旧版 tokenizer（GLM 5.3 与 5.2 逐字节相同；Qwen/DeepSeek 新版
仅有少量新增 token，偏差可忽略）。模板/编码器仍会参与 token 计数，因此计数随官方
prompt 形状变化。

## 纯文本工具桥（CALLTOOL / RUNCMD）

GenAI 平台会在模型输入/输出两侧解析并丢弃**原生工具语法**：GLM/Qwen 的
`<tool_call>` 块和 DeepSeek 的 DSML 块都会在模型看到之前被剥离，或在回传客户端之前
被拦截（模型只会"说"要调用工具，然后流被截断）。因此 `glm_5_3`、`qwen_3_8`、
`deepseek_v4_1` 三个 adapter 改用平台不识别的纯文本桥：

- 工具定义与调用指令以文本注入 system 消息；
- 模型输出 `CALLTOOL {"name": ..., "arguments": {...}}` 行，每行一个调用；
- 当只有一个"shell 类"工具（名称/描述含 bash/shell/run/exec/command/terminal）且它
  只有一个必填 string 参数时，额外接受 `RUNCMD <命令>` 简写——实测 DeepSeek V4.1
  在长任务提示下更愿意使用这种形式；
- 解析器同时兼容 `<tool_call>{"name": ...}</tool_call>` 形式（平台不拦截该形状）；
- assistant 的历史工具调用与 `role=tool` 结果在回传上游时重新渲染为同样的文本格式。

**注意**：桥的遵循率是概率性的。实测在普通/短请求下三个模型都稳定产生工具调用
（已做流式、非流式与多轮往返验证）；但在长 agent 提示（如 SWE-bench 的任务模板）下
遵循率明显下降且随模型而异。代理为此内置了重试：bridge 适配器每次客户端请求最多尝试
`BRIDGE_TOOL_ATTEMPTS`（3）次，`tool_choice: required` 或响应为空时触发；重试期间
reasoning 会暂存到成功（或最后一次）再发出。**agent 客户端应显式发送
`tool_choice: required`**（mini-SWE-agent 用
`-c model.model_kwargs.tool_choice=required`），实测三个模型在长任务模板下均达到
3/3 的成功调用率，并各自跑通了 SWE-bench Verified 单实例端到端。

纯文本桥的措辞也影响遵循率：使用 `## Tools` 标题并转储 JSON Schema 会让模型回退到
原生语法（被平台吞掉），因此提示词刻意改用散文式动作描述。

实现见 `models/bridge_tools.py`；`tool_start_tags` 对这三个 adapter 返回
`("CALLTOOL", "RUNCMD")`。Kimi K3 出于同样的原因使用它自己的 `<k3_action>` 桥，
DeepSeek V4 的 `official_transport_messages`（官方编码器 + system/user 拆分）仍保留，
供平台行为恢复时切换。

## 共同约束

- 官方资源固定到完整 commit revision，并在使用前校验 SHA-256。
- 工具 prompt、token 计数和 completion 序列化使用同一个模型家族 codec。
- 图片只允许出现在 user 消息中，目前只交给 Qwen 3.5 或 Kimi K3。
- reasoning 增量会尽可能实时转发；候选工具语法在完整验证前不会暴露给客户端。
- 所有 GenAI 聊天请求都不发送 `chatGroupId`。

各模型的具体差异见：

- [GLM 5.2](glm-5.2.md)
- [DeepSeek V4 Flash 与 Pro](deepseek-v4.md)
- [Qwen 3.5](qwen-3.5.md)
- [Kimi K3](kimi-k3.md)

`src/genai_proxy/models/legacy/` 中的 MiniMax 等代码只保留历史兼容，不进入当前故障
回退目录或在线模型矩阵。
