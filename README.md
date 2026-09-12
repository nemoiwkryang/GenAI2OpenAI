# GenAI2OpenAI

GenAI2OpenAI 把上海科技大学 GenAI 网页服务转换为 OpenAI Chat Completions、
OpenAI Responses 和 Anthropic Messages 兼容接口。项目重点处理模型专用 prompt、
tool calling、推理增量、视觉输入、token 计数、上游重试和模型目录缓存。

当前重点维护的 Xinference 模型是：

| GenAI 模型 ID | 实际模型 | 视觉 | Tool calling |
| --- | --- | --- | --- |
| `chatglm` | GLM 5.3 Flash | 否 | 内联官方模板 + 纯文本桥（`CALLTOOL`/`RUNCMD`） |
| `chatglm`（旧） | GLM 5.2 | 否 | 官方 Hugging Face 模板 |
| `deepseek-chat` | DeepSeek V4 Flash | 否 | 官方 `encoding_dsv4.py` |
| `deepseek-pro` | DeepSeek V4.1（552B） | 否 | 内联官方 `encoding.py` + 纯文本桥 |
| `qwen-instruct` | Qwen 3.8 | 是 | 内联官方模板 + 纯文本桥 |
| `kimi-k3` | Kimi K3 | 是 | GenAI 通道专用桥接协议 |

模型目录记录里的名称带版本号（如 `GLM-5.3-Flash`、`DeepSeek-V4.1`），代理据此把
`chatglm` / `qwen-instruct` / `deepseek-pro` 路由到新版本 adapter；旧版本记录仍走原
adapter。Kimi K3 与三个新版本都使用"平台不识别的纯文本协议"传递工具调用，因为
GenAI 通道会剥离各家原生工具语法。普通消息和 token 计数仍使用（内联或固定 revision
的）官方模板/编码器。聊天请求不会发送 `chatGroupId`。

## 快速开始

需要 Python 3.11 或更高版本，推荐使用 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
uv run main.py --token <genai-jwt>
```

也可以使用 keystore 自动登录和刷新 token：

```bash
uv run main.py --keystore /path/to/ids-passkey.keystore
```

使用 Docker Compose：

```bash
cp .env.example .env
# 在 .env 中设置 GENAI_TOKEN，或设置 KEYSTORE_PATH 和 KEYSTORE_HOST_PATH
./scripts/docker-compose.sh up -d --build
```

默认监听 `http://127.0.0.1:5000`。最小请求示例：

```bash
curl http://127.0.0.1:5000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chatglm","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

如果设置了代理 API key，还需添加 `Authorization: Bearer <proxy-key>`。

## 接口

| 接口 | 路径 |
| --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| OpenAI Responses | `POST /v1/responses` |
| Responses token 计数 | `POST /v1/responses/input_tokens` |
| Anthropic Messages | `POST /v1/messages` |
| Anthropic token 计数 | `POST /v1/messages/count_tokens` |
| 模型目录 | `GET /v1/models` |
| 健康检查 | `GET /health` |

## 文档

完整说明放在 [项目 Wiki](docs/index.md)：

- [安装与首次请求](docs/getting-started.md)
- [架构总览](docs/architecture/overview.md)
- [消息、prompt 与 token](docs/architecture/messages-prompts-tokens.md)
- [模型适配](docs/models/index.md)
- [配置与部署](docs/operations/configuration.md)
- [测试与验证](docs/development/testing.md)

## 本地验证

普通测试不会访问真实上游，也不会读取 keystore：

```bash
uv run pytest -q
uv run python -m compileall -q src tests
```

真实上游和 OMP 长任务测试必须显式提供 keystore，详见
[测试与验证](docs/development/testing.md)。

## 入口与包名

Python 包名保持为 `genai_proxy`，distribution 名称保持为 `genai`。可用入口：

```bash
uv run main.py --token <genai-jwt>
uv run python -m genai_proxy --token <genai-jwt>
uv run genai2openai --token <genai-jwt>
```

生产部署使用 `genai_proxy.wsgi:app`。启动日志会打印完整 commit hash、提交时间、
来源和脏工作区状态；Docker 镜像在构建时写入这些信息，不包含 `.git`。
