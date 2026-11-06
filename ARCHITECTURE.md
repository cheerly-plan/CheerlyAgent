# 架构总览

AI 智能体后端。按「功能分层 + 业务域拆分」组织，依赖方向单向：

```
core  ←  llm  ←  storage  ←  {kb, memory, report}  ←  tools  ←  agents  ←  api
                                            ↑                          ↑
                                            └──────────── tasks ────────┘
```

- 越靠左越底层，不允许反向 import（见下方「依赖规则」）。
- `prompts` 是纯文案层，不 import 任何业务域；skill 目录由编排层注入。
- `report` 是确定性流水线（零 LLM token），独立成域。

## 一级包职责

| 包 | 职责 | 关键子包 / 文件 |
|---|---|---|
| `core` | 配置、日志、异常、类型 | `config.py`、`logging.py` |
| `llm` | LLM 客户端与 JSON 修复 | `client.py`、`json_utils.py` |
| `storage` | 存储与认证基础设施 | `db.py`（连接池/schema）、`auth_store.py`、`embeddings.py`、`store.py`（全局 store 装配根） |
| `kb` | 知识库：摄取、检索、图谱 | `ingest/`（解析·切分·表格·OCR）、`retrieval/`（混合检索·精排·路由·引用）、`graph/`（实体/关系抽取与查询）、`store.py`、`pipeline.py` |
| `memory` | 长期记忆 | `session/`、`profile/`、`episode/`、`checkpoint/`、`short_term/`、`llm_calls.py` |
| `report` | 确定性研报流水线 | `parse/extract/compare/compose/validate/pdf`、`skills/`（声明式注册） |
| `tools` | 工具定义与注册 | `kb/`、`memory/`、`report/`、`misc/`、`registry.py`、`base.py` |
| `prompts` | 系统/工具提示词模板 | `chat.py`、`tools.py`、`builder.py`（纯文案，无业务 import） |
| `agents` | Agent 编排循环 | `loop.py`、`turn.py`、`budget.py` |
| `api` | FastAPI 端点 | `app.py`、`routers/`、`dependencies.py`、`middleware/` |
| `tasks` | 后台任务 | 与 `api` 同层，依赖 storage/llm/kb |
| `migrations` | 数据库迁移 | Alembic |

## 依赖规则

1. **单向分层**：底层包（`core`/`llm`/`storage`）不 import 上层；业务域（`kb`/`memory`/`report`）只 import `storage`，不互 import。
2. **`prompts` 不反向依赖业务域**：它提供模板与渲染函数；`skill_catalog()` 等业务数据由编排层（`agents/loop.py`、`tools/report_generate.py`）注入。
3. **子包通过 `__init__` 暴露公共 API**：如 `ai_agent.kb.retrieval` 再导出 `search_knowledge`、`mentioned_subjects` 等；调用方与测试从包根导入，内部子模块重排不破坏外部引用。子包只再导出函数/类，不遮蔽同名子模块。
4. **`storage.store` 是装配根（软环，刻意保留）**：`store.py` 的 `MemoryStore` 组装 kb 与 memory 的各 Store 类，因此 `storage → {kb, memory}`；而各域 Store 只依赖 `storage.db`（不依赖 `storage.store`），运行时无 import 错误。若强行把门面移出 storage 可破环，但装配根本就引用它组合的域，是常见合理模式，故保留。

## 重构历史

- **storage 析出**：原 `memory` 同时装「全局 DB 基础设施 + 认证 + 记忆」，导致 `kb ↔ memory` 包级互相依赖。析出 `storage` 后该硬环消除（`memory` 只剩记忆功能）。
- **kb 拆子包**：原 32 文件平铺，按子域拆 `ingest/`、`retrieval/`、`graph/`；`store.py`/`pipeline.py` 留根做编排与存取。
- **tools 拆子包**：原 14 文件平铺，按域拆 `kb/`、`memory/`、`report/`、`misc/`；`base.py`/`registry.py` 留顶层。
- **prompts 解倒挂**：原 `prompts.tools` 反向 import `report.skills` 拿 skill 目录；改为模板 + 渲染函数，目录由编排层注入。
