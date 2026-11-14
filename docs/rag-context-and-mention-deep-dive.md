# 知识检索链路深读：上下文组装与点名检测

> 本文是对 `ai_agent_backend` 真实代码的逐层深读，覆盖两条主线：
> **上下文是怎么组装的**（`agents/loop.py` / `memory/short_term.py` / `agents/turn.py` / `agents/messages.py`）
> 和 **点名检测是怎么实现的**（`kb/retrieval/retrieval.py` / `kb/ingest/meta.py` / `kb/graph/graph_store.py`）。
> 所有结论均出自代码与 docstring，行号以写作时仓库为准。

---

## 0. 代码索引

| 环节 | 模块 | 关键函数 |
|---|---|---|
| 流程入口 | `agents/loop.py` | `run_stream` / `run_loop` / `_must_call_kb_tool` / `_system_prompt` |
| 会话状态 | `agents/turn.py` | `_session_state` / `_persist_turn` / `_capture_turn` |
| 上下文装配 | `memory/short_term.py` | `assemble` / `fit_turns` / `compress` / `ShortTermMemory` |
| 送前裁剪 | `agents/messages.py` | `_fit_context` / `_keep_prefill_tail` |
| 静态提示词 | `prompts/chat.py` | `build_system_prompt`（PromptBuilder 分层） |
| 主体/实体检测 | `kb/retrieval/retrieval.py` | `mentioned_subjects` / `mentioned_entities` / `name_in_text` 调用点 |
| 名称规则 | `kb/ingest/meta.py` | `normalize_name` / `subject_core` / `extract_subject` / `fragment_subjects` |
| 图谱实体查询 | `kb/graph/graph_store.py` | `KgStore.query_mentions_entity` |

---

## 1. 一次问答的完整时序

以「中国能建 2026Q3 的营业收入是多少」为例，服务端发生的事按序为：

1. **`run_stream`（loop.py:327）** 被调用。先处理三种特殊模式：断点续跑（`resume_run_id`，从 checkpoint 快照恢复）、重新生成（`regenerate`，内存里剔除旧回答重问）、普通新消息。
2. **`_session_state`（turn.py:64）** 取回会话记忆：进程内热缓存优先（`short_term_memory.load`，TTL 字典），缓存没有再读数据库（`get_session_summary` + `load_recent_turns`，按 `max_turns` 取最近 N 轮）。
3. **`_system_prompt`（loop.py:297）** 组装系统提示词（详见 §2.2）。
4. **`assemble`（short_term.py:107）** 把 system / 摘要 / 近期轮次拼成 `messages`，当前问话追加在尾。
5. **`_checkpoint_create`** 落快照（断点续跑与「该会话有正在进行的回复」互斥都靠它），`yield run_started` 把 run_id 发给前端。
6. **`run_loop`（loop.py:598）** 进入工具循环：
   - 每轮先问 `_must_call_kb_tool`：点名了库内主体/实体且这句还没查过 → `tool_choice="required"`、工具列表收窄成 `kb_search`/`kg_search`（详见 §3）。
   - `_fit_context` 裁出送模型的副本 → `stream_chat` 流式调用 → 文本增量 `yield` 给前端。
   - 模型给出 `tool_calls` → 逐个执行（`execute_tool`），工具输出追加为 `tool` 消息；引用经 `collect_citations` 旁路收集。
   - 模型不再调工具 → 输出即最终回答，`_attach_sources` 把引用挂到回答消息，`yield done`。
7. **收尾（finally 块）**：turn 落库（新消息 append / 重新生成 replace）、热缓存 `short_term_memory.save`、长期记忆抽取（只从完整回合）与摘要压缩丢后台线程（`_turn_finalizer`），不阻塞 SSE 尾延迟。

---

## 2. 上下文组装

### 2.1 会话状态：热缓存与数据库两级

```python
# agents/turn.py:64
def _session_state(user_id, chat_session_id):
    cached = short_term_memory.load(chat_session_id, user_id=user_id)
    if cached is not None:
        return cached.messages, cached.summary      # 进程内命中，不打库
    summary = store.get_session_summary(...)         # DB：会话摘要
    recent = store.load_recent_turns(..., turns=short_term_memory.max_turns)
    return recent, summary
```

`ShortTermMemory`（short_term.py:186）是一个带锁的 `OrderedDict`：键 `(user_id, chat_session_id)`，值 `{messages, summary, used_at}`；后台清理线程按 TTL 过期、按 `max_sessions` LRU 淘汰。**热缓存写入前也走双重裁剪**（`save` 里 `fit_turns(compact(messages, max_turns), max_chars)`），保证缓存里的窗口不会比模型实际看到的更宽。

会话被删除/清空时必须 `invalidate` 同步失效，否则会出现「删掉的会话自己回来了」（docstring 原话）。

### 2.2 system 提示词的四块结构

`_system_prompt` = **静态核心 + 三个动态块**：

```python
core = build_system_prompt(use_kb=use_kb, report_catalog=skill_catalog())  # 静态核心
block = _profile_prompt(user_id)          # 用户画像（长期记忆）
cap_block = _kb_capability_block(...)     # 知识库能力（仅 use_kb 时）
tail = "\n\n".join(p for p in (_today_line(), block, cap_block) if p)
tail = clip_text(tail, settings.system_dynamic_max_chars, marker=_DYNAMIC_CLIP_MARK)
```

**静态核心**（`prompts/chat.py:83`）用 `PromptBuilder` 按**本轮实际挂载的能力**分层组装：identity / style / format 恒有；citation、kb 纪律、report 纪律三节 `when=use_kb`；memory 节 `when=use_memory`；truthfulness 恒有。设计理由写在 docstring 里：给一个没挂知识库的会话讲「必须先调 kb_search」，模型要么困惑、要么去调一个根本不存在的工具——**工具列表已在 loop.py 按开关裁剪，提示词必须同步**，否则两边打架。

**三个动态块**：

1. `_today_line()`：当前日期 + 「此日之前的报告期都是已披露文件」——专治模型按训练截止日判断「某年财报还没披露」。
2. `_profile_prompt`：用户画像条目格式化。
3. `_kb_capability_block`：按库里实际文档类型分组生成「图谱覆盖哪些实体/关系」，**文件清单本身不进提示词**（文件一多既撑爆上下文、模型也挑不准还会编造清单外文件名——这个职责移交给了文档路由）。类型超过 8 类只列文档最多的 8 类并显式提示「完整清单用 kb_list_files 查」。末尾附报告期清单（`_period_inventory`），同样是为了不让模型按训练知识推断披露进度。

动态尾部整体受 `system_dynamic_max_chars` 硬上限裁剪（保头保尾按行裁），裁掉时插入显式省略标记——让模型知道清单不完整、需要时自己查，而不是拿着半截清单硬答。

### 2.3 摘要伪装：为什么以及脆弱点

历史摘要不是 system 段落，而是**伪装成一轮 user/assistant 对话**插在 system 之后：

```python
# memory/short_term.py:107
def assemble(system, summary, recent):
    messages = [{"role": "system", "content": system}]
    if summary.strip():
        messages.append({"role": "user", "content": SUMMARY_USER_PREFIX + summary})
        messages.append({"role": "assistant", "content": SUMMARY_ASSISTANT})
    messages.extend(recent)
    return messages
```

读取侧 `take_summary`（short_term.py:28）靠**前缀字面值反解**：检查前两条是否是「user 带 `SUMMARY_USER_PREFIX` + assistant」，是则剥前缀取出摘要。short_term.py 顶部有明确警告：**`SUMMARY_USER_PREFIX` 改一个字就会静默丢掉全部历史摘要**（反解不中，摘要对被当成普通对话留在 rest 里，后续逻辑全部错位）。改这个常量必须同时改 `take_summary` 的判据。

### 2.4 双重裁剪：轮数窗口 + 字符预算

长度口径统一为 `messages_chars`（short_term.py:63）：正文 + tool_calls 参数的字符数粗估。**送模型前的裁剪与收尾时的摘要共用这一个口径**，避免「一边按字符裁、一边按轮数摘要」导致被裁掉的历史既不进窗口也不进摘要。

- `fit_turns`（short_term.py:77）：超预算时**从最旧的完整轮次开始整轮丢弃**，system 与摘要对永不参与丢弃，最后一个 turn（当前回合）永不丢。返回 `(保留轮次, 被丢消息)`——被丢的消息交给调用方并入摘要，**不静默蒸发**。
- `compress`（short_term.py:133）：轮数窗口（`keep`）裁一次 + 字符预算（`max_chars`）再裁一次，两道挤出来的 `overflow` 一起交摘要函数（LLM 摘要失败退回 `crude_summary` 拼接兜底）。docstring 点名了只按轮数压缩的坑：窗口内几轮大工具输出仍会超预算，被整轮丢掉的历史既没进摘要也没进长期记忆——「用户视角就是模型突然忘事」。

### 2.5 `_fit_context`：整轮丢弃与孤儿 tool 消息

`agents/messages.py:158` 是每次送模型前的最后一道闸：

- 只裁**副本**，`messages` 本体不动（落库、checkpoint 快照都依赖原始列表）。
- 整轮（user + 其后的 assistant/tool）一起丢：保证 `assistant(tool_calls)` 与它的 tool 结果永远成对，**孤儿 tool 消息会被聊天 API 直接 400**。
- 裁完重新 `assemble`（system + 摘要对 + 保留轮），再过一遍 `_enforce_budget` 兜底。

### 2.6 运行中追加的消息

组装只发生在回合开头；回合进行中 `messages` 还会追加：

- **图谱邻域预注入**（loop.py:526 `_attach_kg_prefetch`）：关系类问句在模型选工具之前，把图谱邻域作为一条 system 消息（`【图谱邻域】` 开头）塞进上下文；续跑时检测到已注入过就不再查一次（`_kg_prefetch_present`）。
- **assistant 消息**（含 `tool_calls`）与 **tool 结果消息**：工具循环的每一轮。
- **运行时纠偏 system 消息**（不渲染、不污染气泡）：空回答追讨（`RUNTIME_EMPTY_ANSWER`）、重复调用提示（`RUNTIME_DUP_CALL`/`RUNTIME_DUP_OUTPUT`）、预算耗尽强制收尾（`RUNTIME_BUDGET_EXHAUSTED`）。

每步之后 `_checkpoint_update` 增量落快照，断线重连/继续生成（`resume_run_id` + `_keep_prefill_tail` 的 partial 标记）都基于它。

### 2.7 回合收尾

finally 块按序做四件事：判定终态（completed/failed/interrupted）→ turn 落库（普通 append / 重新生成与续跑成功后 replace，**失败或中断保留库里已有内容**）→ 快路径存热缓存 → 慢路径（长期记忆抽取、摘要压缩，含多次 LLM 调用）提交后台线程。两个纪律：

- **长期记忆只从完整回合抽取**：中断/失败的半截回答进记忆库，之后会被当成「用户说过的话」反复召回，污染长期画像。摘要压缩照跑（窗口管理，半截留在短期上下文里保持连续）。
- `finalize_background=True` 时先同步完成落库 + 热缓存，保证下一轮上下文连续，慢路径不阻塞 SSE 尾部。

---

## 3. 点名检测

### 3.1 入口与守卫条件

`_must_call_kb_tool`（loop.py:548）决定「本步是否强制调用知识库工具」。先过四道守卫，任何一道不满足直接放行（返回 False）：

1. 工具列表里得有 `kb_search` / `kg_search`（知识库开关关了就不强制）；
2. **这句问话之后还没有工具结果**（`_question_already_retrieved`：续跑、或模型已经查完准备作答时不再强制）；
3. 问句非空；
4. 不是「生成一份研报」类请求（`_REPORT_VERB` + `_REPORT_NOUN` 同时命中才豁免——「生成/出一份/写一份」+「研报/分析报告/PDF」；只中一个不放行，避免误伤「研报里说了什么」这种检索题）。

然后做三通道匹配，任一命中 → 本步 `tool_choice="required"`、`step_tools` 收窄为两个检索工具。**只强制「必须查」，选 kb_search 还是 kg_search 仍由模型**按系统提示词的【知识库能力】块决定。读库失败（异常）也放行——检测是加速器不是闸门，故障不能挡住对话。

### 3.2 通道一：主体整段匹配（`mentioned_subjects`，retrieval.py:627）

对库里每份 `status="ready"` 的文件，取**三个名字来源**：

1. `subject`：主体全称（入库时规则抽取，见下）；
2. `subject_aliases`：别名表（含简称——**简称无法由规则推导**，「中国能源建设」→「中国能建」这种缩写由图谱对齐产出后写回文件）；
3. `_file_subject(filename)`：文件名去噪主体。

每个名字过 `name_in_text`（meta.py:202）：

```python
def name_in_text(text, name, *, min_len=2):
    hay = normalize_name(text)
    for token in (normalize_name(name), normalize_name(subject_core(name))):
        if len(token) >= min_len and token in hay:
            return True
    return False
```

语义要点：

- **整段子串判断，刻意不做最长公共子串**。「中国能建」不会因为和「中国能源建设股份有限公司」共享「中国能」三个字而命中——必须名字本身（全称/核心名/归一化形/别名）整段出现在归一化后的问句里。宁可漏（漏了由通道二/文档路由兜）也不误（误了会强制检索无关库）。
- `normalize_name`（meta.py:40）= 抹平标点/空白/全半角/大小写（`NAME_PUNCT` 正则），**只用于比对不用于展示**。
- `subject_core` = 剥组织形式后缀（`ENTITY_SUFFIX`：股份有限公司/有限责任公司/有限公司/集团公司/集团/公司，锚定结尾）。

**subject 本身从哪来**：入库时 `extract_subject`（meta.py:50）对文件名做规则剥壳——去扩展名（`EXT_RE`）→ 去噪声（`FILENAME_NOISE`：年份、第X季度、半年度/季度/年度报告、公告）→ 去残留标点。「中国能源建设股份有限公司2026年第三季度报告.pdf」→「中国能源建设股份有限公司」。封面全称（`cover_company`/`dominant_company`，后缀前最多 24 字防吞导语）只在文件名抽不出合理主体时补全。**整条抽取链零 LLM**——docstring 的理由：规则可回放、可解释、零成本，且「抽错主体比抽不到更糟（该主体的切片整体查不到）」，不能交给模型抽。

### 3.3 通道二：独有片段匹配（`fragment_subjects`，meta.py:334）

补别名表没登记的简称。算法：

1. `_name_windows`：把问句归一化后，取**连续汉字段的 4–8 字滑窗**（所有起点 × 所有长度），过滤 `_GENERIC_NAME_FRAGMENTS` 黑名单（股份有限、有限公司、控股股份……这些四字词每家公司都有，拿来当简称会点名全场）；
2. 对每个主体（含别名）取归一化核心名，只留长度 ≥ 4 的；
3. 某窗口是**恰好一家**主体核心名的真子段（`frag in core and frag != core`）→ 点名了这家；**同一段字出现在两家及以上核心名里则放弃**，避免把无关公司拉进来。

例：「中科飞测」是「深圳中科飞测科技」核心名里独有的一段、库里没有第二家含这四个字 → 命中。这条通道与通道一互补：通道一管「登记过的写法」，通道二管「名字的一段但全局无歧义」。

### 3.4 通道三：图谱实体字面查询（`mentioned_entities` → `KgStore.query_mentions_entity`，graph_store.py:32）

补人名、子公司、产品这类**不在文件 subject 上**的名字。落到 SQL：

```sql
SELECT DISTINCT name FROM (
    SELECT e.name FROM agent_kg_entity e
    WHERE char_length(e.norm_name) >= 2
      AND char_length(e.norm_name) <= char_length(%(q)s)
      AND strpos(%(q)s, e.norm_name) > 0
      AND (file_ids 限定…)
    UNION
    SELECT a.alias_name FROM agent_kg_alias a
    WHERE … 同样的整段包含判断 …
) hits LIMIT 8
```

要点：问句先 `_norm_text` 归一化；实体规范名与别名两路都做 `strpos` 整段包含；名字短于 2 字不参与（单字会撞上整句）；`LIMIT 8` 控制返回量；`file_ids` 可把范围限定在用户圈选的文件内。**同样不做模糊**——「中国能建」对不上图谱里的全称，那条路归文件主体（通道一/二）管。

### 3.5 强制的语义与边界

- **只看当前这句 user 消息**：上一轮查过别的对象不算数（防止跨轮误强制）；这句之后已有工具结果（续跑场景）不再强制。
- 命中主体后**实体检测照跑**：问「公司 A 里的实体 B」时，日志里要能看到 B，不能因为 A 命中就跳过。
- 命中即收口的是「必须调一个检索工具」；检索范围、选哪个工具、查几次，仍是模型的自由。
- 通道一/二在 Python 内存里对文件清单做匹配（一次 `list_kb_files`），通道三打一次库；整体成本毫秒级，远小于一次 LLM 调用。

### 3.6 失效模式与防御（现状归纳）

| 失效模式 | 现有防御 |
|---|---|
| 简称不在别名表、也不构成独有片段 → 不强制 | 不强制 ≠ 不能查：模型仍可自主调 kb_search；文档路由（`router.py`）的 `fragment_subjects` 复用同一套判据圈文件 |
| 误命中（字面撞车，如「深圳带伞」撞「深圳市顺丰投资」） | 检索侧另有 `query_looks_on_topic` 主体/标签级复核；点名检测本身用「整段 + 独有性 + 黑名单」三重收窄 |
| 检测读库失败 | 捕获异常放行，绝不挡对话 |
| 别名表覆盖不足 | 图谱对齐持续产出别名写回文件（`extract_aliases` 落全称/核心名/两归一化形四种写法） |

---

## 4. 设计原则提炼

1. **能力与提示词同裁**：工具列表按开关裁，提示词纪律节同步裁——两边描述同一个现实。
2. **一个口径管到底**：字符预算在「送模型裁剪 / 热缓存写入 / 摘要压缩」三处共用 `messages_chars` + `fit_turns`，历史要么在窗口、要么在摘要，绝不静默蒸发。
3. **整轮原子性**：轮（user+assistant+tool）是裁剪与落库的最小单位，孤儿 tool 消息会 400，半截回答不进长期记忆。
4. **规则优先于 LLM 的地方**：元数据（subject/period/别名匹配）全部规则化——可回放、零成本、错了可解释；「抽错比抽不到更糟」的领域尤其如此。
5. **宁可漏不可误**：点名检测整段匹配 + 独有性 + 黑名单，漏了有文档路由和模型自主检索兜底，误了会强制查无关库。
6. **检测是加速器不是闸门**：任何检测故障都放行，对话可用性优先。
