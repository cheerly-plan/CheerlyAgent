# 渐进式记忆系统（L0–L4）评审清单

> 范围：L1 索引块 / L2 情景 / L3 候选晋升 / L4 归档 / 用户级开关。
> 阅读日期：2026-11-13。结论先行：**系统功能上是通的，不是坏管线**；
> 项目工作记忆里「默认关闭 / 候选绝不自动进 L1」的描述已过时，实际是
> 「恒定启用 + per-user 开关 + 后台自动晋升/归档」。以下是真实短板。

---

## 已验证正常（避免误改）

| 项 | 位置 | 结论 |
|---|---|---|
| episode 跨会话合并 / 证据累计 | `memory/episode_store.py:18` `write_episode` | 按 embedding 相似度合并，`hit_count` 求和、`sessions_seen` 取并集，晋升链可触发 |
| 召回累计热度 | `memory/episode_store.py:176` `search_episode` → `record_episode_hits` | 召回即 +1 热度、刷新 `last_seen_at`（带去重窗），TTL 续期有效 |
| 用户关闭记忆端到端生效 | `storage/auth_store.py:99,142` + `memory/setting_store.py:20` | `user.memory_enabled` 经 `LEFT JOIN agent_memory_setting` 读同一张表，关闭后不注入、不挂工具 |
| 自动晋升 / 归档已接调度 | `tasks/cleanup.py:68,77` | `review_all_pending_candidates` + `distill_all_users` 在后台周期跑 |
| 候选/归档按 user_id 隔离、出处门禁 | `api/routers/memory.py` 全程、`promotion.py:123` | 含数字候选无出处被 hold；人工批准仍过 `evaluate_candidate` |

---

## 短板清单

### M1 — `evaluate_candidate` 不检查 `hit_count`（配置旋钮半失效）
- **严重度**：中（契约不符 + 旋钮失效）
- **位置**：`src/ai_agent/memory/promotion.py:132-156`
- **现状**：
  ```python
  def evaluate_candidate(candidate, *, min_sessions=3, duplicate=False):
      if duplicate: return hold
      if needs_source(...) and not has_source(...): return hold
      if candidate.session_seen < min_sessions: return hold
      return promote
  ```
  只卡 `session_seen`，**从不看 `candidate.hit_count`**。
- **影响**：`memory_promotion_min_hits`（`config.py:441`）只在「episode→候选」创建时经 `evaluate_episode`（`promotion.py:127` 的 `hit_count >= min_hits`）生效；一旦成为候选，命中次数再低也能升 L1。旋钮对「晋升」环节是空操作。
- **修复建议**：给 `evaluate_candidate` 加 `min_hits` 参数并补 `hit_count < min_hits → hold`；`review_pending_candidates` / `approve_candidate` 透传 `settings.memory_promotion_min_hits`。默认配置（min_hits=3，候选创建已满足）下行为不变。
- **风险**：低（默认无行为变化）。**需补测试**：构造 `session_seen>=3` 但 `hit_count<min_hits` 的候选，断言 hold。

### M2 — `session_seen` 只数「被重新抽取」的会话，不数「被召回」的会话（设计缺口）
- **严重度**：中（召回证据被系统性低估，真实记忆可能永不晋升）
- **位置**：
  - 召回写证据：`src/ai_agent/memory/episode_store.py:176` `search_episode` → `record_episode_hits`（`promotion_store.py:20-48`）只 +1 `hit_count`、刷新 `last_seen_at`，**不往 `sessions_seen` 加当前会话**；
  - 只有写入时：`src/ai_agent/memory/promotion_store.py:50` `touch_episode_session`（由 `observe_episode` 在 `write_episode` 后调用）才把会话加进 `sessions_seen` 集合。
- **现状**：`session_seen` 仅随「LLM 重新抽取并落库」增长。一条被反复召回、但很少被重新抽取的记忆，可能永远凑不够 3 个「抽取会话」而无法晋升。
- **影响**：高频但「模型不主动复述」的稳定偏好/项目，沉淀不进 L1 画像。
- **修复建议**：召回命中时（带 `chat_session_id`）也把当前会话并入 `sessions_seen`。需在 `search_episode` 调用处透传 `chat_session_id`（目前 `_recall_block` / `loop.py:255` 调用未传），并在 `record_episode_hits` 增加会话并集更新；复用 `touch_episode_session` 的去重逻辑避免重复计数。
- **风险**：中（改变晋升行为）。**需补测试**：模拟「3 个不同会话各召回一次」→ 断言 `session_seen` 达 3；并回归确认不会因「长会话内多次召回」而刷满。

### M3 — `expand_memory` 归档块缩进错误（纯格式）
- **严重度**：低（PEP8 违规，功能无碍）
- **位置**：`src/ai_agent/tools/memory/memory.py:96-102`（`else:` 子句体多缩进一层，16 空格）
- **修复建议**：缩进收回到 12 空格。零风险，无需测试（或一行 lint 检查）。

### M4 — `search_archive` 翻页窗口偏小（边界 bug）
- **严重度**：低（仅大 offset 且 `k>batch` 时少取）
- **位置**：`src/ai_agent/memory/archive.py:70-89`
- **现状**：`store.list_archive_rows(uid, limit=settings.memory_archive_batch + start)` 取 `batch+start` 行，rank 后切 `ranked[start:start+k]`。当 `offset` 大、`k(=memory_archive_top_k=5)` 接近 `batch(=200)` 以外时不会触发；但深翻页 + 大 `k` 会少取。
- **修复建议**：读取窗口改为 `max(batch, start + k) + 1`，或分页直接走 DB `OFFSET`。低优先级。
- **风险**：低。

### M5 — 项目工作记忆文档与实际实现背离（会误导维护）
- **严重度**：中（文档即事实来源，错会引导错误修改）
- **位置**：`.workbuddy/memory/2026-11-12.md` 第三节「渐进式记忆」
- **现状**：文档称 `memory_progressive_enabled=False` / `memory_promotion_enabled=False` / `memory_archive_enabled=False`「默认全 False、关闭时 prompt 逐字节不变」、「候选绝不自动进 L1」。实际：
  - config 中**无**这些全局开关（仅 `memory_trace_enabled`）；
  - `_profile_prompt`（`loop.py:225`）注释明写「渐进式记忆**恒定启用**」，无条件走 `build_memory_index`；
  - 候选**会**经 `cleanup.py` 后台自动晋升（"候选绝不自动进 L1" 不成立）。
- **修复建议**：更新工作记忆，改为描述真实架构（恒定启用 + per-user `agent_memory_setting` 总开关 + 后台自动晋升/归档 + 默认不进 system）。不改代码。

---

## 建议动手顺序（待确认）
1. **M3**（缩进，5 分钟，零风险）
2. **M1**（补 `min_hits` 门禁，默认无行为变化，带测试）
3. **M5**（更新文档，避免后续误改）
4. **M2**（召回计会话，行为变更，需充分回归）
5. **M4**（翻页窗口，低优）

> 注：M1/M2 涉及晋升判定，改动需回归 `test_memory_progressive.py`(31) / `test_memory_promotion.py`(43) / `test_memory_archive.py`(36)，并确认默认配置下晋升行为不变。
