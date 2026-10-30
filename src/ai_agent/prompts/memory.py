"""记忆链路的提示词：长期记忆抽取、画像写入、情景合并、会话摘要。

迁移原则：**文案一字未改**。这些措辞是针对具体误判调出来的（例如「问董事长
≠ 用户是董事长」就是模型把检索对象当成用户身份之后补的），重写风险远大于收益。
这里做的只是把它们聚到一处，并把原先散在 f-string 里的内插改成显式槽位。

一处实质改动：`FACET_WRITE_RULES` 原来由 `long_term.py` 和 `profile.py`
各自 f-string 内插同一份常量——两处拼装方式不同，改规则时容易只改一个。
现在统一走 `{{FACET_WRITE_RULES}}` 槽位，由 `render()` 注入。
"""

from __future__ import annotations

from ai_agent.prompts.builder import render

# ——— 画像写入规则（被抽取器和写入器共用） ———

FACET_WRITE_RULES = """
facet 行为：
- name / language / locale：互斥，新事实直接替换旧内容，不要拼接历史；旧值不写回当前画像
- locale：只写当前地区/居住地。已有「北京」、新事实「搬到上海」→ locale 写成「上海」，不要写成「曾住北京，现住上海」
- occupation：只写当前实际从事的职业，可附必要的过往历史。必须区分「已完成转行」和「打算转行」：
  * 已完成（已经转行 / 现在做 X / 刚换到 X 岗位）→ 更新现状。已有「画家」、新事实「已经转行做前端」→ occupation 写成「曾是画家，现为前端开发工程师」
  * 只是意向（打算 / 计划 / 准备 / 想 / 考虑 转行做 X）→ 这不是事实，occupation 保持当前职业不变，严禁写成「现为 X」；意向可写入 other，如「计划转行做 Agent 开发」
- preference / constraint：可同时成立多条独立偏好/约束，每条单独输出一个 add，不要合并成一段、也不要覆盖已有条目；只有用户明确否定某条（「我不再…」「其实我不喜欢…」）才 delete 那一条，delete 时 content 写被否定条目的原文
- other：其他稳定事实；姓名、职业、地区、语言不要写进 other
转行（已发生的）、工种、职业身份一律 occupation。「想学后端 / 准备考证」这类学习或意向不是 occupation。
已有 occupation 已覆盖该转行/工种时 skip，不要再写 other。
本轮如果只改了姓名，不要把已有职业再写一遍。
""".strip()

# ——— 长期记忆抽取 ———

EXTRACT_TEMPLATE = """
你是用户长期记忆抽取器。只记「关于用户本人、且下一场对话还用得上」的事实。
只输出 JSON，不要解释。格式：
{"memories":[{"kind":"episode","content":"..."},{"kind":"profile","op":"add"|"update"|"delete","facet":"name"|"occupation"|"locale"|"language"|"preference"|"constraint"|"other","content":"..."}]}
没有可记内容时必须输出 {"memories":[]}。拿不准就空数组。

kind:
- profile: 用户用第一人称说清的姓名、职业、居住地、语言、长期偏好、约束
- episode: 用户自己正在做、会跨会话延续的任务（项目、已确认的技术方案、明确要求记住的进展）

严禁写入：
- 本轮问答摘要（「用户询问了…」「讨论了某公司…」「助手建议…」）
- 查知识库/招股书/年报/公告得到的公司事实、股东、财务数字
- 把用户问过的公司写成用户的职业或身份（问董事长 ≠ 用户是董事长）
- 一次性查询：天气、股价、某家公司概况、某页表格
- 助手建议、工具调用过程
- 已有相同事实、只是换种说法、闲聊

episode 的 content 约 40～120 字，写用户本人近况，不要写成对话纪要。
profile 必须带 op 和 facet。name / occupation / locale / language 同一侧面只输出一条：
- add: 新的侧面
- update: 同一侧面事实变化或更完整，content 写成该侧面的完整当前描述
- delete: 用户明确否定旧事实
preference / constraint 是累积型：每条独立事实单独输出一个 add，可多条并存，不要 update 合并已有条目；否定某一条时 delete 并在 content 写该条原文。
{{FACET_WRITE_RULES}}
若已有 other 把姓名/职业混在一起，要拆开写入正确 facet，并 delete other。

规则：
1. 只记用户明确陈述的自身事实，或双方已确认、属于用户本人的技术选择
2. 身份和近况拆开
3. 最多 2 条
"""

# Jev 门控：本轮是否值得抽取长期记忆
GATE_INSTRUCTIONS = (
    "这段对话是否包含值得写入跨会话长期记忆的信息？"
    "值得记忆：用户用第一人称明确陈述的姓名、职业、身份、居住地、语言、"
    "长期偏好或约束；用户自己正在做且会跨会话延续的任务或技术方案。"
    "不值得记忆：查文档或问某家公司、天气、新闻、一次性事实查询、"
    "本轮问答摘要、助手建议、把检索结果当成用户身份。"
)

# ——— 画像写入 ———

RESOLVE_TEMPLATE = """
你是用户画像写入器。根据已有画像和新事实，决定如何更新。
只输出 JSON，不要解释。格式：
{"actions":[{"op":"add"|"update"|"delete"|"skip","facet":"name"|"occupation"|"locale"|"language"|"preference"|"constraint"|"other","content":"..."}]}

facet:
- name: 姓名
- occupation: 职业或长期身份
- locale: 地区或居住地
- language: 语言
- preference: 长期偏好
- constraint: 长期约束
- other: 其他稳定事实

{{FACET_WRITE_RULES}}

规则：
1. 一条输入可拆成多个 actions：姓名用 name，职业用 occupation
2. 已有 other 把姓名/职业混在一起时，要拆开写入正确 facet，并 delete other
3. 已有相同事实或只是换种说法：skip
4. 用户明确否定旧事实：delete
5. 不要把当前任务写成画像
6. preference / constraint 的独立新事实输出 add 新条目，不要 update 合并已有条目
"""

# Jev 语义去重：两条画像是否同一事实
DUP_INSTRUCTIONS = (
    "判断两条用户画像条目是否表达同一个事实或偏好（意思相同或高度相近，"
    "只是措辞不同也算）。是则输出高概率，不是则输出低概率。"
)

# ——— 情景记忆合并 ———

MERGE_SYSTEM = """
你是情景记忆合并器。把同一主题的多条近况合成一段完整、可检索的情景记忆。
规则：
1. 只输出合并后的正文，不要标题、前缀、解释、列表或引号；
2. 保留任务、技术栈、目标、进展等不重复关键事实，不要编造；
3. 去掉重复表述；
4. 写成 40～120 字的一两句话，不要压成口号。
"""

MERGE_USER_TEMPLATE = "请合并这些情景记忆：\n{{ITEMS}}"

# ——— 会话摘要 ———

SUMMARY_SYSTEM = """
你是会话摘要器。把对话压缩成后续回复仍需要的上下文摘要。
规则：
1. 只输出摘要正文，不要标题、前缀或解释；
2. 保留关键事实、用户偏好、已做约定、未完成事项和重要结论；
3. 去掉寒暄、重复内容和工具调用细节；
4. 一次性时效信息（天气、汇率、股价、临时查询结果）不写入摘要——
   它们很快过时，留在摘要里会误导后续回复；
5. 不要编造对话中没有的信息；
6. 使用简洁中文陈述句，控制在 400 字以内。
"""

SUMMARY_USER_TEMPLATE = "已有摘要：\n{{PREVIOUS}}\n\n新增对话：\n{{SOURCE}}"
SUMMARY_USER_FIRST_TEMPLATE = "对话内容：\n{{SOURCE}}"

# 摘要质量校验（Jev 双 Noul 并行）。摘要会每轮注入对话开头直到会话结束，
# 一条编造内容会持续污染后续上下文，所以生成后加一道独立第二意见。
FABRICATION_INSTRUCTIONS = "摘要是否包含对话中不存在、也无法从对话合理推断的信息（编造）"
OMISSION_INSTRUCTIONS = "摘要是否遗漏了对话中已确认的约定、偏好或未完成事项"

# 摘要预过滤（Jev 逐轮 Noul）：溢出轮次先判断有无保留价值
TURN_KEEP_INSTRUCTIONS = (
    "这一轮对话是否包含值得保留进会话摘要的信息？"
    "值得保留：关键事实、用户偏好、已做约定、未完成事项、重要结论。"
    "不值得保留：寒暄、闲聊、一次性时效信息（如天气、汇率、临时查询结果）、"
    "工具调用的中间过程。"
)

RETRY_FEEDBACK_TEMPLATE = (
    "质量校验发现上一版摘要{{PROBLEMS}}。"
    "请严格基于对话原文重新生成摘要，不得添加原文没有的信息。"
)

# ——— 短期记忆：摘要伪造成一轮对话注入 ———
#
# 注意：SUMMARY_USER_PREFIX 的字面值被 short_term.take_summary() 用来反解
# 摘要（按前缀切片）。改一个字就会静默丢掉全部历史摘要，测试中已冻结。
SUMMARY_USER_PREFIX = "此前对话摘要：\n"
SUMMARY_ASSISTANT = "已了解此前对话摘要，会据此继续当前会话。"

# 画像注入块（每轮塞进 system 消息尾部）
PROFILE_BLOCK_HEADER = "用户画像（稳定偏好/约束）："

# ——— user 侧数据拼装标签 ———

LABEL_SESSION_SUMMARY = "会话摘要："
LABEL_EXISTING_PROFILE = "已有用户画像："
LABEL_EXISTING_PROFILE_EMPTY = "已有用户画像：无"
LABEL_EXISTING_EPISODES = "已有相近情景记忆："
LABEL_EXISTING_EPISODES_EMPTY = "已有相近情景记忆：无"
LABEL_CURRENT_TURN = "本轮对话："
LABEL_NEW_FACT = "新事实："
LABEL_INHERITED_SUMMARY = "已有摘要（其内容并入新摘要属正常继承，不算编造）："
LABEL_ORIGINAL_DIALOG = "对话原文："
LABEL_SUMMARY_TO_CHECK = "待检摘要："


def build_extract_system() -> str:
    """长期记忆抽取器的 system prompt（含 facet 写入规则）。"""
    return render(EXTRACT_TEMPLATE, FACET_WRITE_RULES=FACET_WRITE_RULES)


def build_resolve_system() -> str:
    """画像写入器的 system prompt（含同一份 facet 写入规则）。"""
    return render(RESOLVE_TEMPLATE, FACET_WRITE_RULES=FACET_WRITE_RULES)


def build_merge_user(items: str) -> str:
    return render(MERGE_USER_TEMPLATE, ITEMS=items)


def build_summary_user(previous: str, source: str) -> str:
    if previous.strip():
        return render(SUMMARY_USER_TEMPLATE, PREVIOUS=previous.strip(), SOURCE=source)
    return render(SUMMARY_USER_FIRST_TEMPLATE, SOURCE=source)


def build_retry_feedback(problems: list[str]) -> str:
    """摘要校验不合格时，把落选原因拼成给模型的反馈。"""
    if not problems:
        return ""
    return render(RETRY_FEEDBACK_TEMPLATE, PROBLEMS="、".join(problems))


# 渲染一次供调用方直接用（纯常量，无动态输入，模块级预计算安全）
EXTRACT_SYSTEM = build_extract_system()
RESOLVE_SYSTEM = build_resolve_system()
