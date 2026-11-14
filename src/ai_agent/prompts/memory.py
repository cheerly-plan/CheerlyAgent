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
- locale：只写当前居住地。搬到新城市时写成新城市，不要写成「曾住X，现住Y」——历史由系统按时间区间保存，不需要你拼接
- occupation：只写当前实际身份。必须区分「已发生」和「只是计划」：
  * 已发生（已经换了 / 现在是 / 目前在做）→ 更新现状
  * 只是意向（打算 / 准备 / 想 / 考虑 / 计划）→ 尚未成立，不写进 occupation；
    意向可以单独记一条 other，内容写明是计划
- preference / constraint：可同时成立多条独立偏好/约束，每条单独输出一个 add，
  不要合并成一段、也不要覆盖已有条目；只有用户明确否定某条时才 delete 那一条，
  delete 时 content 写被否定条目的原文
- directive：用户对「以后怎么回答我」的长期要求。这是交互规则，不是内容偏好——
  凡是带「以后 / 每次 / 一律 / 不要 / 必须」这类长期措辞的回答方式要求，写 directive
- other：其他稳定事实；姓名、职业、地区、语言不要写进 other
已发生的变化与「只是计划」的区分适用于所有 facet：只有已成立的写进对应 facet，
没发生的放 other 并标明是计划。
已有条目已覆盖新事实时 skip，不要再写一遍。
""".strip()

# ——— 长期记忆抽取 ———

EXTRACT_TEMPLATE = """
你是用户长期记忆抽取器。只记「关于用户本人、且下一场对话还用得上」的事实。
只输出 JSON，不要解释。格式：
{"memories":[{"kind":"episode","content":"..."},{"kind":"profile","op":"add"|"update"|"delete","facet":"directive"|"name"|"occupation"|"locale"|"language"|"preference"|"constraint"|"other","content":"..."}]}
没有可记内容时必须输出 {"memories":[]}。拿不准就空数组。

判断标准只有一条：
「这是关于用户本人的长期事实，还是本次任务的对象？」
- 用户本人的长期特征（身份、偏好、要求、在做的长期事情）→ 记
- 本次任务的对象（被查询/分析/生成/处理的那些东西）→ 不记
这条标准适用于任何业务领域，不依赖具体行业词汇。

kind:
- profile: 用户用第一人称说清的姓名、职业、居住地、语言、长期偏好、约束、长期指令
- directive: 用户要求「以后怎么回答我」的交互规则（回答先给结论、不要用表格、都要给出处）
- episode: 用户自己正在做、会跨会话延续的事情（长期项目、已确认的方案、明确要求记住的进展）

严禁写入（按原则，不限领域）：
- 本轮对话纪要：不复述用户问了什么、助手答了什么、讨论过什么
- 本轮动作与任务状态：正在查询/正在确认/已提交/后台任务/进度/预计耗时
- 通过工具或检索拿到的外部事实：无论来源是文档、网页、数据库还是接口，
  查来的内容属于任务对象，不是用户本人的事实
- 把任务对象写成用户身份：查了某家公司、某位病患、某个订单，不等于用户就是它的人
- 一次性信息：天气、行情、单次查询结果、某一页内容、某个编号
- 短期安排（下周出差、明天开会）——这不是居住地或身份
- 助手的建议、工具调用过程
- 已经记过的相同事实、只是换个说法、纯闲聊
- 系统默认值（用户说中文不代表要记「语言=中文」）

示例（仅帮助理解，不限于这些领域）：
- 任务对象：公司、客户、病患、订单、合同、文件、代码仓库、词条……
- 用户本人：姓名、职业、居住地、偏好、回答要求、在做的长期项目……

facet 额外约束：
- locale 只能来自用户明确陈述的长期居住地，不能由出差、行程、临时项目推断
- occupation 只能来自用户第一人称陈述的当前身份；「打算/准备」不算

episode 的 content 约 40～120 字，写用户本人正在做的事，不要写成对话纪要。
profile 必须带 op 和 facet。name / occupation / locale / language 同一侧面只输出一条：
- add: 新的侧面
- update: 同一侧面事实变化或更完整，content 写成该侧面的完整当前描述
- delete: 用户明确否定旧事实
preference / constraint 是累积型：每条独立事实单独输出一个 add，可多条并存，
不要 update 合并已有条目；否定某一条时 delete 并在 content 写该条原文。
{{FACET_WRITE_RULES}}
若已有 other 把姓名/职业混在一起，要拆开写入正确 facet，并 delete other。

规则：
1. 只记用户明确陈述的自身事实，或双方已确认、属于用户本人的选择
2. 身份和近况拆开
3. 最多 2 条
"""

# Jev 门控：本轮是否值得抽取长期记忆
GATE_INSTRUCTIONS = (
    "这段对话是否包含值得写入跨会话长期记忆的信息？"
    "值得记忆：用户本人的长期特征——第一人称说清的姓名、职业、居住地、语言、"
    "长期偏好或约束、对回答方式的长期要求、用户自己正在做且会跨会话延续的事情。"
    "不值得记忆：本次任务的对象（查了什么、分析了什么、生成了什么，不论哪个领域）、"
    "本轮正在进行的查询或任务状态（进度、编号、预计耗时）、"
    "通过工具或检索得到的外部事实、短期安排、对话纪要、助手建议、系统默认值。"
)

# ——— 画像写入 ———

RESOLVE_TEMPLATE = """
你是用户画像写入器。根据已有画像和新事实，决定如何更新。
只输出 JSON，不要解释。格式：
{"actions":[{"op":"add"|"update"|"delete"|"skip","facet":"directive"|"name"|"occupation"|"locale"|"language"|"preference"|"constraint"|"other","content":"..."}]}

facet:
- directive: 用户对回答方式的长期要求（先给结论、不要用表格、都要附出处等）
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

# 本地小模型（qwen3 这类思考型）走原生接口时，自由文本会被思考过程污染，
# 而 JSON 语法约束能让它 1 秒内直接产出干净内容。摘要本身是一段文字，
# 包一层 {"summary": "..."} 就能吃到这条快路径，解析失败仍回退纯文本版。
SUMMARY_SYSTEM_JSON = (
    SUMMARY_SYSTEM.strip()
    + '\n\n输出格式：只输出 JSON，形如 {"summary": "摘要正文"}，不要任何其他字段或解释。'
)

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
