"""工具描述文案。

**工具 description 就是提示词**——它随 function schema 一起进模型，直接决定
模型什么时候调、参数怎么填。以前它们散在 6 个工具的源码里，改一处措辞要
翻 6 个文件，且没有任何测试守着。这里集中后由 test_prompts 统一断言。

文案一字未改（含【触发条件】/【禁止条件】这些实测有效的结构化约束）。
"""

from __future__ import annotations

from ai_agent.prompts.builder import render

# ——— 天气 ———

WEATHER_DESC = (
    "查询指定城市的实时天气和每日预报（温度、体感、湿度、风速、降水概率）。"
    "用户问天气、穿什么、要不要带伞、出行安排时调用。"
)
WEATHER_CITY_DESC = "城市名称，可用中文。重名城市可带国家或省份，如「剑桥 英国」"
WEATHER_DAYS_DESC = (
    "预报天数：1=只看今天，2=到明天，最多 7 天。用户问周末/未来安排时适当加大"
)

# ——— 联网搜索 ———

WEB_SEARCH_DESC = (
    "联网搜索最新信息。当问题涉及实时数据、新闻、价格、版本更新、"
    "或你的知识可能过时时调用。query 用具体的搜索词，不要整句照抄"
    '用户提问；查刚发生的事情时传 freshness="day"。'
)
WEB_SEARCH_QUERY_DESC = "搜索关键词，尽量具体，可用引号精确匹配短语"
WEB_SEARCH_MAX_RESULTS_TEMPLATE = "返回结果条数，默认 {{DEFAULT}}"
WEB_SEARCH_FRESHNESS_DESC = (
    "时间范围过滤：day（24小时内）/ week / month / year。"
    "查新闻、刚发生的事件、最新版本时必传 day 或 week；"
    "一般性查询留空。"
)
# 未配 key 时回给模型（也是回给运维看）的引导文案
WEB_SEARCH_NO_KEY_HINT = (
    "ERROR: 联网搜索未配置。请在服务端 .env 里配置任一搜索 API key 后重启：\n"
    "TAVILY_API_KEY（免费 1000 次/月，https://tavily.com）\n"
    "BOCHA_API_KEY（国内直连，https://bocha-ai.com）\n"
    "SERPER_API_KEY（Google 结果，免费 2500 次，https://serper.dev）"
)

# ——— 记忆 ———

MEMORY_SEARCH_DESC = (
    "检索当前用户的跨会话情景记忆，不是当前对话。"
    "【触发条件】用户询问过去的事情、需要历史细节或过往结论，且当前上下文不够时调用。"
    "【禁止条件】系统已注入的用户画像、当前会话里已有的内容，不要再搜。"
)
MEMORY_SEARCH_QUERY_DESC = "要检索的关键词或问题"

MEMORY_WRITE_DESC = (
    "仅在用户明确要求记住，或清楚陈述姓名、职业、身份、长期偏好时写入。"
    "当前任务、正在学习或讨论的技术方案不要在此写入，系统会在本轮结束后自动抽取。"
    "一条只写一件事：身份/偏好用 profile；只有用户明确要求记住近况时才用 episode。"
    "例如「我是画家，请记住」只写 profile='我是画家'。"
    "【禁止条件】用户只是提问、要建议、讨论方案时，绝对不要调用。"
)
MEMORY_WRITE_CONTENT_DESC = (
    "一条只写一件事。身份/偏好用完整陈述，不要把当前任务写进来。"
)
MEMORY_WRITE_KIND_DESC = (
    "profile: 长期成立的姓名、职业、身份、语言、偏好、约束。\n"
    "episode: 仅当用户明确要求记住某件近况时使用。"
)

# ——— 知识库检索 ———

KB_SEARCH_DESC = (
    "检索知识库里的共享文档。涉及公司、产品、财报、数字、日期、"
    "条款等具体事实时必须调用，不要用自己的知识直接回答。"
    "不确定库里有哪些文件时先调 kb_list_files 查看清单，再用 files 圈定"
    "（填文件名，可简称），不要填文件 id；不填则按问题自动匹配。"
)
KB_SEARCH_QUERY_DESC = "检索问题或关键词，用自然语言描述要找的内容"
# kb_search 与 kg_search 的 files 参数说明完全一致，共用一份避免漂移
KB_FILES_DESC = (
    "限定检索的文件名（可简称、可多份），对照 kb_list_files 返回的文件名填写；"
    "不填则按问题自动匹配或查全库"
)

# ——— 知识库文件清单 ———

KB_LIST_DESC = (
    "列出知识库里已就绪（可检索）的文件，含文件名与内容描述。"
    "不确定库里有哪些文件、要查哪几份时调用，再据此用 kb_search 的 files 参数圈定范围。"
)
KB_SEARCH_TOPK_DESC = "返回的片段数"

KB_EMPTY_HINT = "知识库为空或还没有处理完成的文件，请提示用户联系管理员上传资料。"
KB_NO_HIT_SCOPED_TEMPLATE = (
    "在用户限定的 {{COUNT}} 份文档里没有找到相关内容。"
    "可以提示用户清除知识库范围限制后再查，或换个说法。"
)
KB_NO_HIT_HINT = "没有在知识库中找到相关内容，可以换个说法再试。"
KB_ERROR_HINT = "ERROR: 知识库暂时不可用，请稍后重试"

# ——— 知识图谱检索 ———

KG_SEARCH_DESC = (
    "按实体与关系检索知识库图谱。当问题涉及「某公司/技术/指标和谁有关、"
    "上下游、投资关系、产业链位置」这类关系型问题时优先使用；"
    "只查原文表述时用 kb_search。files 填 kb_list_files 返回的文件名（可简称），"
    "不要填文件 id。"
)
KG_SEARCH_QUERY_DESC = "要检索的实体、主题或问题，用自然语言描述"
KG_SEARCH_DEPTH_DESC = "从命中实体向外扩展的跳数"
KG_SEARCH_TOPK_DESC = "参与扩展的种子实体数"
KG_NO_ENTITY_HINT = "图谱里没有找到相关实体，可以换个说法，或用 kb_search 直接检索原文。"


def build_web_search_max_results_desc(default: int) -> str:
    return render(WEB_SEARCH_MAX_RESULTS_TEMPLATE, DEFAULT=str(default))


def build_kb_no_hit_scoped(count: int) -> str:
    return render(KB_NO_HIT_SCOPED_TEMPLATE, COUNT=str(count))
