"""工具描述文案。

**工具 description 就是提示词**——它随 function schema 一起进模型，直接决定
模型什么时候调、参数怎么填。以前它们散在 6 个工具的源码里，改一处措辞要
翻 6 个文件，且没有任何测试守着。这里集中后由 test_prompts 统一断言。

静态文案迁移时一字未改（含【触发条件】/【禁止条件】这些实测有效的结构化
约束）。例外是研报类型清单：report_type 的可选值不手抄，模板留
``{{SKILL_CATALOG}}`` 槽位，由**调用方**（tools / agents 组装层）注入
`report.skills.skill_catalog()` 渲染——注册新 skill 即自动出现在工具描述
和系统提示词里。本模块刻意不 import 业务域：prompts 是文案层，方向只能
被上层依赖，不能反向依赖 report / kb / memory。
"""

from __future__ import annotations

from ai_agent.prompts.builder import assert_rendered, render

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
    "query 保留问句里的公司名和要查的事（如「中国黄金 营收增长」），"
    "不要收成单个科目，也不要自己限定章节或只要表格。"
    "files 一般不填：问句里点到的公司由系统收进这些文件。"
    "只有用户明确给出文件名时才填 files（可多份）。"
    "不要编造文件名，也不要填文件 id。库里有哪些文件先调 kb_list_files。"
)
KB_SEARCH_QUERY_DESC = (
    "检索词。保留公司名和用户要问的事，用自然语言，例如「中国黄金 营收增长」。"
    "不要改写成单个科目名。用户点名了某一章，或只要表格、不要正文时，"
    "才另填 sections、kinds。"
)
# kb_search 与 kg_search 的 files 参数说明完全一致，共用一份避免漂移
KB_FILES_DESC = (
    "选填。要收窄到的文件名或公司名（可简称、可多份）。"
    "不填时系统按问句里点到的公司，以及文档摘要，自动选定文件，"
    "再在这些文件里检索 query。不要编造文件名，也不要填文件 id。"
)

# ——— 知识库文件清单 ———

KB_LIST_DESC = (
    "列出知识库里已就绪（可检索）的文件，含文件名与内容描述。"
    "文件较多时必须带 q，按文件名、主体名或内容描述筛选，一次最多返回 30 条。"
    "用户问库里有哪些公司、哪一期、哪份文件时调用。"
    "需要收窄检索范围时，把这里确认的文件名或主体名传给 kb_search / kg_search 的 files；"
    "范围不确定就不要填 files。"
)

KB_SEARCH_TOPK_DESC = "返回的片段数"

KB_EMPTY_HINT = "知识库为空或还没有处理完成的文件，请提示用户联系管理员上传资料。"
KB_NO_HIT_SCOPED_TEMPLATE = (
    "在用户限定的 {{COUNT}} 份文档里没有找到相关内容。"
    "可以提示用户清除知识库范围限制后再查，或换个说法。"
)
KB_NO_HIT_HINT = "没有在知识库中找到相关内容，可以换个说法再试。"
KB_ERROR_HINT = "ERROR: 知识库暂时不可用，请稍后重试"
KB_FILES_NOT_FOUND_HINT = (
    "未找到匹配的文件或主体（你填的文件名或主体名在知识库里不存在）。"
    "先调 kb_list_files，用 q 传关键字检索文件，再填 files；"
    "或留空 files，让系统按问句选定范围。"
)

# ——— 知识图谱检索 ———

KG_SEARCH_DESC = (
    "按实体与关系检索知识库图谱，适用于「谁和谁有什么关系」这类问题。"
    "图谱只存实体间的关系，不存报表科目金额、日期、条款等原文事实——后者用 kb_search。"
    "当前知识库图谱具体覆盖哪些实体类型与关系，见系统提示词里的【知识库能力】；"
    "不在其中的关系请改走 kb_search 读原文。持股数量等若写在关系说明里可以直接用，"
    "科目余额以 kb_search 的原文为准。"
    "files 一般不填：问句点到的公司或文档由系统收进检索范围，"
    "再在这些文件里查关系。只有用户明确给出文件名或公司名时才填；"
    "不要编造，也不要填文件 id。库里有哪些文件先调 kb_list_files。"
)
KG_SEARCH_QUERY_DESC = (
    "要检索的实体或关系，用简短关键词，不要整句自然语言。\n"
    "· 问某人的任职/持股：query 直接写那个人名（如「金仙」），"
    "不要把「岗位 职务 董事 监事」这类泛词堆进去——它们会冲淡姓名信号。\n"
    "· 问谁和谁的关系：写「实体A 关系 实体B」或「实体 + 关系」"
    "（如「中化装备 持股」「长安汽车 营收来自」）。\n"
    "· 实体名用报告里的原称（可用简称）；不确定实体时只写最关键的那个。\n"
    "· 不要传报表科目名（货币资金/营业收入/净利润等）——它们不是图谱实体，"
    "传了也匹配不到，请改用 kb_search。"
)
KG_SEARCH_DEPTH_DESC = "从命中实体向外扩展的跳数"
KG_SEARCH_TOPK_DESC = "参与扩展的种子实体数"
KG_NO_ENTITY_HINT = "图谱里没有找到相关实体，可以换个说法，或用 kb_search 直接检索原文。"


# ——— 研报生成（聊天侧唯一入口，落盘路径由服务端固定） ———

# 报告类型清单不在文案里手抄：模板留 {{SKILL_CATALOG}} 槽位，由 tools 组装层
# 注入 report.skills.skill_catalog() 渲染。注册表加新 skill 后工具描述自动
# 带上，模型才知道它的存在（否则流水线路由了、模型也点不了名）。
REPORT_GENERATE_DESC_TEMPLATE = (
    "生成财报研报：默认同时交付事实层底稿与独立深度解析两套产物"
    "（底稿 PDF / Markdown / 溯源，以及深度 PDF .deep.pdf / .deep.md / .deep.json）。"
    "【触发条件】用户要求「生成研报」「财报深度分析」「出 PDF 报告」时必须调用，"
    "把需求原话原样填进 request，不要拆分、改写或自己补指标。"
    "报告类型默认按公司行业自动路由（银行 / 通用公司）；用户明确点名其他类型时"
    "才填 report_type（可选：{{SKILL_CATALOG}}），不确定就留空。"
    "报告由服务端确定性流水线生成：正文数字只从知识库表格抽取并过校验门禁，"
    "抽不到的写进缺口，不补估计值；深度层为带警示标注的推断结论，不并入底稿正文。"
    "【禁止条件】不要自己撰写研报正文、不要用检索片段口算同比增减，"
    "也不要说「我无法生成 PDF」——本工具默认就出 PDF。"
    "除非用户明确只要 Markdown / 不要 PDF，否则不要把 write_pdf 设为 false。"
    "不要填任何文件路径，落盘位置由服务端决定。"
    "工具返回后把底稿与深度两套下载链接都**逐字原样**发给用户："
    "必须保持 /v1/report/files/ 相对路径，禁止改成 example.com、http(s) 或其它外链；"
    "不要复述或换算报告里的数字。"
)


# generate_report 工具 report_type 字段的描述模板（tools/report_generate.py
# 引用）。与工具描述共用同一个 catalog 注入，两处不会各说各话。
REPORT_TYPE_DESC_TEMPLATE = (
    "报告类型，留空按公司行业自动路由。"
    "可选：{{SKILL_CATALOG}}。用户没明确点名就不要填"
)


def render_report_generate_desc(catalog: str) -> str:
    """工具描述：catalog 由调用方传 report.skills.skill_catalog()。"""
    return assert_rendered(render(REPORT_GENERATE_DESC_TEMPLATE, SKILL_CATALOG=catalog))


def render_report_type_desc(catalog: str) -> str:
    """report_type 字段描述：catalog 由调用方传 report.skills.skill_catalog()。"""
    return assert_rendered(render(REPORT_TYPE_DESC_TEMPLATE, SKILL_CATALOG=catalog))

# ——— 研报积木：单独抽数 / 算增减 / 校验正文（都不写盘、无路径参数） ———

REPORT_EXTRACT_DESC = (
    "从知识库财报里抽取指定公司、指定报告期的财务指标，返回原值、单位、口径和出处。"
    "【触发条件】用户问某个财务指标的具体数字（如「中行 2026Q3 净息差是多少」）、"
    "或需要为计算准备原始数据时调用；indicators 留空则返回该公司行业的默认指标集。"
    "【禁止条件】不要用它代替 generate_report 生成整份报告；"
    "返回的数字必须原样转述或传给 report_compare，禁止换算、四舍五入或口算。"
)

REPORT_COMPARE_DESC = (
    "计算已抽取指标的多期同比、环比、增减额与幅度，口径不可比时如实说明不算。"
    "【触发条件】拿到 report_extract_indicators 返回的多期原值后需要算增减时调用，"
    "把原值原样传入 points。"
    "【禁止条件】禁止自己心算后再传入；禁止把不同口径（单季/累计/期末余额）"
    "或不同覆盖期数的值混在一条序列里。"
)

REPORT_VALIDATE_DESC = (
    "校验一段研报正文的数字与引用是否与抽取事实一致，返回校验结论与审核队列。"
    "【触发条件】在向用户展示自写的财务分析文字之前必须调用，"
    "正文里每个数字都要能在 facts 里找到出处。"
    "【禁止条件】校验未通过时不得照发正文；不得自行宣布校验通过。"
)


def build_web_search_max_results_desc(default: int) -> str:
    return render(WEB_SEARCH_MAX_RESULTS_TEMPLATE, DEFAULT=str(default))


def build_kb_no_hit_scoped(count: int) -> str:
    return render(KB_NO_HIT_SCOPED_TEMPLATE, COUNT=str(count))
