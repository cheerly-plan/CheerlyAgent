from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str
    api_key: str
    model: str
    small_model: str = ""
    max_turns: int = 10
    log_level: str = "INFO"
    # 流式对话的 HTTP 超时（秒）：openai 客户端默认 600s 且自动重试 2 次，
    # LLM 挂死会占住并发槽与 SSE 线程最久 10 分钟（20 槽全满 → 全站 429）。
    # 流式场景下该超时作用于「连接 + 首 chunk」以及 chunk 之间的最大间隔，
    # 正常生成（chunk 间隔远小于 120s）不受影响。流式不做自动重试
    # （重试会从断点重复 yield 已发内容），彻底失败走断点续跑兜底。
    llm_stream_timeout_seconds: float = 120
    # 送模型的上下文预算（字符数，0 = 不限制）：超过时从最旧的完整轮次开始
    # 整轮丢弃（摘要对与当前回合保留），防止长对话 + 多步工具输出撑爆
    # 模型上下文窗口导致整轮报错。中文 1 字符 ≈ 0.6~1 token，40000 字符
    # 对 32k 窗口的模型是安全值；部署更大窗口的模型可调高。
    llm_context_char_budget: int = 40000
    host: str = "127.0.0.1"
    port: int = 8000
    # 服务端访问令牌：非空时 /v1/** 要求携带 X-API-Key 或 Bearer Token
    auth_token: str = ""
    # 聊天流并发上限，超出返回 429
    max_concurrent_chats: int = 20
    # 数据库连接池上限，需 ≥ 聊天并发 + 标题线程池 + 收尾线程池之和
    db_pool_max_size: int = 28
    # 允许的跨域来源，逗号分隔，如 "http://localhost:5173,https://chat.example.com"
    cors_origins: str = "http://localhost:5173"
    # 仅当部署在可信反向代理之后才开启：开启后限流取 X-Forwarded-For 最右侧 IP
    # （由自己的反代追加，客户端无法伪造）；默认关闭，直接取 socket 对端地址，
    # 防止伪造 XFF 绕过限流
    trust_forwarded_for: bool = False
    # trust_forwarded_for 开启时，视哪些地址为可信反代（uvicorn forwarded_allow_ips）
    forwarded_allow_ips: str = "127.0.0.1"
    # 开放注册开关：本机开发可开；对外暴露前建议关闭（403），改为管理员建号
    registration_enabled: bool = True
    # 断点续跑模式：True = prefill 续写（半截 assistant 留在上下文尾部，
    # 模型接着往下写，观感是"继续生成"）；False = 整段重生成（更兼容，
    # 适用于上下文以 assistant 结尾会被 API 忽略/报错的情况）
    resume_prefill: bool = True

    # 联网搜索工具：按配置的 key 自动选择搜索源（都未配置时工具返回
    # 引导文案）。tavily 免费额度 1000 次/月；bocha 为国内服务，直连可达；
    # serper 走 Google。多个 key 同时配置时按 tavily > bocha > serper 优先
    tavily_api_key: str = ""
    bocha_api_key: str = ""
    serper_api_key: str = ""
    web_search_max_results: int = 5
    web_search_timeout_seconds: float = 15

    # 管理员用户名白名单（逗号分隔）：只有这些账号能上传/删除知识库文件。
    # 空 = 没有任何管理员，所有写操作 403（只读浏览不受影响）
    admin_usernames: str = ""

    # 知识库（P1）：文件上传、切片向量化与检索
    kb_upload_dir: Path = _PROJECT_ROOT / "data" / "uploads"
    kb_max_file_mb: int = 100
    kb_part_size_mb: int = 4
    kb_chunk_size: int = 500
    kb_chunk_overlap: int = 50
    # 按章节切块：识别 PDF 的一级节标题（第X节/第X章），剔页眉页脚后
    # 在单节内聚合切片，一片不跨节。关掉即回到「按页切 + 页眉原样保留」
    kb_section_chunking: bool = True
    # 单页栅栏（默认关）：开启后同节内也强制按页断，一片只落一页。
    # 溯源最精确（page_start == page_end），但分页会把跨页的长论述、
    # 表格注释切断，实体/关系更容易被切散到两片里——审计、法律底稿
    # 这类「每句证据必须绑单页」的场景才打开。
    # 关闭时（默认）允许同节内跨页聚合，引用按「第134–135页」展示。
    kb_chunk_within_page: bool = False
    # PDF 表格识别：把表格抽成独立的 table 切片（整表成片、前端渲染成真
    # 表格），并把表格占用的文本行从正文里剔除，避免同一份内容存两份。
    # 关掉即回到「表格按文本行切成单列」的老行为。
    kb_pdf_table_extraction: bool = True
    # 单批条数：DashScope 兼容接口限制单次 input 最多 10 条，超限整批报
    # InvalidParameter（表现为文件处理 failed），默认对齐上限
    kb_embedding_batch: int = 10
    # 向量化批间并发：大文件是几十个串行 API 往返，批与批互相独立；
    # 并发 3 一般安全，网关报 429 就降回 1
    kb_embedding_concurrency: int = 3
    kb_pipeline_workers: int = 2
    # 图谱构建的独立线程数。必须和 kb_pipeline_workers 分池：一次图谱抽取
    # 是 40 分钟级的长任务，跟解析/向量化挤同一个池子时，后上传的文件只能
    # 排队等它跑完——表现为「切片生成也变慢了」
    kb_graph_workers: int = 1
    # 表格：单个工作表最多取多少行，防止超大表撑爆切片与向量库
    kb_sheet_max_rows: int = 2000

    # 图表识别（视觉模型）：ECharts 等图表的数值不在 PDF/Word 里，
    # 只能整页渲染后交给视觉模型近似还原。模型名为空即整体关闭——
    # 每张图一次调用是真实成本，必须由部署方显式开启。
    kb_vision_model: str = ""
    kb_vision_max_pages: int = 20
    kb_vision_dpi: int = 150
    # 视觉识别并发：页面渲染是本地快的，慢的是视觉调用，按并发数分批送
    kb_vision_concurrency: int = 3
    # 无位图时（ECharts 走 SVG 导出），矢量路径数超过该值视为图表页
    kb_vision_vector_paths: int = 150
    # 单次视觉调用的硬超时。SDK 默认 600s：本地视觉模型一旦卡死（模型要
    # 换入显存、图片 token 超上下文），一张图就能堵十分钟，20 页预算就是
    # 两小时的解析时间——而结果大概率还是 None
    kb_vision_timeout_seconds: float = 90.0
    # 连续失败（超时）多少次就熔断本次进程的视觉识别：卡住的调用不是
    # 「这一张图」的问题，接着试只是把同样的等待重复 20 遍
    kb_vision_max_failures: int = 2
    # 正文超过这么多字的页面不算图表页：招股书/财报里整页表格会画出几百条
    # 矢量边框，按路径数判断会把表格页误送视觉模型（白烧预算还更慢）
    kb_vision_max_text_chars: int = 600
    # 位图占页面面积的比例下限：页眉 logo、装饰线、水印都远小于该值
    kb_vision_min_image_ratio: float = 0.06

    # 知识图谱（P2）：切片入库后异步抽取实体关系（每批 = 一次 LLM 调用）。
    # 默认开启；关闭后新上传的文件不再生成图谱，已有图谱保留、仍可检索。
    # 抽取失败只影响图谱（graph_status=failed），文件本身仍是 ready、照常检索。
    kb_graph_enabled: bool = True
    # 空 = 跟随 MODEL。对话/嵌入走本地时，这里填线上模型名，并配
    # kb_graph_base_url，实体抽取才不会打到 Ollama 上 404。
    kb_graph_model: str = ""
    # 抽取专用网关。空 = 跟 BASE_URL。对话用 Ollama、抽实体用线上时填这个。
    kb_graph_base_url: str = ""
    # 抽取专用 key。空 = 跟 API_KEY。
    kb_graph_api_key: str = ""
    # 抽取温度：结构化输出必须取 0。不传时走服务端默认（OpenAI 系 1.0），
    # 实测同一批切片两次跑出的实体数能差 ±20，JSON 偶发坏输出（裸词没引号、
    # 括号没闭合），实体对齐的裁决结论也会翻。实体对齐裁决共用这个值
    kb_graph_temperature: float = 0.0
    kb_graph_batch_chunks: int = 8
    # 每积累多少批就把当前图谱覆盖落库一次。抽 700 多个切片要跑几十批、
    # 几十分钟，中途崩了不能前功尽弃：落过库的部分图能直接看，
    # 前端轮询也能看到实体数往上涨
    kb_graph_flush_batches: int = 15
    # 短于该字数的切片不进图谱：目录行、页码、页眉页脚抽不出实体，
    # 却要付一次完整的 LLM 调用
    kb_graph_min_chunk_chars: int = 40
    # 抽取批并发数：批与批互相独立，串行是图谱构建慢的主因。
    # 3 路 ≈ 快 3 倍；调大前先确认网关限流（偶发 429 只是丢一批，不致命）
    kb_graph_concurrency: int = 3
    # 单批抽取的超时：云端模型 45s 够，但本地 7B 模型产出 4k token 的 JSON
    # 常要 60~150s，给足余量（超时 = 丢一批，宁可慢也别丢）
    kb_graph_timeout_seconds: float = 180
    # 检索时取多少个种子实体、向前端返回的图最大节点数
    kb_graph_top_k: int = 6
    kb_graph_max_nodes: int = 200
    # 没有名称/关键词依据的候选，向量相似度必须过这条线。
    # 向量模型对无关词也会给 0.45~0.52 的基线，低于 0.55 基本只是话题沾边；
    # 名称命中或关键词命中到一半以上的候选不受此限制。
    kb_graph_min_score: float = 0.55

    # 抽取 prompt 的文档配置（通用底座 + 可插拔配置块，见 kb/doc_profiles.py）。
    # 路径不存在就只用内置配置；业务方加新文档类型改这个 JSON 即可，不用改代码。
    # 格式见 kb/doc_profiles.example.json（复制成该路径即生效）
    kb_doc_profiles_path: Path = _PROJECT_ROOT / "data" / "kb_doc_profiles.json"
    # 没指定文档类型时的兜底：通用配置（不限白名单、不锁谓词）
    kb_default_doc_type: str = "general"

    # 实体对齐（图谱质量 P2）：把「亚厦股份 / 浙江亚厦装饰股份有限公司」
    # 这类同一实体的不同写法合并，孤立点大多由此产生。
    # 向量召回下限：低于它的对子连候选都不进（实体向量基线约 0.45~0.55）
    kg_align_min_sim: float = 0.86
    # 每个实体最多取几个近邻当候选；对子太多会把裁决调用拖成几十次
    kg_align_neighbors: int = 4
    # 一次裁决调用塞几对：受上下文限制，本地 7B 模型 10 对以上开始漏项
    kg_align_batch: int = 8
    # 单次对齐最多裁决多少对（成本闸门，超出的按相似度截断）
    kg_align_max_pairs: int = 240
    # 单次裁决调用的超时（秒）。裁决提示词很短、正常几秒返回，
    # 用抽取那套 180 秒只会在模型卡住时把整轮对齐拖死
    kg_align_timeout_seconds: float = 45.0
    # 一个簇最多并几个成员：超过这个数通常是"包含链把一片法人串起来"的
    # 误判，整簇跳过不合并（宁可图稀疏，也不能把二十个法人压成一个点）
    kg_align_max_cluster: int = 4

    # 检索质量：向量召回的候选数、距离门槛、重排与邻接扩展。
    # 候选先多召回（16 条），再重排 + 去重后取 top_k —— 直接取最近邻的
    # 3 条经常全是同一段话的变体，真正回答问题的片段反而排在第四位。
    kb_search_candidates: int = 16
    # 余弦距离上限（0 完全相同、2 完全相反）。中文 embedding 对无关内容
    # 也有 0.55~0.65 的基线距离，超过 0.72 基本只是话题沾边，宁缺毋滥；
    # 设 2.0 等于关闭这条门槛。
    kb_search_max_distance: float = 0.72
    # 知识库预检索（always-retrieve）：kb 开关打开时，每轮回答前服务端先替
    # 模型查一次库并把命中注入上下文，不依赖模型自觉调用 kb_search——
    # 本地小模型对「要不要查」的主观判断不稳定，用户把原文粘进对话框
    # 都可能不触发。检索是毫秒级操作，闲聊时没有语义接近的片段自然不注入
    kb_prefetch_enabled: bool = True
    kb_prefetch_top_k: int = 3
    # 预检索比工具检索更严的距离门槛：注入内容是「被动给模型看的」，
    # 弱相关片段会把闲聊回答带偏。0.45 在余弦距离下意味着语义相当接近
    kb_prefetch_max_distance: float = 0.45
    # LLM 重排：多一次模型调用（约 1~2 秒），换来排序质量。关掉即退回向量顺序
    kb_rerank_enabled: bool = True
    # 空 = 跟随 MODEL（主模型）。重排是短文本判别，不需要大模型
    kb_rerank_model: str = ""
    # 重排只输出一串编号，要的是稳定排序而不是多样性：0.0 保证同一组候选
    # 每次排出同一个顺序，否则评测结果会随采样抖动，改动好坏无法归因
    kb_rerank_temperature: float = 0.0
    kb_rerank_timeout_seconds: float = 30.0
    # 对话改写：多轮对话里，当前这句省略了主体/语境（上轮在问「中国能源
    # 建设」、这轮问「股东持股情况」），预检索前用小模型结合上一轮对话把
    # 这句改写/扩写成自包含的检索词，避免「股东持股情况」这种全库都有的表
    # 命中别家公司的文件。补全与扩写都允许；第一问无上文时跳过；失败/超时退回原话。
    kb_rewrite_enabled: bool = True
    # 空 = 用 small_model（或回落主模型）。改写是短任务，不需要大模型
    kb_rewrite_model: str = ""
    # 改写是「同一句话补主体」，不该每次补出不同措辞：不同措辞会引出不同的
    # 召回集合，让「重排坏了」和「改写抖了」分不清
    kb_rewrite_temperature: float = 0.0
    kb_rewrite_timeout_seconds: float = 6.0
    # 改写喂给模型的最近几轮「完整对话（user 问句 + assistant 最终回答）」：
    # 主体（公司名）可能出现在更早那轮、中间隔了无主体的追问，取 2 轮更稳。
    kb_rewrite_history_turns: int = 2
    # 命中切片前后各带几段一起给模型：跨段落的答案常常被切片边界切断
    kb_neighbor_expand: int = 1
    # 混合检索：关键词直查 + 向量召回做 RRF 融合。向量对制度编号、
    # 专有名词、精确表述不敏感，关键词路补这个盲区；关掉即纯向量
    kb_hybrid_enabled: bool = True
    # 从查询里提取的关键词上限（jieba TF-IDF）
    kb_hybrid_keywords: int = 6
    # 关键词路的召回条数：与向量路同量级，融合后一起去重/重排
    kb_hybrid_candidates: int = 12
    # 标签预过滤：查询关键词命中文件标签时先收窄到这批文件再检索，
    # 少了别的文件的干扰项（如 FAQ），重排输入更干净。
    # 代价是可能判窄，所以调用方必须带弱召回兜底（见 tools/kb_search）。
    # 提取关键词用的是已有逻辑，没有新增 LLM 调用
    kb_tag_scope_enabled: bool = True

    database_url: str
    embedding_model: str
    embedding_dim: int
    memory_top_k: int
    memory_score_threshold: float
    memory_merge_threshold: float = 0.7
    memory_merge_timeout_seconds: float = 8
    memory_model: str = ""
    # 情景记忆存活时间（秒）。episode 是「当前阶段近况」，不是画像：过期后
    # 检索/合并都看不见，后台再物理删除。招股书、天气、游戏点子混在一个
    # 向量空间里，旧近况不淘汰就会把新问题带到别的公司。0 = 永不过期。
    memory_episode_ttl_seconds: int = 7 * 24 * 3600

    # Jev 门控（System One 决策模型）：长期记忆抽取的前置判断——Noul 概率
    # 低于 memory_gate_threshold 时跳过本轮抽取，省掉一次小模型调用。
    # 未配 key 时门控关闭，行为退化为每轮直接抽取；门控任何失败都
    # fail-open 放行，不影响记忆链路。阈值偏高：错记一轮问答比漏记更糟。
    jev_api_key: str = ""
    jev_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-latest"
    memory_gate_threshold: float = 0.65
    jev_timeout_seconds: float = 4

    def resolve_llm(self, business_model: str = "") -> str:
        return business_model or self.small_model or self.model

    def llm_for_memory(self) -> str:
        return self.resolve_llm(self.memory_model)

    def is_admin_username(self, username: str) -> bool:
        """管理员判定：用户名命中 ADMIN_USERNAMES 白名单（大小写敏感）。"""
        return username in {
            name.strip() for name in self.admin_usernames.split(",") if name.strip()
        }

    short_term_memory_max_turns: int
    short_term_memory_max_tokens: int
    short_term_memory_max_sessions: int = 1024
    short_term_memory_ttl_seconds: int = 3600
    short_term_memory_cleanup_interval_seconds: int = 300

    title_model: str = ""
    short_term_summary_model: str = ""
    short_term_summary_timeout_seconds: float = 12
    title_timeout_seconds: float = 20
    session_cleanup_interval_seconds: int = 300
    session_empty_max_age_seconds: int = 3600


settings = Settings()
