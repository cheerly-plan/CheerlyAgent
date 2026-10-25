"""Jev 门控客户端（System One 决策模型）。

Jev 不生成文本，只对预定义问题返回校准概率，适合 Agent 里高频、
低风险的结构化判断。

设计约束：
- 零第三方依赖：一次 REST 调用，标准库 urllib 足够（与 web_search 同策略）；
- fail-open：任何失败（未配 key / 网络错误 / 响应异常）返回 None，
  调用方视为「门控不可用」直接放行——记忆链路的既有哲学是任何
  故障静默降级、不影响对话主流程；
- 未配 key 时同样返回 None，整段门控等于不存在，零风险灰度。

使用位置：
- 长期记忆门控：单个 Noul 判断本轮是否值得触发抽取（long_term.py）；
- 摘要质量校验：一次请求并行两个 Noul，判编造/遗漏（summarize.py）；
- 摘要逐轮预过滤：一次请求并行 N 个 Noul，剔除低价值轮次（summarize.py）；
- 画像语义去重：写入前并行比对同 facet 已有条目（profile.py）。

API 契约（TypeSafe v1）：
POST {base_url}/v1/systemone
  {"model": "...", "state": "...",
   "questions": {"<id>": {"type": "noul", "instructions": "..."}}}
→ {"answers": {"<id>": {"type": "noul", "noul": 0.87}}, ...}
独立问题在同一请求内并行评估；只按输入 token 计费，输出不计费。
"""

import time

from ai_agent.core import http
from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger

logger = get_logger(__name__)


def jev_noul(instructions: str, state: str) -> float | None:
    """问一个 Noul（yes/no）问题，返回「是」的校准概率。

    返回 None 表示门控不可用（未配 key、网络失败、响应异常），
    调用方应放行而不是拦截。
    """
    answers = jev_noul_batch({"gate": instructions}, state)
    if answers is None:
        return None
    return answers["gate"]


def jev_noul_batch(
    questions: dict[str, str], state: str
) -> dict[str, float] | None:
    """一次请求并行问多个 Noul 问题，返回 {问题ID: 概率}。

    questions 是 {问题ID: 判定指令}；答案按同样的 key 取回。
    任一答案缺失/非法都整体返回 None——调用方按不可用处理，
    避免拿到半套概率做出错误决策。
    """
    key = settings.jev_api_key
    if not key:
        return None
    req_body = {
        "model": settings.jev_model,
        "state": state,
        "questions": {
            qid: {"type": "noul", "instructions": instructions}
            for qid, instructions in questions.items()
        },
    }
    started = time.perf_counter()
    try:
        data = http.request_json(
            f"{settings.jev_base_url.rstrip('/')}/v1/systemone",
            body=req_body,
            headers={"Authorization": f"Bearer {key}"},
            timeout=settings.jev_timeout_seconds,
        )
    except Exception as e:
        logger.warning(
            "Jev 调用失败 elapsed=%dms: %s",
            (time.perf_counter() - started) * 1000,
            e,
        )
        return None
    elapsed_ms = (time.perf_counter() - started) * 1000
    answers = data.get("answers") or {}
    result: dict[str, float] = {}
    for qid in questions:
        noul = (answers.get(qid) or {}).get("noul")
        # Noul 答案没有独立 confidence 字段，概率本身即可信度，直接对阈值；
        # bool 是 int 的子类，必须排除
        if not isinstance(noul, (int, float)) or isinstance(noul, bool):
            logger.warning(
                "Jev 响应缺少 %s.noul 字段 elapsed=%dms model=%s",
                qid,
                elapsed_ms,
                data.get("model"),
            )
            return None
        result[qid] = float(noul)
    logger.info(
        "Jev 调用完成 model=%s elapsed=%dms %s",
        data.get("model"),
        elapsed_ms,
        {qid: round(v, 2) for qid, v in result.items()},
    )
    return result
