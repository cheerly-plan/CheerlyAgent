"""图表识别：把 PDF / DOCX 里的图片交给视觉模型，转成可检索的文字描述。

为什么需要这一步：ECharts 这类图表是前端渲染的，导出成 PDF/Word 后只剩
矢量路径或位图，**数值本身不在文件里**，文字层只能靠图例、轴标签碰运气。
所以只能用视觉模型做近似还原，产出的切片统一标注「图像识别」，让下游
知道这是读图的结论而非原文。

默认关闭（kb_vision_model 为空）：每张图一次视觉调用是真实成本，
由部署方显式开启并设置页数上限。
"""

import base64

from openai import OpenAI

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger

logger = get_logger(__name__)

_client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)

# 只接受视觉模型确定支持的位图格式；EMF/WMF（Word 常见矢量占位）直接跳过
_IMAGE_MIME = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/webp": "webp",
    "image/gif": "gif",
}

_PROMPT = """这是一份文档里的图表或图片。请用中文输出便于检索的描述，按以下顺序：
1. 图表类型（折线/柱状/饼图/流程图/截图表/其他）和标题；
2. 坐标轴含义与单位、图例里的系列名称；
3. 能看清的关键数值，按「系列: 数值」逐条列出；
4. 一句话总结趋势或结论。

要求：只写图上确实存在的信息；看不清的数值写「无法精确读取」，不要推测、
不要补全缺失数据。总长度不超过 300 字。"""


def enabled() -> bool:
    return bool(settings.kb_vision_model.strip())


def supported_mime(mime: str) -> bool:
    return mime.lower() in _IMAGE_MIME


def describe_image(data: bytes, mime: str = "image/png") -> str | None:
    """返回图片描述；关闭、格式不支持或调用失败时返回 None（不阻断解析）。"""
    if not enabled() or not data or not supported_mime(mime):
        return None
    try:
        b64 = base64.b64encode(data).decode()
        resp = _client.chat.completions.create(
            model=settings.kb_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=800,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        # 单张图失败不该让整个文件 failed：记警告后跳过
        logger.warning("kb 图表识别失败，跳过该图", exc_info=True)
        return None
    return text or None
