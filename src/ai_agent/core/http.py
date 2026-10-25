"""零依赖 JSON HTTP 客户端。

web_search（三个搜索源）、weather（Open-Meteo）、gate（Jev）都要发
JSON HTTP 请求，统一走这里，避免每个模块各写一遍 urllib 样板。
错误语义与各调用方一致：网络/解析失败直接抛异常，由调用方决定降级方式。
"""

import json
import urllib.parse
import urllib.request


def request_json(
    url: str,
    *,
    body: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10,
) -> dict:
    """发一次 JSON 请求并解析响应。

    body 为 None 时发 GET（query 已拼在 url 里），否则发 POST（json 序列化）。
    """
    if body is None:
        req = urllib.request.Request(
            url, headers={"Accept": "application/json", **(headers or {})}
        )
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def quote(value: str) -> str:
    """URL query 参数转义。"""
    return urllib.parse.quote(value)
