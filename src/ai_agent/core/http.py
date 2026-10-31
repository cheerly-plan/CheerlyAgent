"""零依赖 JSON HTTP 客户端。

web_search（三个搜索源）、weather（Open-Meteo）、gate（Jev）都要发
JSON HTTP 请求，统一走这里，避免每个模块各写一遍 urllib 样板。
错误语义与各调用方一致：网络/解析失败直接抛异常，由调用方决定降级方式。

超时：``urllib`` 在 Python 3.13 下，连接与读取共用 ``req.timeout``。
一个连不上的主机最多阻塞到该超时即抛错。
"""

import json
import urllib.error
import urllib.parse
import urllib.request


def _build_request(
    url: str, body: dict | None, headers: dict[str, str] | None
) -> urllib.request.Request:
    if body is None:
        return urllib.request.Request(
            url, headers={"Accept": "application/json", **(headers or {})}
        )
    return urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )


def _fetch(req: urllib.request.Request, read_timeout: float) -> dict:
    # req.timeout 同时约束 TCP 连接与读取：一个连不上的主机最多阻塞到此时长。
    req.timeout = read_timeout
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # 4xx/5xx 时部分 API 会在响应体里返回 JSON 错误，尽量解析出来透传；
        # 解析不出就按原异常抛出，由调用方降级成 ERROR 文案。
        try:
            return json.loads(e.read().decode())
        except Exception:
            raise


def request_json(
    url: str,
    *,
    body: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10,
) -> dict:
    """发一次 JSON 请求并解析响应。

    body 为 None 时发 GET（query 已拼在 url 里），否则发 POST（json 序列化）。
    连接与读取共用 timeout，超时由 urllib 抛出。
    """
    read_timeout = max(timeout, 1.0)
    req = _build_request(url, body, headers)
    return _fetch(req, read_timeout)


def quote(value: str) -> str:
    """URL query 参数转义。"""
    return urllib.parse.quote(value)
