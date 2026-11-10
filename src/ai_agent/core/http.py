"""零依赖 HTTP 客户端。

web_search、weather、gate、巨潮公告查询都走这里，避免各模块各写一遍
urllib 样板。错误语义与各调用方一致：网络/解析失败直接抛异常，由调用方
决定降级方式。

超时：``urllib`` 在 Python 3.13 下，连接与读取共用 ``req.timeout``。
一个连不上的主机最多阻塞到该超时即抛错。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


def _json_request(
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


def _read_json(req: urllib.request.Request, read_timeout: float) -> dict:
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
    req = _json_request(url, body, headers)
    return _read_json(req, read_timeout)


def request_form(
    url: str,
    fields: dict[str, str | int],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10,
) -> dict:
    """``application/x-www-form-urlencoded`` POST，响应按 JSON 解析。

    巨潮公告查询等接口走表单而不是 JSON body。
    """
    read_timeout = max(timeout, 1.0)
    data = urllib.parse.urlencode(
        {str(key): str(value) for key, value in fields.items()}
    ).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Accept": "application/json",
            **(headers or {}),
        },
        method="POST",
    )
    return _read_json(req, read_timeout)


def request_content_length(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10,
) -> int | None:
    """发 HEAD 取 ``Content-Length``。拿不到或不支持时返回 None。"""
    read_timeout = max(timeout, 1.0)
    req = urllib.request.Request(
        url, headers=dict(headers or {}), method="HEAD"
    )
    req.timeout = read_timeout
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.headers.get("Content-Length")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        return None
    return value if value >= 0 else None


def request_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    """下载二进制响应。返回 ``(content, content_type)``。

    ``max_bytes`` 超限时截断读取并抛 ``ValueError``，避免把超大文件灌进内存。
    """
    read_timeout = max(timeout, 1.0)
    req = urllib.request.Request(url, headers=dict(headers or {}))
    req.timeout = read_timeout
    with urllib.request.urlopen(req) as resp:
        content_type = str(resp.headers.get("Content-Type") or "")
        if max_bytes is None:
            return resp.read(), content_type
        chunks: list[bytes] = []
        total = 0
        while True:
            block = resp.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise ValueError(f"响应超过上限 {max_bytes} 字节")
            chunks.append(block)
        return b"".join(chunks), content_type


def quote(value: str) -> str:
    """URL query 参数转义。"""
    return urllib.parse.quote(value)
