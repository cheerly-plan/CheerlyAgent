"""图谱抽取的输出结构，以及模型输出结构的容错解析。

从 graph_extract.py 拆出(S5)。这里只管「模型返回的 JSON 长什么样」：
结构定义（_Entity / _Relation / _Extraction）和把任意形状的输出收敛成
结构的解析（_parse_extraction）。

容错之所以必要：模型在上下文吃紧时会悄悄换个 schema，直接丢弃等于静默
丢一批人。紧凑数组（e/r）是主线——同样的内容省掉逐项重复的
name/type/description/chunk 键，输出 token 直接砍一半，而生成 token 就是
这类任务的全部耗时；完整结构仍然认，老版本输出、别家模型也未必听话。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class _Entity(BaseModel):
    name: str
    type: str = "other"
    description: str = ""
    chunk: int = 0


class _Relation(BaseModel):
    source: str
    target: str
    relation: str = "相关"
    description: str = ""
    chunk: int = 0


class _Extraction(BaseModel):
    entities: list[_Entity] = Field(default_factory=list)
    relations: list[_Relation] = Field(default_factory=list)


def _parse_extraction(data: object) -> _Extraction | None:
    """把模型返回的 dict 解成 _Extraction，认不出结构返回 None。

    紧凑格式（e / r 数组）是主线：同样的内容，省掉逐项重复的
    name/type/description/chunk 键，输出 token 直接砍一半，而生成 token
    就是这类任务的全部耗时。冗长格式仍然认——模型偶尔会自己换回去，
    老版本输出、别家模型也未必听话，多认一种不亏。
    """
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("e"), list) or isinstance(data.get("r"), list):
        return _from_compact(data)
    if isinstance(data.get("entities"), list) or isinstance(data.get("relations"), list):
        try:
            return _Extraction.model_validate(data)
        except Exception:
            return None
    return None


def _from_compact(data: dict) -> _Extraction:
    out = _Extraction()
    for item in data.get("e") or []:
        row = _compact_row(item, 4, ("name", "type", "description", "chunk"))
        if row is None:
            continue
        name = str(row[0] or "").strip()
        if not name:
            continue
        out.entities.append(
            _Entity(
                name=name,
                type=str(row[1] or "other"),
                description=str(row[2] or ""),
                chunk=_as_int(row[3]),
            )
        )
    for item in data.get("r") or []:
        row = _compact_row(
            item, 5, ("source", "target", "relation", "description", "chunk")
        )
        if row is None:
            continue
        row = _fix_short_relation(item, row)
        if row is None:
            continue
        source = str(row[0] or "").strip()
        target = str(row[1] or "").strip()
        if not source or not target:
            continue
        out.relations.append(
            _Relation(
                source=source,
                target=target,
                relation=str(row[2] or "相关"),
                description=str(row[3] or ""),
                chunk=_as_int(row[4]),
            )
        )
    return out


_NUMERIC_ROW = re.compile(r"^[-\d,.\s%]+$")


def _fix_short_relation(item: object, row: list) -> list | None:
    """只有 4 项的 r 行：分清是「漏了说明」还是「漏了客体」。

    `_compact_row` 会把短行补到 5 项，补出来的位置全错。实测本地模型对
    「持股」这类关系常写成 [主体, 关系, 空或数值, 序号]——客体被吞掉，
    硬补就变成「公司 --相关--> 持有股份」这种假边，还会把谓语当成实体。
    两种形态靠第 3 项区分：是空/纯数值 = 漏了客体（整条丢掉，不猜）；
    是文字 = [主体, 客体, 关系, 序号]（照常用，说明补空）。
    """
    if not isinstance(item, (list, tuple)) or len(item) != 4:
        return row
    third = item[2]
    text = "" if third is None else str(third).strip()
    if not text or _NUMERIC_ROW.match(text):
        return None
    return [item[0], item[1], item[2], "", item[3]]


def _compact_row(item: object, size: int, keys: tuple[str, ...]) -> list | None:
    """把一行数组/对象规整成定长 list；模型混着给也接。"""
    if isinstance(item, (list, tuple)):
        row = list(item)
    elif isinstance(item, dict):
        row = [item.get(k) for k in keys]
    else:
        return None
    if len(row) < 2:
        return None
    return (row + [None] * size)[:size]


def _as_int(value: object) -> int:
    try:
        return max(0, int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
