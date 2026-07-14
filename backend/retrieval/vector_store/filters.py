"""Mongo-style metadata filter DSL used to restrict vector search results.

Supported shapes:
    {"field": value}                    equality shorthand
    {"field": {"$eq": value}}
    {"field": {"$ne": value}}
    {"field": {"$in": [v1, v2]}}
    {"field": {"$nin": [v1, v2]}}
    {"field": {"$gt"/"$gte"/"$lt"/"$lte": value}}
    {"$and": [filter, filter, ...]}
    {"$or": [filter, filter, ...]}

Multiple keys in one dict are implicitly AND-ed together, and multiple operators
on the same field are implicitly AND-ed too (e.g. a range: {"$gte": 1, "$lte": 9}).
"""

from __future__ import annotations

from typing import Any

from backend.retrieval.vector_store.types import Filter

_COMPARISON_OPERATORS = {"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"}


def _apply_operator(op: str, value: Any, operand: Any) -> bool:
    if op == "$eq":
        return value == operand
    if op == "$ne":
        return value != operand
    if op == "$in":
        return value in operand
    if op == "$nin":
        return value not in operand
    if op == "$gt":
        return value is not None and value > operand
    if op == "$gte":
        return value is not None and value >= operand
    if op == "$lt":
        return value is not None and value < operand
    if op == "$lte":
        return value is not None and value <= operand
    raise ValueError(f"Unknown filter operator: {op}")


def matches(metadata: dict[str, Any], filter_: Filter | None) -> bool:
    """Returns whether `metadata` satisfies `filter_`. An empty/None filter always
    matches (no restriction).
    """
    if not filter_:
        return True

    for key, condition in filter_.items():
        if key == "$and":
            if not all(matches(metadata, sub_filter) for sub_filter in condition):
                return False
            continue
        if key == "$or":
            if not any(matches(metadata, sub_filter) for sub_filter in condition):
                return False
            continue

        value = metadata.get(key)
        if (
            isinstance(condition, dict)
            and condition
            and all(k in _COMPARISON_OPERATORS for k in condition)
        ):
            for op, operand in condition.items():
                if not _apply_operator(op, value, operand):
                    return False
        else:
            if value != condition:
                return False

    return True
