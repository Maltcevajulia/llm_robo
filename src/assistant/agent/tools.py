"""Пять инструментов агента. Требование FR-6."""

from __future__ import annotations

import re
from typing import Any, Callable

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.graphrag.search import local_search, global_search

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
PART_RE = re.compile(r"^P-\d{4}[A-Z]?$")


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Ищет факты в графе знаний: коды ошибок, детали, замены, регламенты.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Обзорный ответ по сводкам сообществ (частые отказы, тенденции).",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Остаток детали на складе и признак снятия с производства.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "pattern": "^P-\\d{4}[A-Z]?$"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                },
                "required": ["part_id", "warehouse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Цена детали в рублях и признак снятия с производства.",
            "parameters": {
                "type": "object",
                "properties": {"part_id": {"type": "string", "pattern": "^P-\\d{4}[A-Z]?$"}},
                "required": ["part_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создаёт заказ детали на складе. Требует подтверждения человеком.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "pattern": "^P-\\d{4}[A-Z]?$"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                    "qty": {"type": "integer", "minimum": 1},
                    "model": {"type": "string"},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            },
        },
    },
]


def _validate_part(part_id: Any) -> str | None:
    if not isinstance(part_id, str) or not PART_RE.match(part_id):
        return f"некорректный номер детали: {part_id!r}"
    return None


def _validate_warehouse(warehouse: Any) -> str | None:
    if warehouse not in WAREHOUSES:
        return f"неизвестный склад: {warehouse!r}"
    return None


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    def search_knowledge(question: str) -> dict:
        ans = local_search(question, kb, llm)
        return {
            "answer": ans.text,
            "facts": ans.facts,
            "sources": ans.sources,
            "refused": ans.refused,
        }

    def summarize_topic(question: str) -> dict:
        ans = global_search(question, kb, llm)
        return {"answer": ans.text, "sources": ans.sources, "refused": ans.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        err = _validate_part(part_id) or _validate_warehouse(warehouse)
        if err:
            return {"error": err}
        try:
            info = erp.get_stock(part_id, warehouse)
        except (ErpNotFound, ErpRejected) as exc:
            return {"error": str(exc)}
        return {
            "part_id": part_id,
            "warehouse": warehouse,
            "qty": info["qty"],
            "discontinued": bool(info.get("discontinued", False)),
        }

    def get_price(part_id: str) -> dict:
        err = _validate_part(part_id)
        if err:
            return {"error": err}
        try:
            info = erp.get_price(part_id)
        except (ErpNotFound, ErpRejected) as exc:
            return {"error": str(exc)}
        return {
            "part_id": part_id,
            "price_rub": info["price_rub"],
            "discontinued": bool(info.get("discontinued", False)),
        }

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        err = _validate_part(part_id) or _validate_warehouse(warehouse)
        if err:
            return {"error": err}
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            return {"error": "количество должно быть больше нуля"}
        try:
            order = erp.create_order(part_id, warehouse, qty, model)
        except (ErpNotFound, ErpRejected) as exc:
            return {"error": str(exc)}
        return {
            "order_id": order["order_id"],
            "part_id": part_id,
            "warehouse": warehouse,
            "qty": qty,
            "model": model,
        }

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
