"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.schemas import AgentResult, ToolStep
from assistant.agent.tools import TOOL_SCHEMAS, WAREHOUSES

MAX_ERP_RETRIES = 3  # первая попытка + 2 повтора
NOT_FOUND = "в базе знаний не найдено"

CODE_RE = re.compile(r"\b(?:P-\d{4}[A-Z]?|AR-\d{3}|BZ-\d{3}|GL-\d{3}|E-\d{2,3}|SP-\d{3})\b")
PRICE_RE = re.compile(r"(\d[\d\s]*)\s*(?:руб|₽|RUB)", re.IGNORECASE)
QTY_RE = re.compile(r"\b(\d+)\s*(?:шт|штук)", re.IGNORECASE)
PART_RE = re.compile(r"^P-\d{4}[A-Z]?$")

SYSTEM_PROMPT = (
    "Ты — ассистент сервисной службы компании «ПолярТех».\n"
    "Помоги диспетчеру найти нужную деталь по коду ошибки и модели оборудования, "
    "проверить остаток и цену, и при необходимости подготовить заказ.\n\n"
    "Инструменты:\n"
    "  search_knowledge(question)         — поиск фактов в графе знаний;\n"
    "  summarize_topic(question)          — обзорный ответ по сводкам сообществ;\n"
    "  check_stock(part_id, warehouse)    — остаток детали на складе;\n"
    "  get_price(part_id)                 — цена детали;\n"
    "  create_service_order(part_id, warehouse, qty, model) — заказ детали.\n\n"
    "Правила:\n"
    "1. Сначала найди деталь через search_knowledge. Не выдумывай номера деталей.\n"
    "2. Если деталь снята с производства, предложи замену — последнюю в цепочке REPLACED_BY.\n"
    "3. Склады: " + ", ".join(WAREHOUSES) + ".\n"
    "4. Количество для заказа должно быть явно названо в вопросе; если его нет — "
    "задай уточняющий вопрос.\n"
    "5. В ответе указывай коды деталей, цены и остатки.\n"
    "6. Не создавай заказ без явного указания пользователя.\n"
)



def _extract_codes(text: str) -> set[str]:
    return set(CODE_RE.findall(text or ""))


def _validate_args(name: str, args: dict, question: str, known_text: str) -> str | None:
    """Возвращает текст ошибки или None, если аргументы валидны."""
    if not isinstance(args, dict):
        return "аргументы должны быть объектом JSON"
    if "_raw" in args:
        return "нечитаемые аргументы (не JSON)"

    if name in ("search_knowledge", "summarize_topic"):
        q = args.get("question")
        if not isinstance(q, str) or not q.strip():
            return "не задан question"
        return None

    if name == "get_price":
        pid = args.get("part_id")
        if not isinstance(pid, str) or not PART_RE.match(pid):
            return f"некорректный part_id: {pid!r}"
        if pid not in known_text:
            return (
                "part_id не встречается в вопросе или результатах прошлых инструментов; "
                "сначала вызови search_knowledge"
            )
        return None

    if name == "check_stock":
        pid = args.get("part_id")
        wh = args.get("warehouse")
        if not isinstance(pid, str) or not PART_RE.match(pid):
            return f"некорректный part_id: {pid!r}"
        if wh not in WAREHOUSES:
            return f"неизвестный склад: {wh!r}"
        if pid not in known_text:
            return (
                "part_id не встречается в вопросе или результатах прошлых инструментов; "
                "сначала вызови search_knowledge"
            )
        return None

    if name == "create_service_order":
        pid = args.get("part_id")
        wh = args.get("warehouse")
        qty = args.get("qty")
        model = args.get("model")
        if not isinstance(pid, str) or not PART_RE.match(pid):
            return f"некорректный part_id: {pid!r}"
        if wh not in WAREHOUSES:
            return f"неизвестный склад: {wh!r}"
        if pid not in known_text:
            return (
                "part_id не встречается в вопросе или результатах прошлых инструментов; "
                "сначала вызови search_knowledge"
            )
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            return "qty должно быть целым числом больше нуля"
        if not isinstance(model, str) or not model.strip():
            return "не задан model"
        if not re.search(rf"(?<!\d){qty}(?!\d)", question):
            return (
                "количество не названо в вопросе; задай уточняющий вопрос пользователю, "
                "прежде чем оформлять заказ"
            )
        return None

    return f"неизвестный инструмент: {name}"


def _call_with_retries(
    fn: Callable[..., dict], args: dict
) -> tuple[dict | None, str | None]:
    """Повторяет вызов при ErpUnavailable до MAX_ERP_RETRIES раз."""
    last_err: str | None = None
    for _ in range(MAX_ERP_RETRIES):
        try:
            return fn(**args), None
        except ErpUnavailable as exc:
            last_err = str(exc)
            continue
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    return None, f"ERP недоступна после {MAX_ERP_RETRIES} попыток: {last_err}"


def _is_consistent(answer: str, question: str, tool_text: str) -> bool:
    """Все коды, цены и количества из ответа должны встречаться в вопросе или результатах."""
    haystack = question + "\n" + tool_text
    haystack_nospace = re.sub(r"\s+", "", haystack)

    for code in _extract_codes(answer):
        if code not in haystack:
            return False
    for m in PRICE_RE.finditer(answer):
        num = re.sub(r"\s+", "", m.group(1))
        if num and num not in haystack_nospace:
            return False
    for m in QTY_RE.finditer(answer):
        if m.group(1) not in haystack:
            return False
    return True




def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    sources: set[str] = set()
    tool_text_parts: list[str] = []
    cache: dict[tuple[str, str], dict] = {}
    step_count = 0
    last_answer = ""

    while step_count < max_steps:
        resp = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(resp.as_message())
        if resp.content:
            last_answer = resp.content

        if not resp.tool_calls:
            break  # модель дала финальный ответ

        for call in resp.tool_calls:
            if step_count >= max_steps:
                break
            step_count += 1

            name = call.name
            raw_args = call.arguments
            args: dict = (
                raw_args if isinstance(raw_args, dict) else {"_raw": str(raw_args)}
            )

           
            if name not in tools:
                err = f"неизвестный инструмент: {name}"
                steps.append(ToolStep(tool=name, args=dict(args), error=err))
                messages.append(tool_message(call, {"error": err}))
                continue

         
            known_text = question + "\n" + "\n".join(tool_text_parts)
            err = _validate_args(name, args, question, known_text)
            if err:
                steps.append(ToolStep(tool=name, args=dict(args), error=err))
                messages.append(tool_message(call, {"error": err}))
                continue

         
            if name == "create_service_order" and not confirm:
                steps.append(
                    ToolStep(tool=name, args=dict(args), error="требуется подтверждение")
                )
                return AgentResult(
                    answer="Готов оформить заказ, нужно подтверждение.",
                    steps=steps,
                    sources=sorted(sources),
                    refused=False,
                    needs_confirmation=True,
                    pending_action={"tool": name, "args": dict(args)},
                )

            cache_key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            if cache_key in cache:
                result = cache[cache_key]
            else:
                result, call_err = _call_with_retries(tools[name], args)
                if call_err:
                    steps.append(ToolStep(tool=name, args=dict(args), error=call_err))
                    messages.append(tool_message(call, {"error": call_err}))
                    continue
                cache[cache_key] = result

           
            steps.append(ToolStep(tool=name, args=dict(args), result=result))
            tool_text_parts.append(json.dumps(result, ensure_ascii=False))
            for s in result.get("sources", []) or []:
                sources.add(s)
            messages.append(tool_message(call, result))

   
    if step_count >= max_steps:
        try:
            final_resp = llm.chat(messages)  # без tools
            if final_resp.content:
                last_answer = final_resp.content
        except Exception:
            pass


    if not last_answer:
        return AgentResult(
            answer=NOT_FOUND,
            steps=steps,
            sources=sorted(sources),
            refused=True,
            needs_confirmation=False,
            pending_action=None,
        )

   
    tool_text = "\n".join(tool_text_parts)
    if not _is_consistent(last_answer, question, tool_text):
        fix_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "В твоём ответе есть коды, цены или количества, которых нет в вопросе "
                    "или в результатах инструментов. Исправь ответ, опираясь только на факты."
                ),
            }
        ]
        try:
            fixed_resp = llm.chat(fix_messages)
            fixed_text = (fixed_resp.content or "").strip()
        except Exception:
            fixed_text = ""

        if not fixed_text or not _is_consistent(fixed_text, question, tool_text):
            return AgentResult(
                answer=NOT_FOUND,
                steps=steps,
                sources=sorted(sources),
                refused=True,
                needs_confirmation=False,
                pending_action=None,
            )
        last_answer = fixed_text

    refused = last_answer.lower().startswith(NOT_FOUND.lower())
    return AgentResult(
        answer=last_answer,
        steps=steps,
        sources=sorted(sources),
        refused=refused,
        needs_confirmation=False,
        pending_action=None,
    )
