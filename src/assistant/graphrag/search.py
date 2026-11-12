"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from __future__ import annotations

import json
from collections import deque

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase, MAP_SCHEMA
from assistant.infra.grounding import find_codes, find_names

MAX_FACTS_CHARS = 8000
NOT_FOUND = "в базе знаний не найдено"


def _doc_of(chunk_id: str) -> str:
    """chunk_id = 'manual_AR-210.md#1' -> doc_id = 'manual_AR-210.md'."""
    return chunk_id.split("#", 1)[0]


def _find_anchor_nodes(question: str, kb: KnowledgeBase) -> list[str]:
    """Опорные узлы графа по кодам и названиям из вопроса."""
    codes = find_codes(question) or []
    try:
        names = find_names(question, kb.vocabulary) or []
    except TypeError:
        # fallback, если сигнатура отличается
        try:
            names = find_names(question) or []
        except Exception:
            names = []

    wanted = set(codes) | set(names)
    if not wanted:
        return []

    anchors: list[str] = []
    for nid, data in kb.graph.nodes(data=True):
        if data.get("name") in wanted:
            anchors.append(nid)
    return list(dict.fromkeys(anchors))


def _edges_around(g: nx.DiGraph, node: str, depth: int) -> list[tuple[str, str, dict]]:
    """Рёбра в окрестности узла до заданной глубины (направление не важно)."""
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, dict]] = []
    q = deque([(node, 0)])
    visited = {node}
    while q:
        cur, d = q.popleft()
        if d >= depth:
            continue
        neighbors = set(g.successors(cur)) | set(g.predecessors(cur))
        for nbr in neighbors:
            for u, v in ((cur, nbr), (nbr, cur)):
                if g.has_edge(u, v):
                    data = g.get_edge_data(u, v)
                    key = (u, v, data.get("type"))
                    if key not in seen:
                        seen.add(key)
                        out.append((u, v, data))
            if nbr not in visited:
                visited.add(nbr)
                q.append((nbr, d + 1))
    return out


def _edges_between(
    g: nx.DiGraph, anchors: list[str], max_len: int = 3
) -> list[tuple[str, str, dict]]:
    """Все рёбра на простых путях длиной до max_len между опорными узлами."""
    ug = g.to_undirected(as_view=True)
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, dict]] = []
    for a in anchors:
        for b in anchors:
            if a == b:
                continue
            try:
                paths = list(nx.all_simple_paths(ug, a, b, cutoff=max_len))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            for path in paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    for x, y in ((u, v), (v, u)):
                        if g.has_edge(x, y):
                            data = g.get_edge_data(x, y)
                            key = (x, y, data.get("type"))
                            if key not in seen:
                                seen.add(key)
                                out.append((x, y, data))
    return out


def _add_replacement_chains(
    g: nx.DiGraph, edges: list[tuple[str, str, dict]]
) -> list[tuple[str, str, dict]]:
    """Добавляет цепочки REPLACED_BY от всех деталей, встретившихся на путях."""
    seen: set[tuple[str, str, str]] = {(u, v, d.get("type")) for u, v, d in edges}
    out = list(edges)
    parts: set[str] = set()
    for u, v, _ in edges:
        if g.nodes[u].get("type") == "Part":
            parts.add(u)
        if g.nodes[v].get("type") == "Part":
            parts.add(v)

    for p in parts:
        cur, guard = p, 0
        while guard < 10:
            guard += 1
            succ = [
                v
                for v in g.successors(cur)
                if g.get_edge_data(cur, v).get("type") == "REPLACED_BY"
            ]
            if not succ:
                break
            nxt = succ[0]
            key = (cur, nxt, "REPLACED_BY")
            if key not in seen:
                seen.add(key)
                out.append((cur, nxt, g.get_edge_data(cur, nxt)))
            cur = nxt
    return out


def _facts_and_sources(
    g: nx.DiGraph, edges: list[tuple[str, str, dict]]
) -> tuple[list[str], list[str]]:
    """Нумерованный список фактов и отсортированные doc_id-источники."""
    lines: list[str] = []
    sources: set[str] = set()
    seen: set[tuple[str, str, str]] = set()
    for u, v, data in edges:
        key = (u, v, data.get("type"))
        if key in seen:
            continue
        seen.add(key)
        un = g.nodes[u].get("name", u)
        vn = g.nodes[v].get("name", v)
        lines.append(f"{un} -{data.get('type')}-> {vn}")
        for cid in data.get("sources", []) or []:
            sources.add(_doc_of(cid))

    # ограничение по объёму
    trimmed: list[str] = []
    total = 0
    for i, line in enumerate(lines, 1):
        prefix = f"{i}. "
        if total + len(prefix) + len(line) > MAX_FACTS_CHARS:
            break
        trimmed.append(prefix + line)
        total += len(prefix) + len(line)
    return trimmed, sorted(sources)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    g = kb.graph
    codes = find_codes(question) or []

    # Если код из вопроса не встречается в графе — отказ без вызова модели.
    graph_names = {d.get("name") for _, d in g.nodes(data=True)}
    for code in codes:
        if code not in graph_names:
            return Answer(text=NOT_FOUND, sources=[], facts=[], refused=True)

    anchors = _find_anchor_nodes(question, kb)
    if not anchors:
        return Answer(text=NOT_FOUND, sources=[], facts=[], refused=True)

    if len(anchors) >= 2:
        edges = _edges_between(g, anchors, max_len=3)
    else:
        edges = _edges_around(g, anchors[0], depth=2)

    edges = _add_replacement_chains(g, edges)
    facts, sources = _facts_and_sources(g, edges)

    if not facts:
        return Answer(text=NOT_FOUND, sources=[], facts=[], refused=True)

    facts_block = "\n".join(facts)
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — диспетчер сервисной службы. Отвечай только по пронумерованным фактам. "
                "Если в фактах есть цепочка REPLACED_BY, назови последнюю деталь цепочки. "
                "Не выдумывай коды, цены и остатки. Если фактов недостаточно, ответь ровно: "
                f"«{NOT_FOUND}»."
            ),
        },
        {"role": "user", "content": f"Вопрос: {question}\n\nФакты:\n{facts_block}"},
    ]
    resp = llm.chat(messages)
    text = (resp.content or "").strip()
    refused = text.lower().startswith(NOT_FOUND.lower())
    return Answer(text=text, sources=sources, facts=facts, refused=refused)



def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    summaries = [c for c in kb.communities if c.summary]
    if not summaries:
        return Answer(text=NOT_FOUND, sources=[], facts=[], refused=True)

    partials: list[tuple[str, int]] = []
    for comm in summaries:
        messages = [
            {
                "role": "system",
                "content": (
                    "По сводке сообщества ответь на вопрос максимально конкретно. "
                    "Если в сводке нет ответа — верни пустую строку и score 0. "
                    "Формат ответа — JSON {\"answer\": \"...\", \"score\": 0..10}."
                ),
            },
            {
                "role": "user",
                "content": f"Вопрос: {question}\nСводка сообщества:\n{comm.summary}",
            },
        ]
        try:
            resp = llm.chat(messages, json_schema=MAP_SCHEMA)
            payload = json.loads(resp.content)
            ans = str(payload.get("answer", "")).strip()
            score = int(payload.get("score", 0))
        except Exception:
            # нечитаемый частичный ответ пропускаем
            continue
        if ans and score > 0:
            partials.append((ans, score))

    if not partials:
        return Answer(text=NOT_FOUND, sources=[], facts=[], refused=True)

    joined = "\n".join(f"- ({s}) {a}" for a, s in partials)
    messages = [
        {
            "role": "system",
            "content": (
                "Объедини частичные ответы в один связный ответ на вопрос. "
                "Не добавляй ничего, чего нет в частичных ответах."
            ),
        },
        {"role": "user", "content": f"Вопрос: {question}\nЧастичные ответы:\n{joined}"},
    ]
    resp = llm.chat(messages)
    text = (resp.content or "").strip() or NOT_FOUND
    refused = text.lower().startswith(NOT_FOUND.lower())
    return Answer(
        text=text,
        sources=[],
        facts=[line.lstrip("- ").strip() for line in joined.splitlines()],
        refused=refused,
    )
