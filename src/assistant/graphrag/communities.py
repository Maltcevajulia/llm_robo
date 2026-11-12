"""Сообщества графа и их сводки. Требование FR-3."""

from __future__ import annotations

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    undirected = graph.to_undirected()
    for _, _, data in undirected.edges(data=True):
        data["weight"] = max(1, int(data.get("weight", 1)))

    partitions = nx.community.louvain_communities(undirected, weight="weight", seed=42)
    ordered = sorted(partitions, key=lambda s: (-len(s), sorted(s)[0] if s else ""))
    return [
        Community(community_id=i, nodes=sorted(nodes), summary="")
        for i, nodes in enumerate(ordered)
    ]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    nodes = community.nodes
    if not nodes:
        return ""

    type_counter: Counter[str] = Counter()
    for n in nodes:
        type_counter[graph.nodes[n].get("type", "?")] += 1

    sub = graph.subgraph(nodes)
    internal = [
        (data.get("weight", 1), data.get("type", "?"), u, v)
        for u, v, data in sub.edges(data=True)
    ]
    internal.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    top_edges = internal[:15]

    weighted_deg = {
        n: sum(d.get("weight", 1) for _, _, d in sub.edges(n, data=True))
        for n in nodes
    }
    top_nodes = sorted(weighted_deg.items(), key=lambda x: (-x[1], x[0]))[:10]

    lines = ["Состав сообщества по типам сущностей:"]
    for t, c in type_counter.most_common():
        lines.append(f"  {t}: {c}")
    lines.append("Самые весомые связи (вес × тип: откуда -> куда):")
    for w, t, u, v in top_edges:
        un = graph.nodes[u].get("name", u)
        vn = graph.nodes[v].get("name", v)
        lines.append(f"  {w} × {t}: {un} -> {vn}")
    lines.append("Наиболее связанные узлы (взвешенная степень):")
    for n, w in top_nodes:
        name = graph.nodes[n].get("name", n)
        etype = graph.nodes[n].get("type", "?")
        lines.append(f"  {name} ({etype}): {w}")

    facts = "\n".join(lines)
    messages = [
        {
            "role": "system",
            "content": (
                "Ты пишешь сводку сообщества графа знаний о холодильном оборудовании. "
                "Используй только приведённые числа и названия, ничего не выдумывай. "
                "Длина — не более 150 слов."
            ),
        },
        {"role": "user", "content": facts},
    ]
    resp = llm.chat(messages)
    words = resp.content.strip().split()
    return " ".join(words[:150])
