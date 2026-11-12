from __future__ import annotations
from collections import defaultdict
import networkx as nx
from assistant.infra.schemas import ExtractionResult

def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    g = nx.DiGraph()
    g.graph["conflicts"] = 0

    # 1. Собираем узлы и множество chunk_id-источников для каждого потенциального ребра.
    #    Ключ ребра: (subject.node_id, object.node_id, type).
    edge_sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for res in results:
        for ent in res.entities:
            nid = ent.node_id
            if g.has_node(nid):
                if res.chunk_id not in g.nodes[nid]["sources"]:
                    g.nodes[nid]["sources"].append(res.chunk_id)
            else:
                g.add_node(
                    nid,
                    type=ent.type,
                    name=ent.name,
                    sources=[res.chunk_id],
                )

    # 2. Собираем рёбра, проверяя схему. Пока только копим источники, в граф не добавляем.
    for res in results:
        for rel in res.relations:
            if not rel.matches_schema():
                continue
            u = rel.subject.node_id
            v = rel.object.node_id
            if not g.has_node(u) or not g.has_node(v):
                continue
            edge_sources[(u, v, rel.type)].add(res.chunk_id)

    # 3. Разрешаем встречные REPLACED_BY.
    #    Разделяем рёбра на обычные и пары замен.
    plain: list[tuple[str, str, str]] = []
    pairs: dict[frozenset[str], list[tuple[str, str, str]]] = defaultdict(list)

    for (u, v, t) in edge_sources:
        if t == "REPLACED_BY":
            pairs[frozenset((u, v))].append((u, v, t))
        else:
            plain.append((u, v, t))

    def _add_edge(u: str, v: str, t: str) -> None:
        sources = sorted(edge_sources[(u, v, t)])
        g.add_edge(u, v, type=t, weight=len(sources), sources=sources)

    # Обычные рёбра добавляем сразу.
    for u, v, t in plain:
        _add_edge(u, v, t)

    # Для пар замен:
    # - если связь только в одну сторону, добавляем её;
    # - если в обе, оставляем более весомую, при равенстве не добавляем ни одну.
    for _pair, rels in pairs.items():
        if len(rels) == 1:
            _add_edge(*rels[0])
            continue

        weighted = sorted(
            ((len(edge_sources[r]), r) for r in rels),
            key=lambda x: x[0],
            reverse=True,
        )
        top_weight = weighted[0][0]
        top = [r for w, r in weighted if w == top_weight]

        if len(top) == 1:
            _add_edge(*top[0])
        # если top_weight совпал у обоих направлений — не добавляем ничего
        g.graph["conflicts"] += 1

    return g
