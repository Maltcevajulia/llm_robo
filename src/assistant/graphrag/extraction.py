"""Извлечение сущностей и связей из чанка. Требование FR-1."""

from __future__ import annotations

import json

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
    EXTRACTION_SCHEMA,
    ENTITY_TYPES,
    RELATION_SIGNATURES,
)
from assistant.infra.grounding import find_codes, normalize_code, canonical_name, code_type
from assistant.infra.config import vocabulary


def _build_messages(chunk: Chunk, vocab: dict[str, list[str]]) -> list[dict]:
    schema_description = (
        "Сущности: " + ", ".join(ENTITY_TYPES) + ".\n"
        "Связи с направлением (subject -> object):\n"
        + "\n".join(f"  {name}: {src} -> {dst}" for name, (src, dst) in RELATION_SIGNATURES.items())
    )
    example = (
        "Пример. Текст: «В модели AR-210 при ошибке E-17 отказывает пусковое реле P-1042, "
        "деталь поставляет КомпрессорМаш.»\n"
        "Ответ: {\"entities\": ["
        "{\"type\":\"Model\",\"name\":\"AR-210\"},"
        "{\"type\":\"ErrorCode\",\"name\":\"E-17\"},"
        "{\"type\":\"Component\",\"name\":\"пусковое реле\"},"
        "{\"type\":\"Part\",\"name\":\"P-1042\"},"
        "{\"type\":\"Supplier\",\"name\":\"КомпрессорМаш\"}], "
        "\"relations\": ["
        "{\"subject_type\":\"ErrorCode\",\"subject\":\"E-17\",\"type\":\"INDICATES\","
        "\"object_type\":\"Component\",\"object\":\"пусковое реле\"},"
        "{\"subject_type\":\"Part\",\"subject\":\"P-1042\",\"type\":\"IS_A\","
        "\"object_type\":\"Component\",\"object\":\"пусковое реле\"},"
        "{\"subject_type\":\"Model\",\"subject\":\"AR-210\",\"type\":\"HAS_PART\","
        "\"object_type\":\"Part\",\"object\":\"P-1042\"},"
        "{\"subject_type\":\"Part\",\"subject\":\"P-1042\",\"type\":\"SUPPLIED_BY\","
        "\"object_type\":\"Supplier\",\"object\":\"КомпрессорМаш\"}]}"
    )
    system = (
        "Ты извлекаешь знания из технической документации по холодильному оборудованию. "
        "Возвращай только JSON по заданной схеме. Извлекай только то, что прямо сказано в тексте. "
        "Не выдумывай коды и названия.\n"
        f"{schema_description}\n"
        "Допустимые названия (для ProductLine, Component, Supplier — только из словаря):\n"
        f"{json.dumps(vocab, ensure_ascii=False, indent=2)}\n"
        f"{example}"
    )
    user = f"Текст чанка:\n\"\"\"\n{chunk.text}\n\"\"\"\n\nВерни JSON."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _validate_entity(
    raw: dict, chunk_text: str, vocab: dict[str, list[str]]
) -> tuple[Entity | None, str | None]:
    etype = raw.get("type")
    if etype not in ENTITY_TYPES:
        return None, f"неизвестный тип сущности {etype!r}"
    name = str(raw.get("name", "")).strip()
    if not name:
        return None, "пустое имя"

    if etype in ("Model", "Part", "ErrorCode", "Procedure"):
        norm = normalize_code(name)
        if not norm or norm not in find_codes(chunk_text):
            return None, f"код {name!r} отсутствует в тексте чанка"
        if code_type(norm) != etype:
            return None, f"код {norm!r} не соответствует типу {etype}"
        name = norm
    else:
        canon = canonical_name(name, vocab)
        if canon is None:
            return None, f"название {name!r} не из словаря"
        name = canon
    return Entity(type=etype, name=name), None


def _validate_relation(raw: dict, by_name: dict[str, Entity]) -> tuple[Relation | None, str | None]:
    rtype = raw.get("type")
    if rtype not in RELATION_SIGNATURES:
        return None, f"неизвестный тип связи {rtype!r}"
    subj = by_name.get(str(raw.get("subject", "")).strip())
    obj = by_name.get(str(raw.get("object", "")).strip())
    if subj is None or obj is None:
        return None, "связь с отброшенным концом"
    rel = Relation(subject=subj, type=rtype, object=obj)
    if not rel.matches_schema():
        return None, f"направление связи {rtype} не по схеме"
    return rel, None


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    vocab = vocabulary()
    messages = _build_messages(chunk, vocab)

    raw = None
    for _ in range(2):
        try:
            resp = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
            raw = json.loads(resp.content)
            if isinstance(raw, dict):
                break
            raw = None
        except Exception:
            raw = None
    if not isinstance(raw, dict):
        return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, dropped=1)

    dropped = 0
    entities: list[Entity] = []
    # ключ — то имя, что вернула модель; значение — каноническая Entity
    by_name: dict[str, Entity] = {}

    for ent_raw in raw.get("entities", []) or []:
        ent, err = _validate_entity(ent_raw, chunk.text, vocab)
        if err:
            dropped += 1
            continue
        entities.append(ent)
        by_name[str(ent_raw.get("name", "")).strip()] = ent

    relations: list[Relation] = []
    for rel_raw in raw.get("relations", []) or []:
        rel, err = _validate_relation(rel_raw, by_name)
        if err:
            dropped += 1
            continue
        relations.append(rel)

    return ExtractionResult(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        entities=entities,
        relations=relations,
        dropped=dropped,
    )
