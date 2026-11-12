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
        # Приводим к нижнему регистру для поиска в словаре
        name = name.lower()
        # Передаем 3 аргумента, как требует функция
        canon = canonical_name(name, etype, vocab)
        if canon is None:
            return None, f"название {name!r} не из словаря"
        name = canon
        
    return Entity(type=etype, name=name), None


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
    entities: dict[str, Entity] = {}  # node_id -> Entity
    by_name: dict[str, Entity] = {}   # исходное имя -> Entity

    # 1. Сначала добавляем сущности из списка entities
    entities_raw = raw.get("entities")
    if not isinstance(entities_raw, list):
        entities_raw = []
    
    for ent_raw in entities_raw:
        if not isinstance(ent_raw, dict):
            dropped += 1
            continue
        ent, err = _validate_entity(ent_raw, chunk.text, vocab)
        if err:
            dropped += 1
            continue
        entities[ent.node_id] = ent
        by_name[str(ent_raw.get("name", "")).strip()] = ent

    # 2. Обрабатываем связи
    relations: list[Relation] = []
    relations_raw = raw.get("relations")
    if not isinstance(relations_raw, list):
        relations_raw = []

    for rel_raw in relations_raw:
        if not isinstance(rel_raw, dict):
            dropped += 1
            continue
        
        rtype = rel_raw.get("type")
        if rtype not in RELATION_SIGNATURES:
            dropped += 1
            continue
            
        subj_name = str(rel_raw.get("subject", "")).strip()
        obj_name = str(rel_raw.get("object", "")).strip()
        
        # Проверяем subject
        if subj_name in by_name:
            subj = by_name[subj_name]
        else:
            subj_type = rel_raw.get("subject_type")
            subj, err = _validate_entity({"type": subj_type, "name": subj_name}, chunk.text, vocab)
            if err:
                dropped += 1
                continue
            entities[subj.node_id] = subj
            by_name[subj_name] = subj
            
        # Проверяем object
        if obj_name in by_name:
            obj = by_name[obj_name]
        else:
            obj_type = rel_raw.get("object_type")
            obj, err = _validate_entity({"type": obj_type, "name": obj_name}, chunk.text, vocab)
            if err:
                dropped += 1
                continue
            entities[obj.node_id] = obj
            by_name[obj_name] = obj
            
        rel = Relation(subject=subj, type=rtype, object=obj)
        if not rel.matches_schema():
            dropped += 1
            continue
        relations.append(rel)

    return ExtractionResult(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        entities=list(entities.values()),
        relations=relations,
        dropped=dropped,
    )
