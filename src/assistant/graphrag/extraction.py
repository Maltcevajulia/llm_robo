"""Извлечение сущностей и связей из чанка. Требование FR-1."""

from __future__ import annotations

import json
import re
from typing import Any

from assistant.infra.config import vocabulary
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)


def _extract_valid_codes(text: str) -> set[str]:
    """Извлекает все валидные коды из текста (с учетом русских букв и пробелов)."""
    # Ищем паттерн: 1-3 буквы, пробел или дефис, 2-4 цифры, опционально еще дефис и цифры, 
    # опционально буква в конце.
    pattern = r'(?<![A-ZА-Я])([A-ZА-Я]{1,3})[\s\-]+(\d{2,4})([\s\-]+\d{2,4})?([A-ZА-Я]?)(?![A-ZА-Я])'
    matches = re.findall(pattern, text.upper())
    
    # Таблица замены русских букв на английские
    trans = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
    codes = set()
    for prefix, digits, sub_digits, suffix in matches:
        p = prefix.translate(trans)
        s = suffix.translate(trans)
        d2 = sub_digits.translate(trans) if sub_digits else ""
        codes.add(f"{p}-{digits}{d2}{s}")
    return codes


def _build_messages(chunk: Chunk, vocab: dict[str, list[str]]) -> list[dict]:
    schema_description = (
        "Сущности: " + ", ".join(ENTITY_TYPES) + ".\n"
        "Связи и направление (subject -> object):\n"
        + "\n".join(f"{name}: {src} -> {dst}" for name, (src, dst) in RELATION_SIGNATURES.items())
    )
    
    example = (
        "Пример. Текст: «В модели AR-210 при ошибке E-17 отказывает пусковое реле P-1042, "
        "деталь поставляет КомпрессорМаш.»\n"
        "Ответ:\n"
        "{\n"
        '  "entities": [\n'
        '    {"type": "Model", "name": "AR-210"},\n'
        '    {"type": "ErrorCode", "name": "E-17"},\n'
        '    {"type": "Component", "name": "пусковое реле"},\n'
        '    {"type": "Part", "name": "P-1042"},\n'
        '    {"type": "Supplier", "name": "КомпрессорМаш"}\n'
        "  ],\n"
        '  "relations": [\n'
        '    {"subject_type": "ErrorCode", "subject": "E-17", "type": "INDICATES", \n'
        '     "object_type": "Component", "object": "пусковое реле"},\n'
        '    {"subject_type": "Part", "subject": "P-1042", "type": "IS_A", \n'
        '     "object_type": "Component", "object": "пусковое реле"},\n'
        '    {"subject_type": "Model", "subject": "AR-210", "type": "HAS_PART", \n'
        '     "object_type": "Part", "object": "P-1042"},\n'
        '    {"subject_type": "Part", "subject": "P-1042", "type": "SUPPLIED_BY", \n'
        '     "object_type": "Supplier", "object": "КомпрессорМаш"}\n'
        "  ]\n"
        "}"
    )

    system = (
        "Ты — эксперт по извлечению знаний из технической документации.\n"
        "Возвращай строго валидный JSON по заданной схеме.\n"
        "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
        "1. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО переводить слова на английский язык. "
        "Используй ТОЛЬКО те слова, которые есть в тексте чанка.\n"
        "2. Строго соблюдай направление связей (subject -> object) согласно схеме. "
        "Например, связь SUPPLIED_BY всегда идет от Part к Supplier, а HAS_PART от Model к Part.\n"
        "3. Для связи кода ошибки (ErrorCode) с компонентом используй тип INDICATES.\n"
        "4. ИЗВЛЕКАЙ ВСЕ СВЯЗИ (ЖАДНОСТЬ). Если в тексте упоминается 5 деталей и 3 поставщика, "
        "ты должен вернуть связи для каждой. Не пропускай ни одной сущности. "
        "Твой ответ должен быть максимально длинным и подробным.\n"
        f"{schema_description}\n"
        "Допустимые названия (ProductLine, Component, Supplier) — только из словаря:\n"
        f"{json.dumps(vocab, ensure_ascii=False, indent=2)}\n"
        f"{example}"
    )
    user = f"Текст чанка:\n\"\"\"\n{chunk.text}\n\"\"\"\nВерни JSON:"
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
        name_upper = name.upper().replace(" ", "-")
        trans = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
        norm = name_upper.translate(trans)
        
        valid_codes = _extract_valid_codes(chunk_text)
        if norm not in valid_codes:
            return None, f"код {name!r} отсутствует в тексте чанка"
            
        return Entity(type=etype, name=norm), None
    else:
        allowed_names = vocab.get(etype, [])
        normalized_name = (
            name.lower()
            .replace("ооо", "")
            .replace("«", "")
            .replace("»", "")
            .replace('"', "")
            .strip()
        )

        canon = None
        for allowed in allowed_names:
            normalized_allowed = (
                allowed.lower()
                .replace("ооо", "")
                .replace("«", "")
                .replace("»", "")
                .replace('"', "")
                .strip()
            )
            if normalized_allowed == normalized_name:
                canon = allowed
                break

        if canon is None:
            return None, f"название {name!r} не из словаря"

        return Entity(type=etype, name=canon), None


def _is_valid_response(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    if not isinstance(parsed.get("entities"), list) or not isinstance(parsed.get("relations"), list):
        return False
    for ent in parsed["entities"]:
        if not isinstance(ent, dict) or "type" not in ent or "name" not in ent:
            return False
    for rel in parsed["relations"]:
        if not isinstance(rel, dict):
            return False
        if not all(k in rel for k in ("subject", "subject_type", "type", "object", "object_type")):
            return False
    return True

def safe_json_loads(raw_text: str):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Пытаемся "починить" обрезанный JSON
        for suffix in ["}", "]}", "}}", "}]}"]:
            try:
                return json.loads(raw_text + suffix)
            except json.JSONDecodeError:
                continue
        # ВАЖНО: Если не смогли починить, вызываем ошибку!
        raise

def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    vocab = vocabulary()
    messages = _build_messages(chunk, vocab)

    raw = None
    for _attempt in range(2):
        try:
            resp = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
            print("\n=== СЫРОЙ ОТВЕТ ОТ НЕЙРОСЕТИ ===\n", resp.content, "\n=================================\n")
            
            # Иногда модели оборачивают JSON в теги ```json ... ```. Убираем их.
            content_clean = resp.content.strip()
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]
                
            parsed = safe_json_loads(content_clean.strip())

            # === НАЧАЛО УМНОГО ФИЛЬТРА ===
            if parsed and "relations" in parsed:
                # Словарь перевода английских галлюцинаций модели на русский
                EN_TO_RU = {
                    "compressor": "компрессор", "condenser_fan": "вентилятор конденсатора",
                    "temperature_sensor": "датчик температуры", "controller": "контроллер",
                    "pressure_relais": "реле давления", "pressure_relay": "реле давления",
                    "evaporator_fan": "вентилятор испарителя", "lighting_block": "блок освещения",
                    "dehumidifier": "осушитель", "thermostatic_valve": "терморегулирующий вентиль",
                    "start_relay": "пусковое реле", "flow_relay": "реле протока",
                    "circulation_pump": "циркуляционный насос"
                }

                for rel in parsed["relations"]:
                    # 1. Переводим английские сущности на русский
                    for key in ["subject", "object"]:
                        val = str(rel.get(key, "")).lower().strip()
                        if val in EN_TO_RU:
                            rel[key] = EN_TO_RU[val]
                            
                    # 2. Разворачиваем SUPPLIED_BY (чтобы всегда было Part -> Supplier)
                    if rel.get("type") == "SUPPLIED_BY" and rel.get("subject_type") == "Supplier":
                        rel["subject"], rel["object"] = rel["object"], rel["subject"]
                        rel["subject_type"], rel["object_type"] = rel["object_type"], rel["subject_type"]

                    # 3. Исправляем HAS_PART на INDICATES для кодов ошибок
                    if rel.get("type") == "HAS_PART" and rel.get("subject_type") == "ErrorCode":
                        rel["type"] = "INDICATES"
                        
                    # 4. Исправляем HAS_PART на BELONGS_TO для линеек
                    if (rel.get("type") == "HAS_PART" and 
                        rel.get("subject_type") == "ProductLine" and 
                        rel.get("object_type") == "ProductLine"):
                        rel["type"] = "BELONGS_TO"

                # Обновляем сущности в списке entities, чтобы они тоже стали русскими
                if "entities" in parsed:
                    for e in parsed["entities"]:
                        val = str(e.get("name", "")).lower().strip()
                        if val in EN_TO_RU:
                            e["name"] = EN_TO_RU[val]
            # === КОНЕЦ УМНОГО ФИЛЬТРА ===
            
            if _is_valid_response(parsed):
                raw = parsed
                break
            else:
                print("\n!!! ОШИБКА ВАЛИДАЦИИ: _is_valid_response вернул False !!!\n")
        except Exception as e:
            # ТЕПЕРЬ МЫ УВИДИМ ОШИБКУ
            print(f"\n!!! ОШИБКА ПРИ ЗАПРОСЕ ИЛИ ПАРСИНГЕ: {type(e).__name__}: {e} !!!\n")
        
        raw = None

    if raw is None:
        return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, dropped=1)

    dropped = 0
    entities: dict[str, Entity] = {}
    by_name: dict[str, Entity] = {}

    for ent_raw in raw["entities"]:
        ent, err = _validate_entity(ent_raw, chunk.text, vocab)
        if err:
            dropped += 1
            continue
        entities[ent.node_id] = ent
        # Сохраняем ключи строго в нижнем регистре
        by_name[str(ent_raw.get("name", "")).strip().lower()] = ent
        by_name[ent.name.lower()] = ent

    relations: list[Relation] = []
    for rel_raw in raw["relations"]:
        rtype = rel_raw.get("type")
        if rtype not in RELATION_SIGNATURES:
            dropped += 1
            continue
            
        subj_name = str(rel_raw.get("subject", "")).strip()
        obj_name = str(rel_raw.get("object", "")).strip()
        
        if subj_name.lower() in by_name:
            subj = by_name[subj_name.lower()] # Берем по нижнему регистру
        else:
            subj_type = rel_raw.get("subject_type")
            subj, err = _validate_entity({"type": subj_type, "name": subj_name}, chunk.text, vocab)
            if err:
                dropped += 1
                continue
            entities[subj.node_id] = subj
            by_name[subj_name.lower()] = subj
            
        if obj_name.lower() in by_name:
            obj = by_name[obj_name.lower()] # Берем по нижнему регистру
        else:
            obj_type = rel_raw.get("object_type")
            obj, err = _validate_entity({"type": obj_type, "name": obj_name}, chunk.text, vocab)
            if err:
                dropped += 1
                continue
            entities[obj.node_id] = obj
            by_name[obj_name.lower()] = obj
            
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