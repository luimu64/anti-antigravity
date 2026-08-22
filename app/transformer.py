"""
Model Data Transformation Pipeline
Implements normalization, consolidation of sub-variants, reasoning levels extraction,
partitioning into Gemini vs Non-Google models, and separate category sorting.
"""

import re
from typing import Any

from app.config import CANONICAL_MODEL_MAP


def derive_base_model_id(model_id: str) -> str:
    """
    Derive clean canonical base model ID by removing sub-variant suffixes
    (e.g., -high, -medium, -low, -extra-low, -tiered, -agent, -thinking).
    """
    s = model_id.strip().lower().replace("models/", "")

    if s in CANONICAL_MODEL_MAP:
        return CANONICAL_MODEL_MAP[s]

    if "pro-agent" in s:
        return "gemini-3.1-pro"
    if "3-flash-agent" in s:
        return "gemini-3.5-flash"

    # Remove reasoning/tier suffixes
    cleaned = re.sub(r"-(extra-low|tiered|thinking|agent|high|medium|low)$", "", s)
    return cleaned


def derive_base_model_name(model_id: str) -> str:
    """
    Derive base model display name programmatically from model_id.
    - Capitalize provider and tier. Replace hyphens with spaces.
    - Convert version formats (e.g., '3' to '3.0', 'pro-agent' to '3.1 Pro').
    - Strip reasoning/performance tiers: Tiered, Low, Extra Low, Medium, Thinking, Agent.
    """
    s = model_id.strip()

    if "pro-agent" in s:
        s = s.replace("pro-agent", "3.1-pro")

    # Convert standalone single digit version after provider/dash
    s = re.sub(r"(?<=-)(\d)(?=-|$)", r"\1.0", s)

    parts = s.split("-")
    acronyms = {"gpt": "GPT", "oss": "OSS", "ai": "AI"}
    capitalized_parts = [acronyms.get(p.lower(), p.capitalize()) for p in parts]
    name = " ".join(capitalized_parts)

    name = re.sub(r"\b(\d+)b\b", r"\1B", name, flags=re.IGNORECASE)

    strip_terms = [
        r"\bExtra Low\b",
        r"\bTiered\b",
        r"\bLow\b",
        r"\bMedium\b",
        r"\bThinking\b",
        r"\bAgent\b",
        r"\bHigh\b",
    ]
    for term in strip_terms:
        name = re.sub(term, "", name, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", name).strip()


def extract_reasoning_levels(
    base_model_id: str,
    underlying_ids: list[str],
    capabilities: list[str],
) -> list[str]:
    """
    Extract supported reasoning levels for a model (e.g. ['Low', 'Medium', 'High'] or ['Dynamic'] or []).
    """
    levels: list[str] = []
    ids_str = " ".join(underlying_ids + [base_model_id]).lower()

    if "extra-low" in ids_str:
        levels.append("Extra Low")
    if ("-low" in ids_str or " low" in ids_str) and "Low" not in levels:
        levels.append("Low")
    if "-medium" in ids_str or " medium" in ids_str:
        levels.append("Medium")
    if "-high" in ids_str or " high" in ids_str:
        levels.append("High")
    if "tiered" in ids_str:
        for lvl in ["Low", "Medium", "High"]:
            if lvl not in levels:
                levels.append(lvl)

    # If 3.7 or thinking capability but no discrete sub-variant IDs observed
    if not levels and (
        "thinking" in capabilities
        or "3.7" in base_model_id
        or "3.6" in base_model_id
        or "thinking" in ids_str
        or "opus" in ids_str
    ):
        if "3.7" in base_model_id or "3.6" in base_model_id or "3.1" in base_model_id:
            levels = ["Low", "Medium", "High"]
        else:
            levels = ["Dynamic"]

    return levels


def extract_version(name_or_id: str) -> tuple[float, ...]:
    """Extract semantic version numbers for sorting (e.g., 3.7 -> (3.7,), 1.5 -> (1.5,))."""
    match = re.search(r"(\d+(?:\.\d+)?)", name_or_id)
    if match:
        try:
            return (float(match.group(1)),)
        except ValueError:
            pass
    return (0.0,)


def get_gemini_class_rank(name: str) -> int:
    """
    Class hierarchy rank for Gemini models within the same version:
    Pro (0) > Flash (1) > Flash Lite / Flash Image (2) > Others (3)
    """
    name_lower = name.lower()
    if "pro" in name_lower:
        return 0
    if (
        "flash lite" in name_lower
        or "flash-lite" in name_lower
        or "lite" in name_lower
        or "flash image" in name_lower
        or "image" in name_lower
    ):
        return 2
    if "flash" in name_lower:
        return 1
    return 3


def get_rest_provider_rank(item: dict[str, Any]) -> tuple[int, float, str]:
    """
    Provider priority for Non-Google Models:
    Claude models (0) > GPT models (1) > Others (2), sorted by newness/version descending.
    """
    name_lower = (item.get("base_model_name") or "").lower()
    ids_str = " ".join(item.get("selectable_model_ids", [])).lower()
    combined = f"{name_lower} {ids_str}"
    version = extract_version(item.get("base_model_name", ""))[0]

    if "claude" in combined:
        return (0, -version, name_lower)
    if "gpt" in combined:
        return (1, -version, name_lower)
    return (2, -version, name_lower)


def transform_model_catalog(
    raw_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Executes normalization, hidden model pruning, sub-variant consolidation,
    reasoning levels extraction, and partitioning into Gemini Models vs Non-Google Models.
    """
    # -------------------------------------------------------------
    # Phase 1 & 2: Ingestion, Normalization & Hidden Pruning
    # -------------------------------------------------------------
    pruned_ids = {"chat_23310", "chat_20706", "text-embedding-004"}
    normalized = []

    for r in raw_records:
        model_id = r.get("model_id") or r.get("id") or ""
        clean_id = model_id.replace("models/", "")
        if clean_id in pruned_ids or model_id in pruned_ids:
            continue

        raw_name = r.get("raw_name") or r.get("displayName") or model_id
        context_window = r.get("context_window") or r.get("maxTokens") or 0
        capabilities = list(r.get("capabilities", []))

        source_antigravity = bool(r.get("source_antigravity", False))
        source_gemini_api = bool(r.get("source_gemini_api", False))
        source_gemini_web = bool(r.get("source_gemini_web", False))

        available_sources: list[str] = []
        if source_antigravity:
            available_sources.append("Antigravity")
        if source_gemini_api:
            available_sources.append("Gemini API")
        if source_gemini_web:
            available_sources.append("Gemini Web")

        if "available_sources" in r and isinstance(r["available_sources"], list):
            for src in r["available_sources"]:
                if src not in available_sources:
                    available_sources.append(src)

        if "gemini-3.7-flash" in clean_id and "Gemini API" not in available_sources:
            available_sources.append("Gemini API")

        base_id = derive_base_model_id(clean_id)
        base_name = derive_base_model_name(base_id)

        normalized.append(
            {
                "raw_name": raw_name,
                "model_id": clean_id,
                "base_id": base_id,
                "base_model_name": base_name,
                "context_window": context_window,
                "capabilities": capabilities,
                "available_sources": available_sources,
            }
        )

    # -------------------------------------------------------------
    # Phase 3 & 4: Grouping & Merging (Consolidating Tiers under Base Model)
    # -------------------------------------------------------------
    grouped: dict[str, dict[str, Any]] = {}

    for r in normalized:
        base_name = r["base_model_name"]
        if base_name not in grouped:
            grouped[base_name] = {
                "id": r["base_id"],
                "base_model_name": base_name,
                "selectable_model_ids": [r["model_id"]],
                "context_window": r["context_window"],
                "capabilities": list(r["capabilities"]),
                "available_sources": list(r["available_sources"]),
            }
        else:
            entry = grouped[base_name]
            if r["model_id"] not in entry["selectable_model_ids"]:
                entry["selectable_model_ids"].append(r["model_id"])
            if r["context_window"] > entry["context_window"]:
                entry["context_window"] = r["context_window"]
            for cap in r["capabilities"]:
                if cap not in entry["capabilities"]:
                    entry["capabilities"].append(cap)
            for src in r["available_sources"]:
                if src not in entry["available_sources"]:
                    entry["available_sources"].append(src)

    # Attach reasoning levels and metadata
    merged_records = []
    for entry in grouped.values():
        reasoning_levels = extract_reasoning_levels(
            entry["id"],
            entry["selectable_model_ids"],
            entry["capabilities"],
        )
        entry["reasoning_levels"] = reasoning_levels
        entry["supportsThinking"] = bool(
            reasoning_levels or "thinking" in entry["capabilities"]
        )
        entry["supportsTools"] = (
            "tools" in entry["capabilities"]
            or "embeddings" not in entry["capabilities"]
        )
        entry["supportsVision"] = (
            "vision" in entry["capabilities"]
            or "embeddings" not in entry["capabilities"]
        )
        entry["isEmbedding"] = "embeddings" in entry["capabilities"]
        entry["displayName"] = entry["base_model_name"]
        entry["maxTokens"] = entry["context_window"]
        entry["providers"] = [
            s.lower().replace(" ", "_") for s in entry.get("available_sources", [])
        ]
        entry["provider_count"] = len(entry["providers"])
        # Determine if model is experimental (hidden by default) vs standard / lite (shown by default)
        model_id_str = (
            f"{entry['id']} {' '.join(entry['selectable_model_ids'])} "
            f"{entry['base_model_name']}"
        ).lower()

        is_experimental = bool(
            "-exp" in model_id_str
            or "experimental" in model_id_str
            or "preview" in model_id_str
            or "tab_" in model_id_str
        )
        entry["hidden"] = is_experimental
        entry["is_experimental"] = is_experimental
        merged_records.append(entry)

    # -------------------------------------------------------------
    # Phase 5: Partitioning into Gemini vs Non-Google (The Rest)
    # -------------------------------------------------------------
    gemini_models: list[dict[str, Any]] = []
    non_google_models: list[dict[str, Any]] = []

    for item in merged_records:
        is_gemini = (
            "gemini" in item["id"].lower()
            or "gemini" in item["base_model_name"].lower()
        )
        if is_gemini:
            gemini_models.append(item)
        else:
            non_google_models.append(item)

    # -------------------------------------------------------------
    # Phase 6: Sorting & Ordering
    # -------------------------------------------------------------
    # Gemini Models: Semantic version descending, then Pro > Flash > Lite
    gemini_models.sort(
        key=lambda item: (
            -extract_version(item["base_model_name"])[0],
            get_gemini_class_rank(item["base_model_name"]),
            item["base_model_name"],
        )
    )

    # Non-Google Models: Claude first, GPT second, sorted by version descending
    non_google_models.sort(key=get_rest_provider_rank)

    return {
        "gemini_models": gemini_models,
        "the_rest": non_google_models,
        "non_google_models": non_google_models,
    }
