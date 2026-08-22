from app.transformer import (
    derive_base_model_id,
    derive_base_model_name,
    transform_model_catalog,
)


def test_derive_base_model_name_and_id():
    assert derive_base_model_name("gemini-2.5-flash-thinking") == "Gemini 2.5 Flash"
    assert derive_base_model_name("gemini-3.5-flash-extra-low") == "Gemini 3.5 Flash"
    assert derive_base_model_name("gemini-pro-agent") == "Gemini 3.1 Pro"
    assert derive_base_model_name("gemini-3.7-flash-tiered") == "Gemini 3.7 Flash"
    assert derive_base_model_name("gemini-3.7-flash-high") == "Gemini 3.7 Flash"
    assert derive_base_model_name("gemini-3-flash") == "Gemini 3.0 Flash"
    assert derive_base_model_name("gpt-oss-120b-medium") == "GPT OSS 120B"

    assert derive_base_model_id("gemini-3.7-flash-high") == "gemini-3.7-flash"
    assert derive_base_model_id("gemini-3.7-flash-low") == "gemini-3.7-flash"
    assert derive_base_model_id("gemini-3.5-flash-extra-low") == "gemini-3.5-flash"
    assert derive_base_model_id("gemini-pro-agent") == "gemini-3.1-pro"
    assert derive_base_model_id("claude-sonnet-4-6") == "claude-3.7-sonnet"
    assert derive_base_model_id("gpt-oss-120b-medium") == "gpt-oss-120b"


def test_transform_pipeline_categories_and_reasoning():
    sample_records = [
        # Pruned hidden models
        {
            "model_id": "chat_23310",
            "raw_name": "Internal Chat",
            "source_antigravity": True,
        },
        {
            "model_id": "chat_20706",
            "raw_name": "Internal Chat 2",
            "source_gemini_web": True,
        },
        {
            "model_id": "text-embedding-004",
            "raw_name": "Embedding",
            "source_gemini_api": True,
        },
        # Gemini 3.7 Flash sub-variants (should be consolidated into 1 clean model)
        {
            "model_id": "gemini-3.7-flash-high",
            "context_window": 1048576,
            "capabilities": ["thinking", "tools", "vision"],
            "source_antigravity": True,
        },
        {
            "model_id": "gemini-3.7-flash-medium",
            "context_window": 1048576,
            "capabilities": ["thinking", "tools", "vision"],
            "source_antigravity": True,
        },
        {
            "model_id": "gemini-3.7-flash-low",
            "context_window": 1048576,
            "capabilities": ["thinking", "tools", "vision"],
            "source_antigravity": True,
        },
        {
            "model_id": "gemini-3.7-flash",
            "context_window": 1048576,
            "capabilities": ["thinking", "tools", "vision"],
            "source_gemini_web": True,
        },
        # Gemini 3.5 Pro & Flash
        {
            "model_id": "gemini-3.5-pro",
            "context_window": 2097152,
            "capabilities": ["tools", "vision"],
            "source_antigravity": True,
        },
        {
            "model_id": "gemini-3.5-flash-extra-low",
            "context_window": 1048576,
            "capabilities": ["tools", "vision"],
            "source_antigravity": True,
        },
        # Gemini 3.1 Pro (pro-agent)
        {
            "model_id": "gemini-pro-agent",
            "context_window": 1048576,
            "capabilities": ["tools"],
            "source_antigravity": True,
        },
        # Gemini 2.5 Flash Thinking
        {
            "model_id": "gemini-2.5-flash-thinking",
            "context_window": 1048576,
            "capabilities": ["thinking", "tools", "vision"],
            "source_antigravity": True,
        },
        # Non-Google Models
        {
            "model_id": "claude-sonnet-4-6",
            "context_window": 200000,
            "capabilities": ["tools", "vision"],
            "source_antigravity": True,
        },
        {
            "model_id": "claude-opus-4-6-thinking",
            "context_window": 200000,
            "capabilities": ["thinking", "tools", "vision"],
            "source_antigravity": True,
        },
        {
            "model_id": "gpt-oss-120b-medium",
            "context_window": 32768,
            "capabilities": ["tools"],
            "source_antigravity": True,
        },
    ]

    result = transform_model_catalog(sample_records)
    gemini_models = result["gemini_models"]
    non_google_models = result["non_google_models"]

    # 1. Gemini Models Verification
    gemini_names = [m["base_model_name"] for m in gemini_models]
    assert gemini_names == [
        "Gemini 3.7 Flash",
        "Gemini 3.5 Pro",
        "Gemini 3.5 Flash",
        "Gemini 3.1 Pro",
        "Gemini 2.5 Flash",
    ]

    # Check 3.7 Flash reasoning levels & consolidation
    g37 = next(m for m in gemini_models if m["base_model_name"] == "Gemini 3.7 Flash")
    assert g37["id"] == "gemini-3.7-flash"
    assert set(g37["reasoning_levels"]) == {"Low", "Medium", "High"}
    assert "gemini-3.7-flash-high" in g37["selectable_model_ids"]
    assert "gemini-3.7-flash-low" in g37["selectable_model_ids"]
    assert "Gemini API" in g37["available_sources"]  # Forced override for 3.7 flash
    assert "Antigravity" in g37["available_sources"]
    assert "Gemini Web" in g37["available_sources"]

    # Check 3.5 Flash extra-low tier extraction
    g35f = next(m for m in gemini_models if m["base_model_name"] == "Gemini 3.5 Flash")
    assert "Extra Low" in g35f["reasoning_levels"]

    # 2. Non-Google Models Verification
    non_google_ids = [m["id"] for m in non_google_models]
    # Claude models must come first, GPT models second
    assert non_google_ids == [
        "claude-3.7-sonnet",
        "claude-3-opus",
        "gpt-oss-120b",
    ]

    # Check Claude Opus thinking tier
    opus = next(m for m in non_google_models if m["id"] == "claude-3-opus")
    assert (
        "Dynamic" in opus["reasoning_levels"] or "Thinking" in opus["reasoning_levels"]
    )


def test_lite_models_visible_and_experimental_models_hidden():
    sample_records = [
        {
            "model_id": "gemini-2.0-flash-lite",
            "context_window": 1048576,
            "capabilities": ["tools", "vision"],
            "source_gemini_api": True,
        },
        {
            "model_id": "gemini-2.5-flash-lite",
            "context_window": 1048576,
            "capabilities": ["thinking", "tools", "vision"],
            "source_gemini_api": True,
        },
        {
            "model_id": "gemini-2.0-pro-exp-02-05",
            "context_window": 2097152,
            "capabilities": ["thinking", "tools", "vision"],
            "source_gemini_api": True,
        },
        {
            "model_id": "gemini-2.0-flash-thinking-exp-01-21",
            "context_window": 1048576,
            "capabilities": ["thinking", "tools", "vision"],
            "source_gemini_api": True,
        },
    ]

    result = transform_model_catalog(sample_records)
    gemini_models = result["gemini_models"]

    # Lite models must be visible (hidden == False)
    g20_lite = next(
        m for m in gemini_models if "lite" in m["id"].lower() and "2.0" in m["id"]
    )
    assert g20_lite["hidden"] is False

    g25_lite = next(
        m for m in gemini_models if "lite" in m["id"].lower() and "2.5" in m["id"]
    )
    assert g25_lite["hidden"] is False

    # Experimental models must be hidden by default (hidden == True)
    g20_pro_exp = next(
        m for m in gemini_models if "exp" in m["id"].lower() and "pro" in m["id"]
    )
    assert g20_pro_exp["hidden"] is True

    g20_flash_exp = next(
        m for m in gemini_models if "exp" in m["id"].lower() and "flash" in m["id"]
    )
    assert g20_flash_exp["hidden"] is True
