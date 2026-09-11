"""Sanity checks for sanitize_schema_for_gemini against the incident payload shape."""

import json

from app.translator import sanitize_schema_for_gemini

# Exact shape from the incident: anyOf branches with "const" (properties[2])
schema = {
    "type": "object",
    "properties": {
        "mode": {
            "description": "How to respond",
            "anyOf": [
                {"const": "always_allow", "title": "Always allow"},
                {"const": "ask_every_time", "title": "Ask every time"},
                {"const": "never_allow", "title": "Never allow"},
                {"const": "remember_choice", "title": "Remember choice"},
            ],
        },
        "path": {"type": ["string", "null"], "description": "File path"},
        "count": {"type": "integer", "maximum": 10},
    },
    "required": ["mode"],
}
out = sanitize_schema_for_gemini(schema)
print(json.dumps(out, indent=2))

FORBIDDEN = {
    "const",
    "oneOf",
    "allOf",
    "$ref",
    "$defs",
    "additionalProperties",
    "pattern",
    "not",
    "examples",
    "title",
}


def walk(n):
    if isinstance(n, dict):
        bad = FORBIDDEN & n.keys()
        assert not bad, f"forbidden keys survived: {bad}"
        for v in n.values():
            walk(v)
    elif isinstance(n, list):
        for v in n:
            walk(v)


walk(out)
mode = out["properties"]["mode"]
assert [b["enum"][0] for b in mode["anyOf"]] == [
    "always_allow",
    "ask_every_time",
    "never_allow",
    "remember_choice",
]
assert all("const" not in b and "title" not in b for b in mode["anyOf"])
assert mode["description"] == "How to respond"
assert out["properties"]["path"]["type"] == "string"
assert out["properties"]["path"]["nullable"] is True
print("OK: incident schema sanitized, forbidden keys gone")

# $ref resolution + oneOf + allOf + non-string const + mixed enum + bad format
schema2 = {
    "$defs": {"Amount": {"type": "number", "minimum": 0}},
    "type": "object",
    "properties": {
        "amount": {"$ref": "#/$defs/Amount"},
        "kind": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        "merged": {
            "allOf": [
                {"type": "object"},
                {"properties": {"x": {"type": "boolean"}}},
                {"required": ["x"]},
            ]
        },
        "flag": {"const": True},
        "mixed": {"enum": ["a", 1, "b"]},
        "fmt": {"type": "string", "format": "uri"},
    },
}
out2 = sanitize_schema_for_gemini(schema2)
walk(out2)
assert out2["properties"]["amount"] == {"type": "number", "minimum": 0}
assert out2["properties"]["kind"]["anyOf"] == [{"type": "string"}, {"type": "integer"}]
assert out2["properties"]["merged"]["required"] == ["x"]
assert "x" in out2["properties"]["merged"]["properties"]
assert "Must be exactly: true" in out2["properties"]["flag"]["description"]
assert "Allowed values" in out2["properties"]["mixed"]["description"]
assert "format" not in out2["properties"]["fmt"]
print("OK: refs/oneOf/allOf/non-string const/mixed enum/format")

# Passthrough of already-clean OpenAPI 3.0 schema
clean = {
    "type": "object",
    "properties": {"q": {"type": "string", "description": "query"}},
    "required": ["q"],
}
assert sanitize_schema_for_gemini(clean) == clean
print("OK: clean schema passthrough unchanged")
