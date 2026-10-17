"""Pydantic model → JSON schema that satisfies OpenAI's *strict* mode (function tools and structured outputs).

Strict mode uses constrained decoding: the model can only emit tokens that keep the output valid against the schema,
so the JSON always parses and always has the right shape. In exchange the schema must be a restricted subset:
* every object has `"additionalProperties": false` and lists ALL its properties in `required`;
* optional values are a union with null (`X | None` → `anyOf: [X, {"type": "null"}]`), never an omitted key;
* no `default`s (the model must produce every field), and a `$ref` may not carry sibling keywords (we inline it).

Value constraints (string length, number ranges) are better enforced by pydantic validators after parsing: the
schema shapes the output, the validator checks the semantics. Both run, which is defence in depth.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

_DROP_KEYS = frozenset({"default", "title", "examples"})
_NAME_MAPS = frozenset({"properties", "$defs", "definitions"})


def _resolve(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported $ref {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(f"$ref {ref!r} does not resolve to an object schema")
    return node


def _strictify(node: Any, root: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_strictify(v, root) for v in node]
    if not isinstance(node, dict):
        return node
    node = {k: v for k, v in node.items() if k not in _DROP_KEYS}
    if "$ref" in node and len(node) > 1:  # e.g. {"$ref": ..., "description": ...}: inline the target
        target = copy.deepcopy(_resolve(root, node.pop("$ref")))
        node = {**target, **node}
    if "allOf" in node and len(node["allOf"]) == 1:
        only = node.pop("allOf")[0]
        node = {**copy.deepcopy(_resolve(root, only["$ref"]) if "$ref" in only else only), **node}
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROP_KEYS:
            continue
        if key in _NAME_MAPS and isinstance(value, dict):  # keys here are field/definition NAMES, not keywords
            out[key] = {name: _strictify(sub, root) for name, sub in value.items()}
        else:
            out[key] = _strictify(value, root)
    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out.get("properties", {}).keys())
    return out


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The strict-mode schema of a pydantic model (shared definitions stay in `$defs`, which strict mode supports)."""
    raw = model.model_json_schema()
    result = _strictify(raw, raw)
    if not isinstance(result, dict):  # pragma: no cover
        raise TypeError("schema root must be an object")
    return result
