"""Taxonomy tree management.

Handles nested dict/YAML/JSON taxonomies, flattens to leaf paths,
and renders prompts for LLM classification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SEPARATOR = " > "


@dataclass
class Taxonomy:
    """A classification taxonomy that can be nested or flat.

    Accepts either:
        - A nested dict: {"Food": {"Burgers": ["Hamburger", "Cheeseburger"]}}
        - A flat list of paths: ["Food > Burgers > Hamburger", ...]
        - A path to a JSON/YAML file containing either format
    """

    tree: dict[str, Any] = field(default_factory=dict)
    _leaf_paths: list[str] = field(default_factory=list, init=False, repr=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Taxonomy:
        t = cls(tree=d)
        t._leaf_paths = _flatten(d)
        return t

    @classmethod
    def from_flat(cls, paths: list[str], separator: str = SEPARATOR) -> Taxonomy:
        tree = _unflatten(paths, separator)
        t = cls(tree=tree)
        t._leaf_paths = paths
        return t

    @classmethod
    def from_file(cls, path: str | Path) -> Taxonomy:
        p = Path(path)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(text)
            except ImportError:
                raise ImportError("PyYAML required for YAML taxonomy files: pip install pyyaml")
        else:
            data = json.loads(text)

        if isinstance(data, list):
            return cls.from_flat(data)
        return cls.from_dict(data)

    @property
    def leaf_paths(self) -> list[str]:
        if not self._leaf_paths:
            self._leaf_paths = _flatten(self.tree)
        return self._leaf_paths

    @property
    def leaf_labels(self) -> list[str]:
        """Just the leaf node names (last segment of each path)."""
        return [p.split(SEPARATOR)[-1] for p in self.leaf_paths]

    def depth(self) -> int:
        return max(len(p.split(SEPARATOR)) for p in self.leaf_paths) if self.leaf_paths else 0

    def top_level_categories(self) -> list[str]:
        return list(self.tree.keys())

    def subtree(self, category: str) -> Taxonomy:
        """Get taxonomy for a single top-level category."""
        if category not in self.tree:
            raise KeyError(f"Category '{category}' not in taxonomy")
        sub = self.tree[category]
        if isinstance(sub, dict):
            return Taxonomy.from_dict(sub)
        if isinstance(sub, list):
            return Taxonomy.from_flat(sub)
        return Taxonomy.from_flat([str(sub)])

    def render_for_prompt(self, indent: int = 0) -> str:
        """Render taxonomy as indented tree for LLM system prompts."""
        return _render_tree(self.tree, indent)

    def render_flat_for_prompt(self) -> str:
        """Render as numbered flat list for simpler taxonomies."""
        return "\n".join(f"{i+1}. {p}" for i, p in enumerate(self.leaf_paths))

    def to_dict(self) -> dict[str, Any]:
        return self.tree

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(self.tree, indent=2)
        if path:
            Path(path).write_text(s)
        return s


def _flatten(d: dict[str, Any], prefix: str = "") -> list[str]:
    """Recursively flatten nested dict to list of leaf paths."""
    paths = []
    for key, value in d.items():
        current = f"{prefix}{SEPARATOR}{key}" if prefix else key
        if isinstance(value, dict):
            paths.extend(_flatten(value, current))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    paths.extend(_flatten(item, current))
                else:
                    paths.append(f"{current}{SEPARATOR}{item}")
        else:
            # Leaf node with a single value
            paths.append(current)
    return paths


def _unflatten(paths: list[str], separator: str = SEPARATOR) -> dict[str, Any]:
    """Convert flat path list back to nested dict."""
    tree: dict[str, Any] = {}
    for path in paths:
        parts = [p.strip() for p in path.split(separator)]
        node = tree
        for i, part in enumerate(parts[:-1]):
            if part not in node:
                node[part] = {}
            elif not isinstance(node[part], dict):
                node[part] = {"_leaf": node[part]}
            node = node[part]
        leaf = parts[-1]
        node[leaf] = leaf
    return tree


def _render_tree(d: dict[str, Any], indent: int = 0) -> str:
    """Render nested dict as indented tree string."""
    lines = []
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}- {key}")
            lines.append(_render_tree(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}- {key}")
            for item in value:
                if isinstance(item, dict):
                    lines.append(_render_tree(item, indent + 1))
                else:
                    lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}- {key}")
    return "\n".join(lines)
