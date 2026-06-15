from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from legal_ai.config.schemas import AgentGraphSettings, DataSources, Settings


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return data


def load_settings(path: str | Path = "configs/settings.yaml") -> Settings:
    return Settings(**load_yaml(path))


def load_data_sources(path: str | Path = "configs/data_sources.yaml") -> DataSources:
    return DataSources(**load_yaml(path))


def load_prompts(path: str | Path = "configs/prompts.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_agent_graph(path: str | Path = "configs/agent_graph.yaml") -> AgentGraphSettings:
    return AgentGraphSettings(**load_yaml(path))
