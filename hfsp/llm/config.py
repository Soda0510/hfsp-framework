"""Configuration for the local LLM backend."""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Settings for the local LLM (defaults target a local Ollama instance)."""

    model: str = "qwen2.5:14b"           # Ollama model tag
    base_url: str = "http://localhost:11434"  # Ollama REST API base
    temperature: float = 0.8             # sampling temperature for mutation
    max_retries: int = 3                 # retries for LLM mutation
    timeout: int = 120                   # per-request timeout (seconds)
    verbose: bool = True                 # print LLM attempts / fallbacks
