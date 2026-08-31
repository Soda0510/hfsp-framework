"""
Insight Archive (阶段5): cross-generation memory of "what works and why".

Each generation we ask the LLM to articulate WHY the current best configuration
performs well, store that insight together with the config, and feed the
accumulated insights back into the next generation's prompts (ReEvo idea).
The stored (config, insight, fitness) triples are also the raw material for the
DPO preference data (阶段6).
"""

from typing import Dict, List, Optional

from .representation import AlgorithmSpec
from .client import LLMClient
from .prompts import ARCHIVE_INSIGHT_SYSTEM, build_archive_prompt


class InsightArchive:
    def __init__(self, client: LLMClient, max_entries: int = 20, verbose: bool = True):
        self.client = client
        self.max_entries = max_entries
        self.verbose = verbose
        self.entries: List[Dict] = []   # {spec, fitness, insight}
        self.stats = {"calls": 0, "failed": 0}

    def record(self, spec: AlgorithmSpec, fitness: float, insight: str):
        self.entries.append({
            "spec": spec.to_dict(), "fitness": float(fitness), "insight": insight,
        })
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def ask_and_record(self, spec: AlgorithmSpec, fitness: float) -> Optional[str]:
        """Ask the LLM why this config works, then store the insight."""
        try:
            self.stats["calls"] += 1
            text = self.client.chat_text(ARCHIVE_INSIGHT_SYSTEM,
                                         build_archive_prompt(spec, fitness)).strip()
            if text:
                self.record(spec, fitness, text[:300])
                return text
        except Exception as e:
            self.stats["failed"] += 1
            if self.verbose:
                print(f"    [Archive] insight failed: {str(e)[:80]}")
        return None

    def summary(self, n: int = 5) -> str:
        """Concatenated recent insights — fed back into the mutation prompt."""
        if not self.entries:
            return ""
        return " ".join(e["insight"] for e in self.entries[-n:])

    def to_records(self) -> List[Dict]:
        """Export (config, fitness, insight) triples — DPO data source (阶段6)."""
        return list(self.entries)
