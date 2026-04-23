"""SearchNode: solution tree node (code, execution, evaluation, search metadata).

Stripped down from MLEvolve/engine/search_node.py — only attributes and
methods used by treevee.py and tree_search.py are kept.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import DataClassJsonMixin
from treevee.utils.metric import MetricValue

logger = logging.getLogger("treevee")


@dataclass(eq=False)
class SearchNode(DataClassJsonMixin):
    """Solution tree node: code, execution, evaluation, and search metadata."""

    # ---- code & plan ----
    code: str
    plan: Optional[str] = field(default=None, kw_only=True)

    # ---- general attrs ----
    step: Optional[int] = field(default=None, kw_only=True)
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["SearchNode"] = field(default=None, kw_only=True)
    children: set["SearchNode"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    _term_out: Optional[list[str]] = field(default=None, kw_only=True)
    exec_time: Optional[float] = field(default=None, kw_only=True)

    # ---- evaluation ----
    metric: Optional[MetricValue] = field(default=None, kw_only=True)

    # ---- search / MCTS ----
    stage: str  # "root", "improve", "debug", etc.
    visits: int = field(default=0, kw_only=True)
    total_reward: float = field(default=0.0, kw_only=True)
    branch_id: Optional[int] = field(default=None, kw_only=True)
    eval_output: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def num_children(self):
        return len(self.children)

    def __eq__(self, other):
        return isinstance(other, SearchNode) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


# ---------------------------------------------------------------------------
# Journal — ordered collection of SearchNodes forming the solution tree
# ---------------------------------------------------------------------------

@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[SearchNode] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.nodes)

    def append(self, node: SearchNode) -> None:
        if node.step is None:
            node.step = len(self.nodes)
        self.nodes.append(node)
