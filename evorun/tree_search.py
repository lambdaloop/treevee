"""Tree search layer for simple_run.py.

Wraps MLEvolve's existing SearchNode, Journal, and MetricValue to provide
Monte Carlo tree search with UCT node selection and backpropagation.

The tree allows simple_run to explore multiple branches of code improvement
in parallel, instead of a linear chain. Each expansion saves a snapshot,
and broken code only kills that branch while others continue.

Key improvements over the original implementation:
  - UCT selection across ALL nodes in the tree (not just root children)
  - Reward normalization via percentile rank (score-invariant tree dynamics)
  - Per-branch expansion tracking for sparse-branch exploration bonus
  - eval_output storage on nodes so the LLM can see previous errors
  - Bug-aware exploration: broken nodes (metric.value is None) get a
    secondary boost when re-expanded so the search can retry debugging
  - Full state serialization for resume across runs
"""

import bisect
import logging
import math
from typing import Optional, TYPE_CHECKING

from evorun.engine.search_node import Journal, SearchNode
from evorun.utils.metric import MetricValue

logger = logging.getLogger("simple_run")

# Maximum number of child expansions per non-root node before it's
# considered "fully expanded" (for stopping the search on that branch).
_DEFAULT_MAX_CHILDREN = 10

# UCT -- "Upper Confidence Bound applied to Trees", exploration constant.
# Controls the exploration vs. exploitation trade-off:
#   Higher value -> more exploration (trying out less-visited nodes)
#   Lower value  -> more exploitation (focusing on nodes with high rewards)
# Default value chosen to keep exploration small so score ranking remains
# the primary driver of node selection.  Root gets extra exploration
# proportional to remaining child capacity so new branches are spawned early.
_DEFAULT_EXPLORE_C = 0.15

# Sparsity bonus -- extra UCT boost given to nodes under their branch's
# average visit count relative to the root.  This prevents the tree from
# collapsing onto 1-2 dominant branches.
_SPARSITY_BONUS = 0.2


class TreeSearch:
    """Lightweight MCTS for iterative LLM-driven optimization.

    Wraps SearchNode + Journal to provide:
        - UCT-based node selection across ALL tree nodes (not just root children)
        - Reward normalization via percentile rank (score-invariant, bounded [0,1])
        - Per-branch expansion tracking with sparsity bonus for under-explored branches
        - Best-node tracking for early termination
        - Full state serialization / deserialization for resume support

    UCT selection strategy:
        1.  Find all expandable nodes across the entire tree
        2.  Score each candidate with UCT = Q + c * sqrt(ln(parent_visits) / visits)
        3.  Add sparsity bonus for nodes in branches with fewer total expansions
        4.  Add bug-aware bonus for nodes that previously broke (metric.value is None)
        5.  Pick the highest-scoring node

    Reward normalization:
        - Converts raw scores to percentile rank in [0, 1] via sorted list
        - maximize=True:  higher raw score -> higher rank -> higher reward
        - maximize=False: lower raw score -> higher rank -> higher reward (reversed)
        - None (broken) scores -> 0.0 reward  (no penalty for exploratory failures)
    """

    def __init__(
        self, maximize: bool = True, max_children: int = _DEFAULT_MAX_CHILDREN,
        explore_c: float | None = None
    ):
        self.maximize = maximize
        self.max_children = max_children
        self.explore_c = explore_c if explore_c is not None else _DEFAULT_EXPLORE_C
        self.sparsity_bonus = _SPARSITY_BONUS

        # Core search structures (reused from MLEvolve)
        self.journal = Journal()
        # Root node has no parent, empty code/plan, stage="root"
        self.root = SearchNode(code="", plan="(root)", stage="root")
        self.journal.append(self.root)

        self._best_node: Optional[SearchNode] = None
        self._best_score: float = float("-inf") if self.maximize else float("inf")

        # Reward normalization (percentile rank)
        self._reward_count = 0
        self._scores: list[float] = []  #(sorted list of all seen raw scores

        # Per-branch expansion tracking (branch = root-child subtree)
        self._branch_expansions: dict[str, int] = {}  # branch_id -> total expansions
        self._next_branch_id = 1

    # ------------------------------------------------------------------
    # Scoring normalization
    # ------------------------------------------------------------------

    def _normalize_reward(self, raw_score: float | None) -> float:
        """Convert raw score to normalized reward using percentile rank.

        - None / NaN (broken/unparseable) -> 0.0 (exploration bonus applied in _score_candidate)
        - First score   -> 0.5  (neutral, no history to rank against)
        - Subsequent scores -> percentile rank in [0, 1]:
            maximize=True:  higher raw score -> higher rank -> higher reward
            maximize=False: lower raw score -> higher rank -> higher reward

        This preserves relative ordering between scores and provides
        rewards in a predictable [0, 1] range for stable UCT computation.
        """
        if raw_score is None or math.isnan(raw_score):
            return 0.0
        self._reward_count += 1
        bisect.insort(self._scores, raw_score)
        if len(self._scores) <= 1:
            return 0.5  # neutral: first score
        idx = bisect.bisect_left(self._scores, raw_score)
        rank = idx / (len(self._scores) - 1)
        return rank if self.maximize else 1.0 - rank

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_node(self, branch_id: int | None = None) -> SearchNode:
        """Select the best node to expand using UCT across the ENTIRE tree.

        Selection algorithm:
            1.  Collect all expandable nodes across the full tree (not just root children)
            2.  Score each candidate with UCT = Q + c * sqrt(ln(parent_visits + 1) / visits)
            3.  Add sparsity bonus for nodes in branches with fewer total expansions
            4.  Pick the highest-scoring node

        If the root has no children yet, returns the root (first expansion must
        always be from root).

        Args:
            branch_id:  Optional.  When creating a new root child, passes this
                branch_id for sparsity tracking.  Also marks unvisited nodes
                with a branch_id on first expansion.

        Returns:
            The SearchNode to expand next.
        """
        if not self.root.children:
            return self.root

        # Collect all expandable nodes across the tree.
        expandable_nodes = [
            node for node in self.journal.nodes
            if node.num_children < self.max_children
        ]

        if not expandable_nodes:
            # No node can expand more -- pick highest-Q node overall.
            return max(
                self.journal.nodes,
                key=lambda n: n.total_reward / max(n.visits, 1),
            )

        # Score each candidate.
        scored_candidates: list[tuple[float, SearchNode]] = []
        for node in expandable_nodes:
            score = self._score_candidate(node, branch_id)
            scored_candidates.append((score, node))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        _best_score, best_node = scored_candidates[0]

        is_first_expansion = best_node.num_children == 0
        node_label = "unvisited" if is_first_expansion else "visited"
        raw = best_node.metric.value if best_node.metric and best_node.metric.value is not None else "N/A"
        logger.info(
            f"[Tree] Selected node {best_node.id[:8]} ({node_label}, "
            f"UCT={_best_score:.4f}, raw={raw}, "
            f"Q={best_node.total_reward / max(best_node.visits, 1):.4f}, "
            f"children={best_node.num_children}, depth={self._node_depth(best_node)})"
        )

        # Assign branch_id for nodes that don't have one yet (unvisited first expansions).
        if branch_id is not None and not best_node.branch_id:
            best_node.branch_id = branch_id
            if branch_id not in self._branch_expansions:
                self._branch_expansions[branch_id] = 0

        return best_node

    def _score_candidate(self, node: SearchNode, branch_id: int | None = None) -> float:
        """Score a candidate node for selection using UCT + sparsity bonus.

        All nodes are scored the same way: mean reward + depth-weighted
        exploration bonus, with sparsity bonus for under-explored branches.
        """
        mean_reward: float = node.total_reward / max(node.visits, 1)

        # Standard UCT exploration, depth-weighted.
        depth_of_node: int = self._node_depth(node)
        # Shallow nodes get more exploration: root-child=1.8, depth2=1.3, depth3+=0.8
        depth_weight = max(2.3 - 0.3 * depth_of_node, 0.5)
        parent_visits: int = node.parent.visits if node.parent else 1
        exploration: float = self.explore_c * math.sqrt(
            math.log(max(parent_visits, 1) + 1) / max(node.visits, 1)
        ) * depth_weight

        # Cap exploration for visited nodes (visits > 1 means expanded at least once).
        if node.visits > 1:
            exploration = min(exploration, self.explore_c * 0.8)

        # Broken node bonus — decays with visits so the node gets tried
        # early but doesn't dominate long-term selection.
        if not node.metric or (node.metric is not None and node.metric.value is None):
            exploration += self.explore_c * 2.0 / max(node.visits, 1)

        uct: float = mean_reward + exploration

        # Sparsity bonus for under-explored branches (root children only).
        if node.parent and node.parent.parent is None:
            node_id: str = node.id
            branch_expansions: int = self._branch_expansions.get(node_id, 0)
            total: int = self._total_branch_expansions()
            if total > 0:
                avg_per_branch: float = total / max(len(self.root.children), 1)
                diff: float = avg_per_branch - branch_expansions
                if diff > 0:
                    uct += self.sparsity_bonus * diff

        return uct

    # ------------------------------------------------------------------
    # Branch tracking helpers
    # ------------------------------------------------------------------

    def record_branch_expansion(self, node: SearchNode) -> None:
        """Record that a branch (root child subtree) was expanded.

        Tracks how many times each root-child subtree has been explored,
        used for sparsity bonus in _score_candidate.
        """
        if node is self.root or not node.parent or node.parent is not self.root:
            return
        node_id: str = node.id
        self._branch_expansions[node_id] = self._branch_expansions.get(node_id, 0) + 1

    def _total_branch_expansions(self) -> int:
        """Sum total expansion counts across all branches."""
        return sum(self._branch_expansions.values())

    def _node_depth(self, node: SearchNode) -> int:
        """Return depth of node from root (root = 0)."""
        depth: int = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    # ------------------------------------------------------------------
    # Node creation + reward propagation
    # ------------------------------------------------------------------

    def make_child(
        self,
        parent: SearchNode,
        score: float | None,
        step: int,
        eval_output: str = "",
    ) -> SearchNode:
        """Create a child of *parent* with the given eval score.

        Applies reward normalization, assigns branch_id, and backpropagates.

        Parameters:
            parent: The SearchNode that was expanded.
            score: Numeric eval score (None for broken/failed code).
            step: Current iteration number.
            eval_output: Raw eval stdout/stderr for debugging broken code.

        Returns:
            The newly created child SearchNode.
        """
        # Generate branch_id -- root children get unique branch_ids.
        if parent is self.root:
            branch_id: int = self._next_branch_id
            self._next_branch_id += 1
        else:
            if parent.branch_id is None:
                parent.branch_id = self._next_branch_id
                self._next_branch_id += 1
            branch_id = parent.branch_id

        child = SearchNode(code="", plan=f"improve from node {parent.id[:8]}",
                           stage="improve", parent=parent)
        child.step = step
        child.branch_id = branch_id
        child.eval_output = eval_output

        # Store the score as a MetricValue on the child.
        child.metric = MetricValue(value=score, maximize=self.maximize)

        # Compute normalized reward.
        reward: float = self._normalize_reward(score)
        child._reward = reward

        # Book-keeping.
        self.journal.append(child)

        # Update best-node (track raw scores, not normalized).
        if score is not None:
            is_better: bool = (score > self._best_score if self.maximize
                               else score < self._best_score)
            if is_better or self._best_score == (
                float("-inf") if self.maximize else float("inf")
            ):
                self._best_score = score
                self._best_node = child
                logger.info(f"[Tree] New best: score={score:.4f} at node {child.id[:8]}")

        # Record branch expansion (for sparsity bonus).
        self.record_branch_expansion(parent)

        # Backpropagate the NORMALIZED reward up to root.
        self._backpropagate(child, reward)

        return child

    def _backpropagate(self, node: SearchNode, reward: float) -> None:
        """Propagate reward up the tree (child -> parent -> ... -> root).

        Each ancestor increments its visits and adds the normalized reward.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    # ------------------------------------------------------------------
    # Tracking helpers
    # ------------------------------------------------------------------

    @property
    def total_nodes(self) -> int:
        """Number of nodes in the tree (including root)."""
        return len(self.journal)

    @property
    def total_expansions(self) -> int:
        """Total number of successful expansions across all nodes."""
        return sum(n.visits for n in self.journal.nodes)

    @property
    def best_node(self) -> Optional[SearchNode]:
        """Node with the highest score seen so far, or None."""
        return self._best_node

    @property
    def best_score(self) -> float:
        """Highest score seen so far."""
        return self._best_score

    # ------------------------------------------------------------------
    # State reporting
    # ------------------------------------------------------------------

    def get_node_info(self) -> dict:
        """Return a summary dict of the current tree state.

        Includes per-branch stats for LLM feedback context.
        """
        depth: int = self._compute_depth(self.root)
        branches: int = self._count_branches()
        all_scores: list[float | None] = [
            n.metric.value for n in self.journal.nodes
            if n.metric and n.metric.value is not None
        ]

        # Build per-branch stats.
        branch_stats: list[dict] = []
        failed_branches: list[int] = []
        for i, child in enumerate(self.root.children, 1):
            child_scores = [
                n.metric.value for n in self._collect_subtree_nodes(child)
                if n.metric and n.metric.value is not None
            ]
            child_visits = sum(n.visits for n in self._collect_subtree_nodes(child))
            branch_depth = self._compute_depth(child)
            best_child_score: float | None = (
                max(child_scores) if self.maximize else min(child_scores)
            ) if child_scores else None
            is_failed: bool = len(child_scores) > 0 and all(s is None for s in child_scores)

            branch_stats.append({
                "id": i, "best_score": best_child_score,
                "total_visits": child_visits, "evals_run": len(child_scores),
                "depth": branch_depth,
            })
            if is_failed:
                failed_branches.append(i)

        return {
            "total_nodes": self.total_nodes,
            "total_expansions": self.total_expansions,
            "max_depth": depth, "branches": branches,
            "best_score": self.best_score,
            "all_scores": all_scores,
            "branch_stats": branch_stats,
            "failed_branches": failed_branches,
            "avg_per_score": (sum(self._scores) / len(self._scores) if self._scores else None),
        }

    def _collect_subtree_nodes(self, node: SearchNode) -> list[SearchNode]:
        """Collect all nodes in a subtree (including the root of the subtree)."""
        result: list[SearchNode] = [node]
        for child in node.children:
            result.extend(self._collect_subtree_nodes(child))
        return result

    def _compute_depth(self, node: SearchNode) -> int:
        """Return the maximum depth from this node to its deepest leaf."""
        if not node.children:
            return 0
        return 1 + max(self._compute_depth(c) for c in node.children)  # type: ignore[operator]

    def _count_branches(self) -> int:
        """Count distinct branches from root (direct children)."""
        return len(self.root.children)

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def get_tree_structure(self) -> dict:
        """Return a flat, JSON-serializable tree structure.

        Each node is represented as a dict with minimal fields for
        inspection (id, parent_id, depth, score, eval_output, visits, reward,
        branch_id, step).  Root node has parent_id = None.
        """
        nodes: list[dict] = []
        for node in self.journal.nodes:
            parent_id = node.parent.id if node.parent else None
            nodes.append({
                "id": node.id, "parent_id": parent_id,
                "depth": self._node_depth(node), "stage": node.stage,
                "score": node.metric.value if node.metric else None,
                "eval_output": getattr(node, "eval_output", ""),
                "visits": node.visits, "total_reward": node.total_reward,
                "branch_id": node.branch_id, "step": node.step,
            })
        depth: int = self._compute_depth(self.root)
        return {
            "root_id": self.root.id,
            "best_node_id": self._best_node.id if self._best_node else None,
            "best_score": (self._best_score if self._best_score != float("inf") else None),
            "maximize": self.maximize, "explore_c": self.explore_c,
            "total_nodes": self.total_nodes,
            "total_expansions": self.total_expansions, "max_depth": depth,
            "branch_expansions": self._branch_expansions,
            "all_scores": self._scores, "reward_count": self._reward_count,
            "nodes": nodes,
        }

    def _restore_from_state(self, data: dict) -> None:
        """Restore tree state from saved data (reverse of get_tree_structure)."""
        self.journal.nodes.clear()
        self.root.children.clear()
        self.root.parent = None

        # Restore config.
        if "maximize" in data:
            self.maximize = data["maximize"]
            self._best_score = float("inf") if not self.maximize else float("-inf")
        self.explore_c = data.get("explore_c", self.explore_c)

        # Restore root.
        root_data = next((nd for nd in data["nodes"] if nd["id"] == data["root_id"]), None)
        self.root.id = data["root_id"]
        self.root.visits = root_data["visits"] if root_data else 0
        self.root.total_reward = root_data["total_reward"] if root_data else 0.0
        self.journal.append(self.root)

        # Restore rewards stats.
        self._scores = data.get("all_scores", [])
        self._reward_count = data.get("reward_count", 0)
        self._branch_expansions = data.get("branch_expansions", {})

        # Track max branch id so new children get unique IDs.
        all_nodes_data = [nd for nd in data["nodes"] if nd["id"] != self.root.id]
        if all_nodes_data:
            max_branch = max((nd.get("branch_id") for nd in all_nodes_data), default=0)
            max_branch = max_branch if max_branch is not None else 0
            self._next_branch_id = max(self._next_branch_id, max_branch + 1)

        # Build node mapping and link parents.
        node_map: dict[str, SearchNode] = {}
        for nd in data["nodes"]:
            if nd["id"] == self.root.id:
                continue
            node = SearchNode(code="", plan=f"improve from saved node",
                              stage=nd["stage"], parent=None)
            node.id = nd["id"]
            node.visits = nd["visits"]
            node.total_reward = nd["total_reward"]
            node.branch_id = nd["branch_id"]
            node.step = nd["step"]
            node.eval_output = nd.get("eval_output", "")
            node.metric = MetricValue(value=nd["score"], maximize=self.maximize)
            node_map[node.id] = node
            self.journal.append(node)

        # Link parents.
        for nd in data["nodes"]:
            if nd["id"] == self.root.id:
                continue
            node = node_map[nd["id"]]
            parent_id = nd["parent_id"]
            if parent_id in node_map:
                parent = node_map[parent_id]
            elif parent_id == self.root.id or parent_id is None:
                parent = self.root
            else:
                raise ValueError(f"Unknown parent {parent_id} in tree restoration")
            node.parent = parent
            parent.children.add(node)

        # Restore best node.
        best_id = data.get("best_node_id")
        if best_id in node_map:
            self._best_node = node_map[best_id]
            self._best_score = data.get("best_score", self._best_score)
        elif data.get("best_score") is not None:
            self._best_score = data.get("best_score")  # type: ignore[assignment]
