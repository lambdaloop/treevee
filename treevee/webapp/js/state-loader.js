const StateLoader = (() => {
  let state = null;

  function load(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          state = JSON.parse(e.target.result);
          resolve(state);
        } catch (err) {
          reject(new Error(`Failed to parse JSON: ${err.message}`));
        }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }

  function loadFromData(data) {
    state = data;
    return state;
  }

  function getState() {
    return state;
  }

  function getHistory() {
    return state?.history || [];
  }

  function getTreeStructure() {
    return state?.tree_structure || null;
  }

  function getNodes() {
    return state?.tree_structure?.nodes || [];
  }

  function getBestScore() {
    return state?.best_score ?? null;
  }

  function getBestSnapshotIteration() {
    return state?.best_snapshot_iteration ?? null;
  }

  function getMaximize() {
    return state?.tree_structure?.maximize ?? true;
  }

  function getAllScores() {
    return state?.tree_structure?.all_scores || [];
  }

  /**
   * Get the best node ID from the tree (fallback when snapshot doesn't match).
   */
  function getBestNodeId() {
    const tree = state?.tree_structure;
    if (tree?.best_node_id) return tree.best_node_id;
    const nodes = getNodes();
    const maximize = getMaximize();
    let best = null;
    let root = nodes.find((n) => n.stage === 'root');
    for (const n of nodes) {
      if (n.score !== null && (best === null || (maximize ? n.score > best.score : n.score < best.score))) {
        best = n;
      }
    }
    // Root is the best when all nodes have null scores.
    if (!best && root) return root.id;
    return best?.id ?? null;
  }

  /**
   * Get the best iteration from history based on best score.
   */
  function getBestIteration() {
    const history = getHistory();
    const maximize = getMaximize();
    let bestIter = null;
    let bestScore = maximize ? -Infinity : Infinity;
    for (const h of history) {
      if (h.score !== null && (maximize ? h.score > bestScore : h.score < bestScore)) {
        bestScore = h.score;
        bestIter = h.iter;
      }
    }
    return bestIter;
  }

  /**
   * Extract a human-readable reason why a node's score is None.
   */
  function getScoreReason(node) {
    if (node.score !== null) return null;
    const evalOutput = (node.eval_output || '').trim();
    const history = getHistory();
    const histEntry = history.find((h) => h.iter === node.step);

    // Root node
    if (node.stage === 'root') return 'Baseline (no eval score)';

    // Check history for timeout
    if (histEntry?.timed_out) return 'Evaluation timed out';

    // Check eval_output for clues
    if (!evalOutput) return 'Score could not be parsed';

    const lower = evalOutput.toLowerCase();

    if (lower.includes('timeout') || lower.includes('timed out')) return 'Evaluation timed out';
    if (lower.includes('error') || lower.includes('exception') || lower.includes('traceback')) {
      // Extract the actual error message
      const lines = evalOutput.split('\n');
      for (const line of lines) {
        const l = line.trim().toLowerCase();
        if (l.includes('error') || l.includes('exception')) {
          return line.trim().slice(0, 200);
        }
      }
      return 'Evaluation error occurred';
    }

    // If eval ran but produced no parseable score, show a clean message
    // Don't show warnings or download messages as errors
    if (lower.includes('warn') || lower.includes('warning')) {
      return 'Score not parsed (eval produced warnings)';
    }
    if (lower.includes('downloading') || lower.includes('download')) {
      return 'Score not parsed (model downloading)';
    }

    // Generic: eval ran but score wasn't extracted
    return 'Score not parsed';
  }

  /**
   * Get history entry for a given node step, or null.
   */
  function getHistoryEntryForStep(step) {
    return getHistory().find((h) => h.iter === step) ?? null;
  }

  /**
   * Get all nodes sorted by step (for chart display).
   */
  function getNodesByStep() {
    return [...getNodes()].sort((a, b) => a.step - b.step);
  }

  /**
   * Check if a node is the best node.
   */
  function isBestNode(nodeId) {
    const bestId = getBestNodeId();
    return nodeId === bestId;
  }

  /**
   * Build the path from root to the best node by walking parent_id chain.
   * Returns entries sorted root → best, with score delta from parent.
   * Each entry has: iter, score, editSummary, datetime, delta, isRoot.
   */
  function getProgressionPath() {
    const bestId = getBestNodeId();
    if (!bestId) return [];

    const nodes = getNodes();
    const nodeMap = new Map();
    for (const n of nodes) {
      nodeMap.set(n.id, n);
    }

    const history = getHistory();
    const historyMap = new Map();
    for (const h of history) {
      historyMap.set(h.iter, h);
    }

    // Walk parent chain from best node back to root.
    const path = [];
    let current = nodeMap.get(bestId);
    while (current) {
      path.push(current);
      current = nodeMap.get(current.parent_id);
    }
    // Reverse so root → best.
    path.reverse();

    const maximize = getMaximize();
    const entries = [];
    let prevScore = null;
    for (const node of path) {
      const step = node.step;
      const histEntry = historyMap.get(step) ?? null;
      const score = node.score;
      const delta = (score !== null && prevScore !== null) ? score - prevScore : null;
      const isImprovement = (score !== null && prevScore !== null && (maximize ? score > prevScore : score < prevScore));
      entries.push({
        iter: step,
        score,
        editSummary: histEntry?.edit_summary ?? null,
        datetime: histEntry?.datetime ?? null,
        delta,
        isImprovement,
        isRoot: node.stage === 'root',
        nodeId: node.id,
      });
      if (score !== null) prevScore = score;
    }
    return entries;
  }

  /**
   * Build the improvement sequence from root to best node.
   * Root is plotted at iter=-1, improvement nodes at their iteration.
   * Each entry has: iter, score, editSummary, datetime, delta, isRoot.
   */
  function getProgressionImprovements() {
    const path = getProgressionPath();
    const entries = [];
    for (const p of path) {
      if (p.isRoot || (p.isImprovement && p.score !== null)) {
        entries.push({
          iter: p.isRoot ? -1 : p.iter,
          score: p.score,
          editSummary: p.editSummary,
          datetime: p.datetime,
          delta: p.delta,
          isRoot: p.isRoot,
          nodeId: p.nodeId,
        });
      }
    }
    return entries;
  }

  /**
   * Build running-best-score progression over iterations.
   * Uses node scores as the authoritative data source, joined to history
   * by step number to get edit summaries.
   * Each entry has: iter, runningBest, isImprovement, editSummary, score, datetime.
   */
  function getBestProgression() {
    const nodes = getNodesByStep();
    const history = getHistory();
    const historyMap = new Map();
    for (const h of history) {
      historyMap.set(h.iter, h);
    }

    const maximize = getMaximize();
    let bestScore = null;
    const progression = [];
    for (const node of nodes) {
      const step = node.step;
      const histEntry = historyMap.get(step) ?? null;
      const score = node.score;
      const isImprovement = bestScore === null || (score !== null && (maximize ? score > bestScore : score < bestScore));
      if (score !== null && isImprovement) bestScore = score;
      progression.push({
        iter: step,
        runningBest: bestScore,
        isImprovement,
        editSummary: (isImprovement && histEntry && histEntry.edit_summary) ? histEntry.edit_summary : null,
        score,
        datetime: histEntry?.datetime ?? null,
      });
    }
    return progression;
  }

  return {
    load, loadFromData, getState, getHistory, getTreeStructure, getNodes,
    getBestScore, getBestSnapshotIteration, getMaximize, getAllScores,
    getBestNodeId, getBestIteration, getScoreReason, getHistoryEntryForStep,
    getNodesByStep, isBestNode, getBestProgression, getProgressionPath, getProgressionImprovements,
  };
})();
