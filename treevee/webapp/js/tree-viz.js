const rootDiffCache = {};

const stageColors = {
  root: '#f7c2e0',
  improve: '#d4b0e8',
  fusion: '#b5e9cd',
  debug: '#f7d0c7',
  draft: '#c2d8f2',
};

const defaultColor = '#d4b0e8';

const stageEmojis = {
  root: '\uD83C\uDF38',
  improve: '\u2728',
  fusion: '\uD83E\uDDEC',
  debug: '\uD83D\uDCAC',
  draft: '\uD83C\uDF1F',
};

function getNodeColor(node) {
  return stageColors[node.stage] || defaultColor;
}

function buildTree(nodes) {
  const nodeMap = new Map();
  const roots = [];

  for (const node of nodes) {
    nodeMap.set(node.id, { ...node, children: [] });
  }

  for (const node of nodes) {
    const mapped = nodeMap.get(node.id);
    if (node.parent_id === null || node.parent_id === undefined) {
      roots.push(mapped);
    } else {
      const parent = nodeMap.get(node.parent_id);
      if (parent) {
        parent.children.push(mapped);
      }
    }
  }

  return roots;
}

// Track expanded nodes across re-renders
let expandedNodes = new Set();

function toggleNode(nodeId) {
  if (expandedNodes.has(nodeId)) {
    expandedNodes.delete(nodeId);
  } else {
    expandedNodes.add(nodeId);
  }
}

function isNodeExpanded(nodeId) {
  return expandedNodes.has(nodeId);
}

function getBestPathNodeIds(bestNodeId, nodes) {
  const nodeMap = new Map();
  for (const n of nodes) nodeMap.set(n.id, n);
  const path = new Set();
  let current = bestNodeId;
  while (current) {
    path.add(current);
    const node = nodeMap.get(current);
    current = node?.parent_id ?? null;
  }
  return path;
}

function createTreeNode(nodeData, allNodes, pathSet) {
  const hasChildren = nodeData.children.length > 0;
  const isBest = StateLoader.isBestNode(nodeData.id);
  const onBestPath = pathSet ? pathSet.has(nodeData.id) : isBest;
  const scoreReason = StateLoader.getScoreReason(nodeData);
  const isActualError = nodeData.score === null && scoreReason && !scoreReason.startsWith('Baseline') && !scoreReason.startsWith('Score not parsed');

  const wrapper = document.createElement('div');
  wrapper.className = 'tree-node';

  const header = document.createElement('div');
  header.className = 'tree-node-header';
  if (onBestPath) header.classList.add('selected');

  const toggle = document.createElement('span');
  toggle.className = 'tree-toggle';
  toggle.textContent = hasChildren ? (isNodeExpanded(nodeData.id) ? '⌄' : '›') : '';

  const dot = document.createElement('span');
  dot.className = 'tree-dot';
  dot.style.background = getNodeColor(nodeData);
  dot.title = `${stageEmojis[nodeData.stage] || ''} ${nodeData.stage} · v${nodeData.visits}`;
  if (isActualError) dot.style.boxShadow = '0 0 8px rgba(255, 183, 178, 0.5)';
  if (onBestPath) dot.style.boxShadow = '0 0 8px rgba(168, 230, 207, 0.5)';

  // Depth badge
  const depthBadge = document.createElement('span');
  depthBadge.className = 'tree-depth-badge';
  depthBadge.textContent = `D${nodeData.depth}`;

  // Root sparkle badge
  let rootBadge = null;
  if (nodeData.stage === 'root') {
    rootBadge = document.createElement('span');
    rootBadge.className = 'tree-depth-badge';
    rootBadge.style.cssText = 'background:var(--accent-pink-dim);color:var(--accent-pink);';
    rootBadge.textContent = '\uD83C\uDF38';
  }

  // Compact label — just the short ID
  const label = document.createElement('span');
  label.className = 'tree-label';
  const shortId = nodeData.id.slice(0, 8);
  label.textContent = `[${shortId}]`;

  const maximize = StateLoader.getMaximize();

  // Score with direction-aware styling
  const score = document.createElement('span');
  score.className = 'tree-score';
  const parent = allNodes.find((n) => n.id === nodeData.parent_id);
  if (nodeData.score !== null) {
    score.textContent = nodeData.score.toFixed(4);
    if (onBestPath) {
      score.style.color = 'var(--accent-mint)';
    } else if (parent && parent.score !== null) {
      const improved = maximize ? nodeData.score > parent.score : nodeData.score < parent.score;
      const degraded = maximize ? nodeData.score < parent.score : nodeData.score > parent.score;
      if (improved) {
        score.classList.add('improved');
      } else if (degraded) {
        score.classList.add('degraded');
      }
    }
  } else {
    score.textContent = 'N/A';
    score.classList.add('na');
  }

  // Score direction arrow (always shows raw numeric direction)
  let arrowEl = null;
  if (parent && nodeData.score !== null && parent.score !== null) {
    const delta = nodeData.score - parent.score;
    if (delta !== 0) {
      const isGood = maximize ? delta > 0 : delta < 0;
      arrowEl = document.createElement('span');
      arrowEl.className = `tree-arrow ${isGood ? 'up' : 'down'}`;
      arrowEl.textContent = delta > 0 ? '↑' : '↓';
      arrowEl.title = `${delta > 0 ? '+' : ''}${delta.toFixed(4)}`;
    }
  }

  // Error indicator — compact icon with tooltip
  let errorEl = null;
  if (isActualError) {
    errorEl = document.createElement('span');
    errorEl.className = 'tree-dot';
    errorEl.style.cssText = 'width:8px;height:8px;border-radius:50%;background:var(--accent-peach);flex-shrink:0;cursor:default;box-shadow:0 0 6px rgba(255,183,178,0.5);';
    errorEl.title = scoreReason;
  }

  // Best badge
  let bestBadge = null;
  if (isBest) {
    bestBadge = document.createElement('span');
    bestBadge.className = 'tree-depth-badge';
    bestBadge.style.cssText = 'background:var(--accent-mint-dim);color:var(--accent-mint);';
    bestBadge.textContent = '\u2B50';
  }

  // Child count badge
  let childCountBadge = null;
  if (hasChildren && !isNodeExpanded(nodeData.id)) {
    childCountBadge = document.createElement('span');
    childCountBadge.className = 'tree-child-count';
    childCountBadge.textContent = `(${nodeData.children.length})`;
  }

  // Info button — opens node details panel
  const infoBtn = document.createElement('span');
  infoBtn.className = 'tree-info-btn';
  infoBtn.textContent = 'ⓘ';
  infoBtn.title = 'Show details';

  header.appendChild(toggle);
  header.appendChild(dot);

  // Stage emoji
  const stageEmoji = document.createElement('span');
  stageEmoji.style.cssText = 'font-size:11px;flex-shrink:0;';
  stageEmoji.textContent = stageEmojis[nodeData.stage] || '';
  header.appendChild(stageEmoji);

  header.appendChild(depthBadge);
  if (rootBadge) header.appendChild(rootBadge);
  header.appendChild(label);
  header.appendChild(score);
  if (arrowEl) header.appendChild(arrowEl);
 
  if (errorEl) header.appendChild(errorEl);
  if (bestBadge) header.appendChild(bestBadge);
  if (childCountBadge) header.appendChild(childCountBadge);
  header.appendChild(infoBtn);

  wrapper.appendChild(header);

  // Children container
  let childrenContainer = null;
  if (hasChildren) {
    childrenContainer = document.createElement('div');
    childrenContainer.className = 'tree-children';
    if (!isNodeExpanded(nodeData.id)) {
      childrenContainer.classList.add('collapsed');
    }

    for (const child of nodeData.children) {
      const childNode = createTreeNode(child, allNodes, pathSet);
      childrenContainer.appendChild(childNode);
    }

    wrapper.appendChild(childrenContainer);
  }

  // Header click toggles expand/collapse only
  header.addEventListener('click', (e) => {
    if (hasChildren) {
      e.stopPropagation();
      toggleNode(nodeData.id);
      const isCollapsed = childrenContainer.classList.toggle('collapsed');
      toggle.textContent = isCollapsed ? '›' : '⌄';

      // Update child count badge
      if (childCountBadge) {
        if (isCollapsed) {
          childCountBadge.style.display = '';
        } else {
          childCountBadge.style.display = 'none';
        }
      }
    }
  });

  // Info button opens node details panel
  infoBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    document.querySelectorAll('.tree-node-header.selected').forEach((el) => el.classList.remove('selected'));
    header.classList.add('selected');
    showNodeDetailToBottom(nodeData);
  });

  return wrapper;
}

function renderTree() {
  const nodes = StateLoader.getNodes();
  const container = document.getElementById('tree-container');
  container.innerHTML = '';

  if (nodes.length === 0) {
    container.innerHTML = '<p style="color:var(--text-muted); padding:16px; text-align:center;">No tree data yet~ (｡•́︿•̀｡) <br><span style="font-size:11px;opacity:0.6;">the tree is sleeping... 🌸💤</span></p>';
    return;
  }

  const roots = buildTree(nodes);

  const bestNodeId = StateLoader.getBestNodeId();
  const pathSet = bestNodeId ? getBestPathNodeIds(bestNodeId, nodes) : new Set();

  // Compact legend
  const legend = document.createElement('div');
  legend.className = 'tree-legend';

  // Score range
  const allScores = nodes.filter(n => n.score !== null).map(n => n.score);
  if (allScores.length > 0) {
    const rangeItem = document.createElement('div');
    rangeItem.className = 'legend-item';
    rangeItem.style.cssText = 'color:#e8d4f4;font-size:10px;font-family:monospace;';
    rangeItem.textContent = `score ${Math.min(...allScores).toFixed(3)}–${Math.max(...allScores).toFixed(3)}`;
    legend.appendChild(rangeItem);
  }

  const stages = [...new Set(nodes.map((n) => n.stage))];
  for (const stage of stages) {
    const item = document.createElement('div');
    item.className = 'legend-item';
    const dot = document.createElement('span');
    dot.className = 'legend-dot';
    dot.style.background = getNodeColor({ stage });
    dot.title = stage;
    item.appendChild(dot);
    const count = nodes.filter(n => n.stage === stage).length;
    item.appendChild(document.createTextNode(`${stageEmojis[stage] || ''} ${stage}(${count})`));
    legend.appendChild(item);
  }

  container.appendChild(legend);

  for (const root of roots) {
    const treeNode = createTreeNode(root, nodes, pathSet);
    container.appendChild(treeNode);
  }
}

function expandAll() {
  const nodes = StateLoader.getNodes();
  const nodeMap = new Map();
  for (const n of nodes) {
    nodeMap.set(n.id, { ...n, children: [] });
  }
  for (const n of nodes) {
    const mapped = nodeMap.get(n.id);
    if (n.parent_id != null) {
      const parent = nodeMap.get(n.parent_id);
      if (parent) parent.children.push(mapped);
    }
  }
  for (const n of nodes) {
    const mapped = nodeMap.get(n.id);
    if (mapped.children.length > 0) {
      expandedNodes.add(n.id);
    }
  }
  renderTree();
}

function collapseAll() {
  expandedNodes.clear();
  renderTree();
}

function showNodeDetailToBottom(nodeData) {
  const diffSection = document.getElementById('diff-section');
  const diffOutput = document.getElementById('diff-output');
  const closeBtn = document.getElementById('close-diff');

  if (!diffSection || !diffOutput || !closeBtn) {
    return;
  }

  const shortId = nodeData.id.slice(0, 8);
  const isRoot = nodeData.stage === 'root';
  const historyEntry = isRoot ? null : StateLoader.getHistoryEntryForStep(nodeData.step);
  const scoreReason = StateLoader.getScoreReason(nodeData);
  const isBest = StateLoader.isBestNode(nodeData.id);

  let html = '';

  // Header
  html += `<p style="color:var(--text-secondary); padding:8px 0; font-size:13px;"><strong>${stageEmojis[nodeData.stage] || ''} ${nodeData.stage} node [${shortId}]</strong> (Step ${nodeData.step})</p>`;

  // Status banners
  if (isBest) {
    html += `<div style="background:var(--accent-mint-dim);border:1px solid rgba(168,230,207,0.3);border-radius:var(--radius-sm);padding:8px 12px;margin-bottom:12px;color:var(--accent-mint);font-weight:600;">\u2B50 Best node!</div>`;
  }

  if (nodeData.score === null && scoreReason && !scoreReason.startsWith('Baseline')) {
    html += `<div style="background:var(--accent-peach-dim);border:1px solid rgba(255,183,178,0.3);border-radius:var(--radius-sm);padding:8px 12px;margin-bottom:12px;color:var(--accent-peach);font-weight:600;">\u26A0 ${escapeHtml(scoreReason)}</div>`;
  }

  // Score comparison with parent
  const nodes = StateLoader.getNodesByStep();
  const parent = nodes.find((n) => n.id === nodeData.parent_id);
  const maximize = StateLoader.getMaximize();
  if (parent && parent.score !== null && nodeData.score !== null) {
    const delta = nodeData.score - parent.score;
    const isImprovement = maximize ? delta > 0 : delta < 0;
    const isDegradation = maximize ? delta < 0 : delta > 0;
    const deltaColor = isImprovement ? 'var(--accent-mint)' : isDegradation ? 'var(--accent-peach)' : 'var(--text-secondary)';
    html += `<p style="color:${deltaColor}; padding:4px 0; font-size:13px;font-weight:600;">Score: ${parent.score.toFixed(4)} → ${nodeData.score.toFixed(4)} (${delta > 0 ? '+' : ''}${delta.toFixed(4)})</p>`;
  } else if (nodeData.score !== null) {
    html += `<p style="color:var(--text-secondary); padding:4px 0; font-size:13px;">Score: ${nodeData.score.toFixed(4)}</p>`;
  } else if (parent && parent.score !== null) {
    html += `<p style="color:var(--accent-peach); padding:4px 0; font-size:13px;">Score: ${parent.score.toFixed(4)} → N/A</p>`;
  }

  // Node info
  html += '<div style="background:var(--bg-primary);border:1px solid var(--border-color);border-radius:6px;padding:10px;margin-bottom:12px;font-size:12px;">';
  html += detailRow('Stage', nodeData.stage);
  html += detailRow('Depth', nodeData.depth);
  html += detailRow('Visits', nodeData.visits);
  html += detailRow('Reward', nodeData.total_reward != null ? nodeData.total_reward.toFixed(4) : 'N/A');
  html += detailRow('Branch', nodeData.branch_id ?? 'N/A');
  html += detailRow('Parent', parent ? `Step ${parent.step}` : 'Root');
  if (nodeData.score !== null) {
    html += detailRow('Score', nodeData.score.toFixed(6));
  } else {
    html += `<div class="detail-row"><span class="detail-label">Score</span><span class="detail-value" style="color:var(--accent-peach);">None</span></div>`;
  }
  html += '</div>';

  // History info
  if (historyEntry) {
    html += '<div style="background:var(--bg-primary);border:1px solid var(--border-color);border-radius:6px;padding:10px;margin-bottom:12px;font-size:12px;">';
    html += '<h4 style="color:var(--accent-pink);margin-bottom:6px;font-size:13px;">Iteration Details</h4>';
    html += detailRow('Iteration', historyEntry.iter);
    if (historyEntry.timed_out) {
      html += `<div class="detail-row"><span class="detail-label" style="color:var(--accent-peach);">Timeout</span><span class="detail-value" style="color:var(--accent-peach);">Yes</span></div>`;
    }
    html += detailRow('Exec Time', `${historyEntry.exec_time.toFixed(2)}s`);
    if (historyEntry.datetime) {
      try {
        html += detailRow('Completed', new Date(historyEntry.datetime).toLocaleString());
      } catch {
        html += detailRow('Completed', historyEntry.datetime);
      }
    }
    html += detailRow('Files Modified', historyEntry.files_modified.length ? historyEntry.files_modified.join(', ') : 'None');
    html += detailRow('Files Added', historyEntry.files_added.length ? historyEntry.files_added.join(', ') : 'None');
    html += detailRow('Files Deleted', historyEntry.files_deleted.length ? historyEntry.files_deleted.join(', ') : 'None');
    if (historyEntry.edit_summary) {
      html += detailRow('Edit Summary', historyEntry.edit_summary);
    }
    html += '</div>';

  }

  // Eval output / Error output
  const hasEvalOutput = nodeData.eval_output && nodeData.eval_output.trim();
  const hasHistoryEntry = !!historyEntry;
  const isError = nodeData.score === null && scoreReason && !scoreReason.startsWith('Baseline') && !scoreReason.startsWith('Score not parsed');
  if (hasEvalOutput || hasHistoryEntry) {
    const sectionBorder = isError ? 'rgba(255,183,178,0.4)' : 'var(--border-color)';
    const sectionBg = isError ? 'rgba(255,183,178,0.05)' : 'var(--bg-primary)';
    const headerColor = isError ? 'var(--accent-peach)' : 'var(--accent-pink)';
    const preBorder = isError ? 'rgba(255,183,178,0.3)' : 'var(--border-color)';
    const preBg = isError ? 'rgba(255,183,178,0.08)' : 'var(--bg-primary)';
    const preColor = isError ? 'var(--accent-peach)' : 'var(--text-secondary)';
    const btnBg = isError ? 'var(--accent-peach-dim)' : 'var(--bg-tertiary)';
    const btnColor = isError ? 'var(--accent-peach)' : 'var(--accent-pink)';
    const title = isError ? 'Error Output' : 'Evaluation Output';

    html += `<div style="background:${sectionBg};border:1px solid ${sectionBorder};border-radius:6px;padding:10px;margin-bottom:12px;">`;
    html += `<h4 style="color:${headerColor};margin-bottom:6px;font-size:13px;">${title}</h4>`;

    // Show timeout/exec_time from history entry
    if (hasHistoryEntry) {
      if (historyEntry.timed_out) {
        html += `<div class="detail-row"><span class="detail-label" style="color:var(--accent-peach);">Status</span><span class="detail-value" style="color:var(--accent-peach);">Timed out</span></div>`;
      } else {
        html += `<div class="detail-row"><span class="detail-label">Status</span><span class="detail-value">OK</span></div>`;
      }
      html += detailRow('Exec Time', `${historyEntry.exec_time.toFixed(2)}s`);
    }

    if (hasEvalOutput) {
      const evalPreview = nodeData.eval_output;
      const isLong = evalPreview.length > 400;
      const display = isLong ? evalPreview.slice(0, 400) + '\n... (truncated)' : evalPreview;
      html += `<pre style="background:${preBg};padding:10px;border-radius:6px;font-size:11px;overflow:auto;max-height:200px;border:1px solid ${preBorder};color:${preColor};white-space:pre-wrap;word-break:break-word;">${escapeHtml(display)}</pre>`;
      if (isLong) {
        html += `<button class="expand-btn" style="background:${btnBg};color:${btnColor};border:1px solid ${sectionBorder};padding:4px 10px;border-radius:4px;font-size:11px;cursor:pointer;margin-top:6px;">Show full output (${evalPreview.length} chars)</button>`;
      }
    } else if (hasHistoryEntry) {
      html += `<p style="color:var(--text-muted); padding:4px 0; font-size:11px;">No eval output captured.</p>`;
    }
    html += '</div>';
  }

  // Diff (with vs-parent / vs-root toggle)
  html += '<div style="background:var(--bg-primary);border:1px solid var(--border-color);border-radius:6px;padding:10px;margin-bottom:12px;">';
  html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">';
  html += '<h4 style="color:var(--accent-lavender);font-size:13px;margin:0;">Code Diff</h4>';
  if (!isRoot) {
    const btnBase = 'border:1px solid var(--border-color);border-radius:4px;padding:2px 8px;font-size:11px;cursor:pointer;';
    html += `<button id="diff-toggle-parent" class="active" style="${btnBase}background:var(--accent-lavender);color:var(--bg-primary);">vs parent</button>`;
    html += `<button id="diff-toggle-root" class="inactive" style="${btnBase}background:var(--bg-tertiary);color:var(--accent-lavender);">vs root</button>`;
  }
  html += '</div>';
  html += '<div id="diff-content">';
  if (historyEntry?.diff_text && historyEntry.diff_text.trim()) {
    html += renderDiffHTML(historyEntry.diff_text);
  } else if (isRoot) {
    html += '<p style="color:var(--text-muted); padding:4px 0; font-size:12px;">This is the root node.</p>';
  } else {
    html += '<p style="color:var(--text-muted); padding:4px 0; font-size:12px;">No diff available (no history entry for this step).</p>';
  }
  html += '</div>';
  html += '</div>';

  if (historyEntry) {
    // Planner input
    if (historyEntry?.planner_input && historyEntry.planner_input.trim()) {
      const pInput = historyEntry.planner_input;
      const pInputLong = pInput.length > 600;
      const pInputDisplay = pInputLong ? pInput.slice(0, 600) + '\n... (truncated, click to expand)' : pInput;
      html += '<div style="background:var(--bg-primary);border:1px solid var(--border-color);border-radius:6px;padding:10px;margin-bottom:12px;">';
      html += '<h4 style="color:var(--accent-sky);margin-bottom:6px;font-size:13px;">Planner Input</h4>';
      html += `<pre id="planner-input-display" style="background:var(--bg-tertiary);padding:10px;border-radius:6px;font-size:11px;overflow:auto;max-height:300px;border:1px solid var(--border-color);color:var(--text-secondary);white-space:pre-wrap;word-break:break-word;">${escapeHtml(pInputDisplay)}</pre>`;
      if (pInputLong) {
        html += `<button id="planner-input-expand" style="background:var(--bg-tertiary);color:var(--accent-sky);border:1px solid var(--border-color);padding:4px 10px;border-radius:4px;font-size:11px;cursor:pointer;margin-top:6px;">Show full planner input (${pInput.length} chars)</button>`;
      }
      html += '</div>';
    }

    // Planner output
    if (historyEntry?.planner_output && historyEntry.planner_output.trim()) {
      const pOutput = historyEntry.planner_output;
      const pOutputLong = pOutput.length > 600;
      const pOutputDisplay = pOutputLong ? pOutput.slice(0, 600) + '\n... (truncated, click to expand)' : pOutput;
      html += '<div style="background:var(--bg-primary);border:1px solid var(--border-color);border-radius:6px;padding:10px;margin-bottom:12px;">';
      html += '<h4 style="color:var(--accent-mint);margin-bottom:6px;font-size:13px;">Planner Output</h4>';
      html += `<pre id="planner-output-display" style="background:var(--bg-tertiary);padding:10px;border-radius:6px;font-size:11px;overflow:auto;max-height:300px;border:1px solid var(--border-color);color:var(--text-secondary);white-space:pre-wrap;word-break:break-word;">${escapeHtml(pOutputDisplay)}</pre>`;
      if (pOutputLong) {
        html += `<button id="planner-output-expand" style="background:var(--bg-tertiary);color:var(--accent-mint);border:1px solid var(--border-color);padding:4px 10px;border-radius:4px;font-size:11px;cursor:pointer;margin-top:6px;">Show full planner output (${pOutput.length} chars)</button>`;
      }
      html += '</div>';
    }

    // Editor input
    if (historyEntry?.editor_input && historyEntry.editor_input.trim()) {
      const eInput = historyEntry.editor_input;
      const eInputLong = eInput.length > 600;
      const eInputDisplay = eInputLong ? eInput.slice(0, 600) + '\n... (truncated, click to expand)' : eInput;
      html += '<div style="background:var(--bg-primary);border:1px solid var(--border-color);border-radius:6px;padding:10px;margin-bottom:12px;">';
      html += '<h4 style="color:var(--accent-lavender);margin-bottom:6px;font-size:13px;">Editor Input</h4>';
      html += `<pre id="editor-input-display" style="background:var(--bg-tertiary);padding:10px;border-radius:6px;font-size:11px;overflow:auto;max-height:300px;border:1px solid var(--border-color);color:var(--text-secondary);white-space:pre-wrap;word-break:break-word;">${escapeHtml(eInputDisplay)}</pre>`;
      if (eInputLong) {
        html += `<button id="editor-input-expand" style="background:var(--bg-tertiary);color:var(--accent-lavender);border:1px solid var(--border-color);padding:4px 10px;border-radius:4px;font-size:11px;cursor:pointer;margin-top:6px;">Show full editor input (${eInput.length} chars)</button>`;
      }
      html += '</div>';
    }

    // Editor output
    if (historyEntry?.editor_output && historyEntry.editor_output.trim()) {
      const eOutput = historyEntry.editor_output;
      const eOutputLong = eOutput.length > 600;
      const eOutputDisplay = eOutputLong ? eOutput.slice(0, 600) + '\n... (truncated, click to expand)' : eOutput;
      html += '<div style="background:var(--bg-primary);border:1px solid var(--border-color);border-radius:6px;padding:10px;margin-bottom:12px;">';
      html += '<h4 style="color:var(--accent-pink);margin-bottom:6px;font-size:13px;">Editor Output</h4>';
      html += `<pre id="editor-output-display" style="background:var(--bg-tertiary);padding:10px;border-radius:6px;font-size:11px;overflow:auto;max-height:300px;border:1px solid var(--border-color);color:var(--text-secondary);white-space:pre-wrap;word-break:break-word;">${escapeHtml(eOutputDisplay)}</pre>`;
      if (eOutputLong) {
        html += `<button id="editor-output-expand" style="background:var(--bg-tertiary);color:var(--accent-pink);border:1px solid var(--border-color);padding:4px 10px;border-radius:4px;font-size:11px;cursor:pointer;margin-top:6px;">Show full editor output (${eOutput.length} chars)</button>`;
      }
      html += '</div>';
    }
  }

  diffOutput.innerHTML = html;
  diffSection.style.display = 'block';
  closeBtn.style.display = 'inline-block';
  diffSection.scrollIntoView({ behavior: 'smooth' });

  // Diff toggle (vs parent / vs root)
  const toggleParent = diffOutput.querySelector('#diff-toggle-parent');
  const toggleRoot = diffOutput.querySelector('#diff-toggle-root');
  if (toggleParent && toggleRoot) {
    const diffContent = diffOutput.querySelector('#diff-content');
    function setActive(activeBtn, inactiveBtn) {
      activeBtn.classList.add('active');
      activeBtn.classList.remove('inactive');
      inactiveBtn.classList.add('inactive');
      inactiveBtn.classList.remove('active');
    }

    function showParentDiff() {
      setActive(toggleParent, toggleRoot);
      if (historyEntry?.diff_text && historyEntry.diff_text.trim()) {
        diffContent.innerHTML = renderDiffHTML(historyEntry.diff_text);
      } else {
        diffContent.innerHTML = '<p style="color:var(--text-muted); padding:4px 0; font-size:12px;">No diff available (no history entry for this step).</p>';
      }
    }

    async function showRootDiff() {
      setActive(toggleRoot, toggleParent);
      if (rootDiffCache[nodeData.id] !== undefined) {
        const cached = rootDiffCache[nodeData.id];
        diffContent.innerHTML = cached ? renderDiffHTML(cached) : '<p style="color:var(--text-muted); padding:4px 0; font-size:12px;">No changes from root.</p>';
        return;
      }
      diffContent.innerHTML = '<p style="color:var(--text-muted); padding:4px 0; font-size:12px;">Loading root diff\u2026</p>';
      try {
        const res = await fetch(`/api/diff_from_root?node_id=${encodeURIComponent(nodeData.id)}`);
        const data = await res.json();
        if (!res.ok) {
          diffContent.innerHTML = `<p style="color:var(--accent-peach); padding:4px 0; font-size:12px;">Could not compute root diff: ${escapeHtml(data.error || res.statusText)}</p>`;
          return;
        }
        rootDiffCache[nodeData.id] = data.diff_text || '';
        diffContent.innerHTML = data.diff_text?.trim()
          ? renderDiffHTML(data.diff_text)
          : '<p style="color:var(--text-muted); padding:4px 0; font-size:12px;">No changes from root.</p>';
      } catch (e) {
        diffContent.innerHTML = `<p style="color:var(--accent-peach); padding:4px 0; font-size:12px;">Error fetching root diff.</p>`;
      }
    }

    toggleParent.addEventListener('click', showParentDiff);
    toggleRoot.addEventListener('click', showRootDiff);
  }

  // Handle expand button
  const expandBtn = diffOutput.querySelector('.expand-btn');
  if (expandBtn) {
    expandBtn.addEventListener('click', () => {
      const pre = diffOutput.querySelector('pre');
      if (pre) pre.textContent = nodeData.eval_output;
      expandBtn.style.display = 'none';
    });
  }

  // Handle planner input expand button
  const pInputExpandBtn = diffOutput.querySelector('#planner-input-expand');
  if (pInputExpandBtn && historyEntry?.planner_input) {
    pInputExpandBtn.addEventListener('click', () => {
      const pre = diffOutput.querySelector('#planner-input-display');
      if (pre) pre.textContent = historyEntry.planner_input;
      pInputExpandBtn.style.display = 'none';
    });
  }

  // Handle planner output expand button
  const pOutputExpandBtn = diffOutput.querySelector('#planner-output-expand');
  if (pOutputExpandBtn && historyEntry?.planner_output) {
    pOutputExpandBtn.addEventListener('click', () => {
      const pre = diffOutput.querySelector('#planner-output-display');
      if (pre) pre.textContent = historyEntry.planner_output;
      pOutputExpandBtn.style.display = 'none';
    });
  }

  // Handle editor input expand button
  const eInputExpandBtn = diffOutput.querySelector('#editor-input-expand');
  if (eInputExpandBtn && historyEntry?.editor_input) {
    eInputExpandBtn.addEventListener('click', () => {
      const pre = diffOutput.querySelector('#editor-input-display');
      if (pre) pre.textContent = historyEntry.editor_input;
      eInputExpandBtn.style.display = 'none';
    });
  }

  // Handle editor output expand button
  const eOutputExpandBtn = diffOutput.querySelector('#editor-output-expand');
  if (eOutputExpandBtn && historyEntry?.editor_output) {
    eOutputExpandBtn.addEventListener('click', () => {
      const pre = diffOutput.querySelector('#editor-output-display');
      if (pre) pre.textContent = historyEntry.editor_output;
      eOutputExpandBtn.style.display = 'none';
    });
  }
}

function detailRow(label, value) {
  return `<div class="detail-row"><span class="detail-label">${label}:</span><span class="detail-value">${escapeHtml(String(value))}</span></div>`;
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderDiffInline(diffText) {
  const lines = diffText.split('\n');
  let html = '';

  for (const line of lines) {
    if (line.startsWith('--- ') || line.startsWith('+++ ')) {
      html += `<div class="diff-line header">${escapeHtml(line)}</div>`;
    } else if (line.startsWith('@@')) {
      html += `<div class="diff-line header">${escapeHtml(line)}</div>`;
    } else if (line.startsWith('+') && line.length > 1) {
      html += `<div class="diff-line added">${escapeHtml(line)}</div>`;
    } else if (line.startsWith('-') && line.length > 1) {
      html += `<div class="diff-line removed">${escapeHtml(line)}</div>`;
    } else if (line.startsWith('\\')) {
      // skip
    } else {
      html += `<div class="diff-line context">${escapeHtml(line)}</div>`;
    }
  }

  return html;
}

function initTreeControls() {
  document.getElementById('expand-all').addEventListener('click', expandAll);
  document.getElementById('collapse-all').addEventListener('click', collapseAll);
}
