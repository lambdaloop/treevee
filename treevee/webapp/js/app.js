function renderSummary() {
  const history = StateLoader.getHistory();
  const tree = StateLoader.getTreeStructure();
  const bestScore = StateLoader.getBestScore();
  const bestNodeId = StateLoader.getBestNodeId();
  const bestIter = StateLoader.getBestIteration();
  const maximize = StateLoader.getMaximize();
  const nodes = StateLoader.getNodes();

  document.getElementById('sum-iterations').textContent = nodes.length;
  document.getElementById('sum-best-score').textContent = bestScore !== null ? bestScore.toFixed(6) : 'N/A';
  document.getElementById('sum-best-score').className = 'value best';
  document.getElementById('sum-best-iter').textContent = bestIter !== null ? bestIter : '-';

  document.getElementById('sum-nodes').textContent = nodes.length;
  document.getElementById('sum-depth').textContent = tree?.max_depth ?? '-';
  document.getElementById('sum-maximize').textContent = maximize ? '\u2191 yes' : '\u2193 no';
}

function renderIterations() {
  const nodes = StateLoader.getNodesByStep();
  const bestNodeId = StateLoader.getBestNodeId();
  const container = document.getElementById('iterations-list');
  container.innerHTML = '';

  // Render in reverse order (newest first)
  const reversed = [...nodes].reverse();

  for (const node of reversed) {
    const item = document.createElement('div');
    item.className = 'iteration-item';

    const isBest = StateLoader.isBestNode(node.id);
    const scoreReason = StateLoader.getScoreReason(node);
    const isActualError = node.score === null && scoreReason && !scoreReason.startsWith('Baseline') && !scoreReason.startsWith('Score not parsed');
    const isRoot = node.stage === 'root';
    const histEntry = isRoot ? null : StateLoader.getHistoryEntryForStep(node.step);

    if (isBest) item.classList.add('best');
    if (isActualError) item.classList.add('timed-out');
    if (isRoot) item.classList.add('is-root');

    const header = document.createElement('div');
    header.className = 'iteration-header';

    const num = document.createElement('span');
    num.className = 'iteration-num';
    num.textContent = isRoot ? '\uD83C\uDF38 Root' : `Step ${node.step}`;

    const score = document.createElement('span');
    score.className = 'iteration-score';
    if (node.score === null) {
      score.textContent = 'N/A';
      score.classList.add('no-change');
    } else {
      score.textContent = node.score.toFixed(4);
      // Color based on improvement from parent
      const parent = isRoot ? null : StateLoader.getNodes().find((n) => n.id === node.parent_id);
      const maximize = StateLoader.getMaximize();
      if (parent && parent.score !== null) {
        if (maximize ? node.score > parent.score : node.score < parent.score) score.classList.add('improved');
        else if (maximize ? node.score < parent.score : node.score > parent.score) score.classList.add('degraded');
        else score.classList.add('no-change');
      }
    }

    header.appendChild(num);
    header.appendChild(score);

    // Badges
    if (isBest) {
      const badge = document.createElement('span');
      badge.className = 'iteration-badge badge-best';
      badge.textContent = '\u2B50 BEST';
      header.appendChild(badge);
    }

    if (isActualError) {
      const badge = document.createElement('span');
      badge.className = 'iteration-badge badge-timeout';
      badge.textContent = '\u26A0 ERR';
      header.appendChild(badge);
    }

    if (node.score === null && scoreReason === 'Baseline (no eval score)') {
      const badge = document.createElement('span');
      badge.className = 'iteration-badge';
      badge.style.cssText = 'background:var(--accent-sky-dim);color:var(--accent-sky);';
      badge.textContent = '\uD83C\uDF38 BASELINE';
      header.appendChild(badge);
    }

    item.appendChild(header);

    // Summary line from history or score reason (always shown)
    const summary = document.createElement('div');
    summary.className = 'iteration-summary';
    if (histEntry?.edit_summary) {
      summary.textContent = histEntry.edit_summary;
    } else if (node.score === null) {
      summary.style.color = 'var(--text-muted)';
      summary.textContent = scoreReason;
    } else {
      summary.style.color = 'var(--text-muted)';
      summary.textContent = 'No edit summary';
    }
    item.appendChild(summary);

    // Timestamp
    if (histEntry?.datetime) {
      const ts = document.createElement('div');
      ts.className = 'iteration-time';
      try {
        ts.textContent = new Date(histEntry.datetime).toLocaleString();
      } catch {
        ts.textContent = histEntry.datetime;
      }
      item.appendChild(ts);
    }

    // Files
    const files = [];
    if (histEntry) {
      if (histEntry.files_modified.length > 0) files.push(`M: ${histEntry.files_modified.join(', ')}`);
      if (histEntry.files_added.length > 0) files.push(`A: ${histEntry.files_added.join(', ')}`);
      if (histEntry.files_deleted.length > 0) files.push(`D: ${histEntry.files_deleted.join(', ')}`);
    }
    if (files.length > 0) {
      const filesEl = document.createElement('div');
      filesEl.className = 'iteration-files';
      filesEl.textContent = files.join(' | ');
      item.appendChild(filesEl);
    }

    // Inline diff section (hidden by default)
    if (histEntry?.diff_text && histEntry.diff_text.trim()) {
      const diffContainer = document.createElement('div');
      diffContainer.className = 'iteration-diff';
      diffContainer.style.cssText = 'display:none; margin-top:8px;';

      const diffHeader = document.createElement('div');
      diffHeader.style.cssText = 'display:flex;align-items:center;gap:6px;cursor:pointer;padding:4px 0;font-size:12px;color:var(--accent-pink);';

      const diffToggle = document.createElement('span');
      diffToggle.textContent = '›';
      diffToggle.style.cssText = 'font-size:9px;transition:transform 0.15s cubic-bezier(0.34, 1.56, 0.64, 1);';

      const diffLabel = document.createElement('span');
      diffLabel.textContent = 'View diff';

      diffHeader.appendChild(diffToggle);
      diffHeader.appendChild(diffLabel);
      diffContainer.appendChild(diffHeader);

      const diffContent = document.createElement('div');
      diffContent.className = 'diff-inline';
      diffContent.style.cssText = 'display:none;';
      diffContent.innerHTML = renderDiffInline(histEntry.diff_text);
      diffContainer.appendChild(diffContent);

      diffHeader.addEventListener('click', (e) => {
        e.stopPropagation();
        const isHidden = diffContent.style.display === 'none';
        diffContent.style.display = isHidden ? 'block' : 'none';
        diffToggle.style.transform = isHidden ? 'rotate(90deg)' : '';
        diffLabel.textContent = isHidden ? 'Hide diff' : 'View diff';
      });

      item.appendChild(diffContainer);
    }

    // Click to show node details in the bottom section
    item.addEventListener('click', () => {
      showNodeDetailToBottom(node);
    });

    container.appendChild(item);
  }
}

function getBestIterFromSnapshot(snapshotName, nodes) {
  if (!snapshotName || !nodes) return null;
  const nodeId = snapshotName.replace('iter_snapshot_', '');
  const bestNode = nodes.find((n) => n.id === nodeId);
  if (!bestNode) return null;

  const history = StateLoader.getHistory();
  const entry = history.find((h) => h.iter === bestNode.step);
  return entry ? entry.iter : null;
}

function renderAll() {
  renderSummary();
  renderScoreChart();
  renderTree();
  renderIterations();
}

// File input handling
function initFileInput() {
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browse-btn');
  const dropZone = document.getElementById('file-drop-zone');

  browseBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  // Drag and drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // Also allow drop on the whole body
  document.body.addEventListener('dragover', (e) => e.preventDefault());
  document.body.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.json')) handleFile(file);
  });
}

async function handleFile(file) {
  try {
    await StateLoader.load(file);
    document.getElementById('dashboard').style.display = 'block';
    document.getElementById('file-drop-zone').style.display = 'none';
    renderAll();
  } catch (err) {
    alert(`Error loading file: ${err.message}`);
  }
}

// Auto-load from server API
async function autoLoadFromServer() {
  try {
    const resp = await fetch('/api/state');
    if (!resp.ok) return false;
    const data = await resp.json();
    StateLoader.loadFromData(data);
    document.getElementById('dashboard').style.display = 'block';
    document.getElementById('file-drop-zone').style.display = 'none';
    renderAll();
    return true;
  } catch (err) {
    console.warn('Auto-load failed:', err);
    return false;
  }
}

// Initialize
(async () => {
  const loaded = await autoLoadFromServer();
  if (!loaded) {
    initFileInput();
  }
  initTreeControls();
  initDiffClose();
})();
