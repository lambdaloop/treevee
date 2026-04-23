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
  document.getElementById('sum-maximize').textContent = maximize ? '↑ yes' : '↓ no';
}

function renderIterations() {
  const nodes = StateLoader.getNodesByStep();
  const container = document.getElementById('iterations-list');
  container.innerHTML = '';

  const reversed = [...nodes].reverse();

  for (const node of reversed) {
    const item = document.createElement('div');
    item.className = 'iteration-item compact';

    const isBest = StateLoader.isBestNode(node.id);
    const scoreReason = StateLoader.getScoreReason(node);
    const isActualError = node.score === null && scoreReason && !scoreReason.startsWith('Baseline') && !scoreReason.startsWith('Score not parsed');
    const isRoot = node.stage === 'root';
    const histEntry = isRoot ? null : StateLoader.getHistoryEntryForStep(node.step);

    if (isBest) item.classList.add('best');
    if (isActualError) item.classList.add('timed-out');
    if (isRoot) item.classList.add('is-root');

    const row = document.createElement('div');
    row.className = 'iteration-row';

    const emoji = document.createElement('span');
    emoji.className = 'iteration-emoji';
    emoji.textContent = stageEmojis[node.stage] || '';
    row.appendChild(emoji);

    const num = document.createElement('span');
    num.className = 'iteration-num';
    num.textContent = isRoot ? 'Root' : `#${node.step}`;
    row.appendChild(num);

    const score = document.createElement('span');
    score.className = 'iteration-score';
    if (node.score === null) {
      score.textContent = 'N/A';
      score.classList.add('no-change');
    } else {
      score.textContent = node.score.toFixed(4);
      const parent = isRoot ? null : StateLoader.getNodes().find((n) => n.id === node.parent_id);
      const maximize = StateLoader.getMaximize();
      if (parent && parent.score !== null) {
        if (maximize ? node.score > parent.score : node.score < parent.score) score.classList.add('improved');
        else if (maximize ? node.score < parent.score : node.score > parent.score) score.classList.add('degraded');
        else score.classList.add('no-change');
      }
    }
    row.appendChild(score);

    if (isBest) {
      const badge = document.createElement('span');
      badge.className = 'iteration-badge badge-best';
      badge.textContent = 'BEST';
      row.appendChild(badge);
    }

    if (isActualError) {
      const badge = document.createElement('span');
      badge.className = 'iteration-badge badge-timeout';
      badge.textContent = 'ERR';
      row.appendChild(badge);
    }

    const summary = document.createElement('span');
    summary.className = 'iteration-summary';
    if (histEntry?.edit_summary) {
      summary.textContent = histEntry.edit_summary;
    } else if (node.score === null) {
      summary.textContent = scoreReason || '';
    }
    row.appendChild(summary);

    item.appendChild(row);

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
