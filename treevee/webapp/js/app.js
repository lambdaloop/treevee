function renderSummary() {
  const history = StateLoader.getHistory();
  const bestScore = StateLoader.getBestScore();
  const maximize = StateLoader.getMaximize();

  const bar = document.getElementById('summary-bar');
  if (bar) {
    const iterText = history.length > 0 ? history[history.length - 1].iter : '0';
    const scoreColor = bestScore !== null ? 'var(--accent-mint)' : 'var(--text-muted)';
    bar.innerHTML = `<strong>Best Score:</strong> <span style="color:${scoreColor};font-weight:700">${bestScore !== null ? bestScore.toFixed(6) : 'N/A'}</span>&emsp;&emsp;<strong>${maximize ? 'Maximizing' : 'Minimizing'} score:</strong> ${maximize ? '↑' : '↓'}&emsp;&emsp;<strong>Iterations:</strong> <span style="color:var(--accent-lavender);font-weight:600">${iterText}</span>`;
  }
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

async function loadFromServer() {
  const resp = await fetch('/api/state');
  if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
  const data = await resp.json();
  StateLoader.loadFromData(data);
  document.getElementById('dashboard').style.display = 'block';
  renderAll();
}

// Initialize
(async () => {
  try {
    await loadFromServer();
  } catch (err) {
    console.error('Failed to load state:', err);
  }
  initTreeControls();
  initDiffClose();
})();
