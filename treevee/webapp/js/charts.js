let scoreChart = null;
let currentProgression = [];
let _chartDiscarded = [];
let _chartImprovements = [];

// Map an iteration number to the corresponding tree node and show its details.
function showNodeForIteration(iter) {
  const nodes = StateLoader.getNodesByStep();
  const node = nodes.find((n) => n.step === iter);
  if (node && typeof showNodeDetailToBottom === 'function') {
    showNodeDetailToBottom(node);
  }
}

function wrapText(ctx, text, maxWidth) {
  const words = text.split(' ');
  const lines = [];
  let current = '';
  for (const word of words) {
    const test = current ? current + ' ' + word : word;
    if (current && ctx.measureText(test).width > maxWidth) {
      lines.push(current);
      current = word;
    } else {
      current = test;
    }
  }
  if (current) lines.push(current);
  return lines;
}

// Store label bounding boxes for hit-testing.
let _labelBounds = [];
// Store computed label positions for use in afterDraw.
let _labelPositions = [];

function _drawLabelBox(ctx, boxX, boxY, boxW, boxH, isHovered) {
  if (isHovered) {
    ctx.fillStyle = 'rgba(35, 21, 53, 1.0)';
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 4);
    ctx.fill();

    ctx.strokeStyle = 'rgba(249, 168, 212, 1.0)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 4);
    ctx.stroke();
  } else {
    ctx.fillStyle = 'rgba(25, 15, 40, 1.0)';
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 4);
    ctx.fill();

    ctx.strokeStyle = 'rgba(249, 168, 212, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 4);
    ctx.stroke();
  }
}

/**
 * Place label boxes near data points to minimize visual clutter.
 * All math is done in pixel space to avoid coordinate mismatches.
 * Points with labelW=0 are treated as unlabeled (used only for collision).
 *
 * @param {{left:number, top:number, right:number, bottom:number}} bounds - Usable area
 * @param {number[]} pointsX - X positions of ALL data points in pixels
 * @param {number[]} pointsY - Y positions of ALL data points in pixels
 * @param {number[]} labelW - Width of each label box (0 = no label)
 * @param {number[]} labelH - Height of each label box (0 = no label)
 * @returns {Array<[number, number]>} Label box centers in pixels (same length as points)
 */
function placeLabels(bounds, pointsX, pointsY, labelW, labelH) {
  const POINT_R = 8;

  const n = pointsX.length;
  if (n === 0) return [];

  // 1. Generate 24 candidate positions per labeled point (close + far rings).
  //    Unlabeled points (w=0) get no candidates — they only serve as obstacles.
  const candidates = [];
  for (let i = 0; i < n; i++) {
    const w = labelW[i], h = labelH[i];
    if (w === 0 || h === 0) {
      candidates.push([]);
      continue;
    }
    const px = pointsX[i], py = pointsY[i];
    const vy = h / 2 + POINT_R + 6;
    const vx = w / 2 + POINT_R + 6;
    const vyFar = vy + 25;
    const vxFar = vx + 25;

    const raw = [
      [px, py - vy],
      [px, py + vy],
      [px + w * 0.3, py - vy],
      [px - w * 0.3, py - vy],
      [px + w * 0.3, py + vy],
      [px - w * 0.3, py + vy],
      [px + vx, py],
      [px - vx, py],
      [px + w * 0.15, py - vy],
      [px - w * 0.15, py - vy],
      [px + w * 0.15, py + vy],
      [px - w * 0.15, py + vy],
      [px + w * 0.6, py - vy],
      [px - w * 0.6, py - vy],
      [px + w * 0.6, py + vy],
      [px - w * 0.6, py + vy],
      [px, py - vyFar],
      [px, py + vyFar],
      [px + w * 0.3, py - vyFar],
      [px - w * 0.3, py - vyFar],
      [px + w * 0.3, py + vyFar],
      [px - w * 0.3, py + vyFar],
      [px + vxFar, py],
      [px - vxFar, py],
    ];

    const seen = new Set();
    const clamped = [];
    for (const [cx, cy] of raw) {
      const clX = Math.max(bounds.left + w / 2, Math.min(bounds.right - w / 2, cx));
      const clY = Math.max(bounds.top + h / 2, Math.min(bounds.bottom - h / 2, cy));
      const key = `${Math.round(clX * 100)},${Math.round(clY * 100)}`;
      if (!seen.has(key)) {
        seen.add(key);
        clamped.push([clX, clY]);
      }
    }
    candidates.push(clamped);
  }

  // 2. Geometry helpers (pixel coords)
  function _onSeg(ax, ay, bx, by, px, py) {
    return Math.min(ax, bx) <= px && px <= Math.max(ax, bx) &&
           Math.min(ay, by) <= py && py <= Math.max(ay, by);
  }

  function _segSegInt(x1, y1, x2, y2, x3, y3, x4, y4) {
    function ccw(ax, ay, bx, by, cx, cy) {
      return (cy - ay) * (bx - ax) - (by - ay) * (cx - ax);
    }
    const d1 = ccw(x3, y3, x4, y4, x1, y1);
    const d2 = ccw(x3, y3, x4, y4, x2, y2);
    const d3 = ccw(x1, y1, x2, y2, x3, y3);
    const d4 = ccw(x1, y1, x2, y2, x4, y4);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
    if (d1 === 0 && _onSeg(x3, y3, x4, y4, x1, y1)) return true;
    if (d2 === 0 && _onSeg(x3, y3, x4, y4, x2, y2)) return true;
    if (d3 === 0 && _onSeg(x1, y1, x2, y2, x3, y3)) return true;
    if (d4 === 0 && _onSeg(x1, y1, x2, y2, x4, y4)) return true;
    return false;
  }

  function _segRect(x1, y1, x2, y2, rcx, rcy, rw, rh) {
    const hw = rw / 2, hh = rh / 2;
    if (rcx - hw <= x1 && x1 <= rcx + hw && rcy - hh <= y1 && y1 <= rcy + hh) return true;
    if (rcx - hw <= x2 && x2 <= rcx + hw && rcy - hh <= y2 && y2 <= rcy + hh) return true;
    for (const [ex1, ey1, ex2, ey2] of [
      [rcx - hw, rcy - hh, rcx + hw, rcy - hh],
      [rcx - hw, rcy + hh, rcx + hw, rcy + hh],
      [rcx - hw, rcy - hh, rcx - hw, rcy + hh],
      [rcx + hw, rcy - hh, rcx + hw, rcy + hh],
    ]) {
      if (_segSegInt(x1, y1, x2, y2, ex1, ey1, ex2, ey2)) return true;
    }
    return false;
  }

  // Matches Chart.js stepped:'before' — horizontal at y1 to x2, then vertical at x2
  function _steppedSegRect(x1, y1, x2, y2, rcx, rcy, rw, rh) {
    return _segRect(x1, y1, x2, y1, rcx, rcy, rw, rh) ||
           _segRect(x2, y1, x2, y2, rcx, rcy, rw, rh);
  }

  function _labelOverlap(lx1, ly1, lw1, lh1, lx2, ly2, lw2, lh2) {
    const ox = Math.max(0, Math.min(lx1 + lw1 / 2, lx2 + lw2 / 2) - Math.max(lx1 - lw1 / 2, lx2 - lw2 / 2));
    const oy = Math.max(0, Math.min(ly1 + lh1 / 2, ly2 + lh2 / 2) - Math.max(ly1 - lh1 / 2, ly2 - lh2 / 2));
    return (ox * oy) / (lw1 * lh1);
  }

  // 3. Greedy assignment (left-to-right)
  const placed = new Array(n).fill(null);  // each entry: [cx, cy, w, h] or null
  const nLabeled = labelW.filter(w => w > 0).length;
  const maxSegments = Math.max(n - 1, 1);
  const maxLabelPairs = Math.max(nLabeled * (nLabeled - 1) / 2, 1);
  const bw = bounds.right - bounds.left, bh = bounds.bottom - bounds.top;
  const canvasDiag = Math.sqrt(bw * bw + bh * bh);

  for (let i = 0; i < n; i++) {
    const w = labelW[i], h = labelH[i];
    if (w === 0 || h === 0) continue;

    let bestCand = null;
    let bestCost = Infinity;

    for (const [cx, cy] of candidates[i]) {
      // Hard-filter: skip candidates overlapping own point
      if (Math.abs(cx - pointsX[i]) <= w / 2 + POINT_R &&
          Math.abs(cy - pointsY[i]) <= h / 2 + POINT_R) continue;

      let cost = 0;
      // Point overlap penalty — all other points
      for (let k = 0; k < n; k++) {
        if (k === i) continue;
        if (Math.abs(cx - pointsX[k]) <= w / 2 + POINT_R &&
            Math.abs(cy - pointsY[k]) <= h / 2 + POINT_R) {
          cost += 10.0;
        }
      }
      // Line segment penalty (stepped:'before')
      for (let k = 0; k < n - 1; k++) {
        if (_steppedSegRect(pointsX[k], pointsY[k], pointsX[k + 1], pointsY[k + 1], cx, cy, w, h)) {
          cost += 10.0 / maxSegments;
        }
      }
      // Label-label overlap
      for (let k = 0; k < n; k++) {
        if (k === i || !placed[k]) continue;
        cost += 15.0 / maxLabelPairs * _labelOverlap(cx, cy, w, h, placed[k][0], placed[k][1], placed[k][2], placed[k][3]);
      }
      // Proximity bonus
      const dx = cx - pointsX[i];
      const dy = cy - pointsY[i];
      const distNorm = Math.sqrt(dx * dx + dy * dy) / canvasDiag;
      cost -= 0.5 * (1.0 - Math.tanh(distNorm / 0.05));

      if (cost < bestCost) {
        bestCost = cost;
        bestCand = [cx, cy];
      }
    }
    placed[i] = bestCand ? [bestCand[0], bestCand[1], w, h] : [pointsX[i], pointsY[i], w, h];
  }

  // 4. Refinement passes with convergence check
  for (let pass = 0; pass < 10; pass++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      const w = labelW[i], h = labelH[i];
      if (w === 0 || h === 0) continue;

      let bestCand = null;
      let bestCost = Infinity;

      for (const [cx, cy] of candidates[i]) {
        if (Math.abs(cx - pointsX[i]) <= w / 2 + POINT_R &&
            Math.abs(cy - pointsY[i]) <= h / 2 + POINT_R) continue;

        let cost = 0;
        for (let k = 0; k < n; k++) {
          if (k === i) continue;
          if (Math.abs(cx - pointsX[k]) <= w / 2 + POINT_R &&
              Math.abs(cy - pointsY[k]) <= h / 2 + POINT_R) {
            cost += 10.0;
          }
        }
        for (let k = 0; k < n - 1; k++) {
          if (_steppedSegRect(pointsX[k], pointsY[k], pointsX[k + 1], pointsY[k + 1], cx, cy, w, h)) {
            cost += 10.0 / maxSegments;
          }
        }
        for (let k = 0; k < n; k++) {
          if (k === i || !placed[k]) continue;
          cost += 15.0 / maxLabelPairs * _labelOverlap(cx, cy, w, h, placed[k][0], placed[k][1], placed[k][2], placed[k][3]);
        }
        const dx = cx - pointsX[i];
        const dy = cy - pointsY[i];
        const distNorm = Math.sqrt(dx * dx + dy * dy) / canvasDiag;
        cost -= 0.5 * (1.0 - Math.tanh(distNorm / 0.05));

        if (cost < bestCost) {
          bestCost = cost;
          bestCand = [cx, cy];
        }
      }
      if (bestCand && (bestCand[0] !== placed[i][0] || bestCand[1] !== placed[i][1])) {
        moved = true;
        placed[i] = [bestCand[0], bestCand[1], w, h];
      }
    }
    if (!moved) break;
  }

  return placed.map(p => p ? [p[0], p[1]] : [0, 0]);
}

// Boxed edit-summary labels near each improvement point, with leader lines.
const labelPlugin = {
  id: 'improvementLabels',

  beforeDraw(chart) {
    const { ctx, scales: { x: xScale, y: yScale } } = chart;
    const MAX_TEXT_W = 110;

    const areaBounds = chart.chartArea;

    // Pass ALL progression points to the algorithm so it knows about every
    // line segment and data point.  Non-labeled points get 0×0 label size.
    const allX = [];
    const allY = [];
    const allW = [];
    const allH = [];
    const labelIndices = [];  // indices into the all* arrays that have real labels
    const labels = [];

    ctx.save();
    ctx.font = '11px sans-serif';

    let dataIdx = 0;
    for (const p of currentProgression) {
      if (p.score === null) continue;
      const px = xScale.getPixelForValue(p.iter);
      const py = yScale.getPixelForValue(p.score);
      allX.push(px);
      allY.push(py);

      const hasLabel = !p.isRoot && p.editSummary;
      if (hasLabel) {
        const lines = wrapText(ctx, p.editSummary, MAX_TEXT_W);
        const boxW = Math.min(Math.max(...lines.map((l) => ctx.measureText(l).width)), MAX_TEXT_W) + 12;
        const boxH = lines.length * 15 + (lines.length - 1) * 3 + 10;
        allW.push(boxW);
        allH.push(boxH);
        labelIndices.push(allX.length - 1);
        labels.push({ idx: dataIdx, iter: p.iter, lines, px, py, score: p.score });
        dataIdx++;
      } else {
        allW.push(0);
        allH.push(0);
      }
    }
    ctx.restore();

    if (labels.length === 0) {
      _labelBounds = [];
      _labelPositions = [];
      return;
    }

    const centers = placeLabels(areaBounds, allX, allY, allW, allH);

    for (let i = 0; i < labels.length; i++) {
      const ai = labelIndices[i];
      labels[i].boxX = centers[ai][0] - allW[ai] / 2;
      labels[i].boxY = centers[ai][1] - allH[ai] / 2;
      labels[i].boxW = allW[ai];
      labels[i].boxH = allH[ai];
    }

    _labelBounds = labels.map(l => ({ x: l.boxX, y: l.boxY, w: l.boxW, h: l.boxH, idx: l.idx }));
    _labelPositions = labels;
  },

  afterDraw(chart) {
    const { ctx } = chart;
    const hoveredIdx = labelPlugin._hoveredIdx;
    const labels = _labelPositions;

    // Draw leader line from data point to label box.
    function drawLeader(px, py, bx, by, bw, bh, alpha) {
      ctx.save();
      ctx.strokeStyle = `rgba(249, 168, 212, ${alpha * 0.5})`;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      const cx = bx + bw / 2;
      const cy = by + bh / 2;
      let ex, ey;
      if (px < bx) { ex = bx; ey = by + bh / 2; }
      else if (px > bx + bw) { ex = bx + bw; ey = by + bh / 2; }
      else if (py < by) { ex = cx; ey = by; }
      else { ex = cx; ey = by + bh; }
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // First pass: draw all non-hovered labels.
    for (const l of labels) {
      if (l.idx === hoveredIdx) continue;
      drawLeader(l.px, l.py, l.boxX, l.boxY, l.boxW, l.boxH, 0.8);
      _drawLabelBox(ctx, l.boxX, l.boxY, l.boxW, l.boxH, false);

      ctx.save();
      ctx.font = '11px sans-serif';
      ctx.fillStyle = 'rgba(249, 168, 212, 0.95)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const textCenterX = l.boxX + l.boxW / 2;
      const textBlockH = l.lines.length * 16;
      const textTopY = l.boxY + (l.boxH - textBlockH) / 2 + 8;
      l.lines.forEach((line, i) => {
        ctx.fillText(line, textCenterX, textTopY + i * 16);
      });
      ctx.restore();
    }

    // Second pass: draw hovered label last so it's always on top.
    if (hoveredIdx !== null) {
      const l = labels.find((l) => l.idx === hoveredIdx);
      if (l) {
        drawLeader(l.px, l.py, l.boxX, l.boxY, l.boxW, l.boxH, 1);
        _drawLabelBox(ctx, l.boxX, l.boxY, l.boxW, l.boxH, true);

        ctx.save();
        ctx.font = '11px sans-serif';
        ctx.fillStyle = 'rgba(249, 168, 212, 0.95)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const textCenterX = l.boxX + l.boxW / 2;
        const textBlockH = l.lines.length * 16;
        const textTopY = l.boxY + (l.boxH - textBlockH) / 2 + 8;
        l.lines.forEach((line, i) => {
          ctx.fillText(line, textCenterX, textTopY + i * 16);
        });
        ctx.restore();
      }
    }
  },

  afterEvent(chart, args) {
    const { event } = args;
    const { x, y } = event;

    // Check label box hit first (they're on top).
    let labelHit = null;
    for (const b of _labelBounds) {
      if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) {
        labelHit = b.idx;
        break;
      }
    }

    // Hover tracking for labels.
    if (labelHit !== labelPlugin._hoveredIdx) {
      labelPlugin._hoveredIdx = labelHit;
      chart.canvas.style.cursor = labelHit !== null ? 'pointer' : '';
      chart.draw();
    }

    // Click handling.
    if (event.type === 'click') {
      // Label box click — show node details.
      if (labelHit !== null) {
        const p = currentProgression[labelHit];
        if (p && p.isRoot) {
          const root = StateLoader.getNodes().find((n) => n.stage === 'root');
          if (root && typeof showNodeDetailToBottom === 'function') showNodeDetailToBottom(root);
        } else if (p) {
          showNodeForIteration(p.iter);
        }
        return;
      }

      // Scatter point click — check all datasets.
      const { data: { datasets } } = chart;
      for (let di = 0; di < datasets.length; di++) {
        const ds = datasets[di];
        if (!ds.showLine) {
          for (let pi = 0; pi < ds.data.length; pi++) {
            const pt = ds.data[pi];
            const px = chart.scales.x.getPixelForValue(pt.x);
            const py = chart.scales.y.getPixelForValue(pt.y);
            const hitR = 10;
            const dx = x - px, dy = y - py;
            if (dx * dx + dy * dy <= hitR * hitR) {
              const p = currentProgression[pi];
              if (p && p.isRoot) {
                const root = StateLoader.getNodes().find((n) => n.stage === 'root');
                if (root && typeof showNodeDetailToBottom === 'function') showNodeDetailToBottom(root);
              } else {
                showNodeForIteration(pt.x);
              }
              return;
            }
          }
        }
      }
    }
  },
};
labelPlugin._hoveredIdx = null;

function renderScoreChart() {
  currentProgression = StateLoader.getProgressionImprovements();
  if (currentProgression.length === 0) return;

  _chartImprovements = currentProgression.filter((p) => !p.isRoot);
  _chartDiscarded = [];

  const pathValues = currentProgression.map((p) => p.score).filter((v) => v !== null);
  const dataMin = Math.min(...pathValues);
  const dataMax = Math.max(...pathValues);
  const range = dataMax - dataMin;

  // Add symmetric padding for visual breathing room (8% on each side)
  // Labels use canvasBounds and don't need axis padding
  const pad = range * 0.08 || 0.001;
  const yMin = dataMin - pad;
  const yMax = dataMax + pad;

  // Step line: connect all improvement nodes in order.
  const stepData = currentProgression.map((p) => ({ x: p.iter, y: p.score }));

  if (scoreChart) scoreChart.destroy();

  const canvas = document.getElementById('score-chart');
  const ctx = canvas.getContext('2d');

  // Solid color for the step line.
  const lineColor = 'rgba(200, 170, 220, 0.9)';

  scoreChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Improvements',
          data: currentProgression.map((p) => ({ x: p.iter, y: p.score })),
          pointRadius: 6,
          pointHoverRadius: 8,
          pointBackgroundColor: currentProgression.map((p) =>
            p.isRoot ? '#ffb347' :
              p.score === StateLoader.getBestScore() ? '#80f0b0' :
              'rgba(255, 176, 224, 0.9)'
          ),
          pointBorderColor: 'rgba(255,255,255,0.25)',
          pointBorderWidth: 1,
          showLine: false,
        },
        {
          label: 'Path',
          data: stepData,
          showLine: true,
          stepped: 'before',
          borderColor: lineColor,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 0,
          fill: false,
          tension: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600, easing: 'easeOutQuart' },
      interaction: { intersect: true, mode: 'nearest' },
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: `${_chartImprovements.length} improvements along the path`,
          color: '#f0e0ff',
          font: { size: 15, weight: 'normal' },
          padding: { bottom: 12 },
        },
        tooltip: { enabled: false },
      },
      scales: {
        x: {
          type: 'linear',
          min: -2,
          ticks: {
            color: '#f0e0ff',
            font: { size: 14 },
            maxTicksLimit: 20,
            callback: (value) => value >= 0 ? value : '',
          },
          title: { display: true, text: 'Iteration', color: '#f0e0ff', font: { size: 15 } },
          grid: { color: 'rgba(80, 60, 110, 0.25)', drawBorder: false },
          border: { display: false },
        },
        y: {
          min: yMin,
          max: yMax,
          ticks: { color: '#f0e0ff', font: { size: 14 }, padding: 8 },
          title: { display: true, text: 'Score', color: '#f0e0ff', font: { size: 15 } },
          grid: { color: 'rgba(80, 60, 110, 0.25)', drawBorder: false },
          border: { display: false },
        },
      },
    },
    plugins: [labelPlugin],
  });
}
