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
    // Big glow behind the box to make it visually pop forward.
    ctx.save();
    ctx.shadowColor = 'rgba(249, 168, 212, 0.9)';
    ctx.shadowBlur = 18;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.fillStyle = 'rgba(35, 21, 53, 1.0)';
    ctx.beginPath();
    ctx.roundRect(boxX - 4, boxY - 4, boxW + 8, boxH + 8, 6);
    ctx.fill();
    ctx.restore();

    // Bright pink border.
    ctx.strokeStyle = 'rgba(249, 168, 212, 1.0)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(boxX - 4, boxY - 4, boxW + 8, boxH + 8, 6);
    ctx.stroke();

    // Fully opaque main box.
    ctx.fillStyle = 'rgba(35, 21, 53, 1.0)';
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 4);
    ctx.fill();

    ctx.strokeStyle = 'rgba(249, 168, 212, 1.0)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  } else {
    // Non-hovered: fully opaque but darker background.
    ctx.fillStyle = 'rgba(25, 15, 40, 1.0)';
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 4);
    ctx.fill();

    ctx.strokeStyle = 'rgba(249, 168, 212, 0.4)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

// AABB overlap test for two label boxes.
function _boxesOverlap(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

// Check if a box overlaps a point (with a small radius).
function _boxOverlapsPoint(box, px, py, radius) {
  // Expand the box by the radius and test AABB against a point-expanded box.
  const expanded = { x: box.x - radius, y: box.y - radius, w: box.w + 2 * radius, h: box.h + 2 * radius };
  return expanded.x <= px && expanded.x + expanded.w >= px && expanded.y <= py && expanded.y + expanded.h >= py;
}

// Shift a label box away from an obstacle (another box or a point).
function _shiftAway(label, obstacle, maximize, bounds) {
  const labelBox = { x: label.boxX, y: label.boxY, w: label.boxW, h: label.boxH };
  if (!_boxesOverlap(labelBox, obstacle)) return false;

  // Push label away from obstacle based on relative position, not maximize direction.
  const labelCenterY = label.boxY + label.boxH / 2;
  const obstacleCenterY = obstacle.y + obstacle.h / 2;

  let shiftedY;
  if (labelCenterY <= obstacleCenterY) {
    // Label is above obstacle — push it further up.
    shiftedY = obstacle.y - label.boxH - 8;
  } else {
    // Label is below obstacle — push it further down.
    shiftedY = obstacle.y + obstacle.h + 8;
  }
  const clampedY = Math.max(bounds.top, Math.min(shiftedY, bounds.bottom - label.boxH));
  label.boxY = clampedY;
  return true;
}

// Boxed edit-summary labels near each improvement point, with leader lines.
const labelPlugin = {
  id: 'improvementLabels',

  beforeDraw(chart) {
    const { ctx, scales: { x: xScale, y: yScale }, chartArea } = chart;
    const maximize = StateLoader.getMaximize();
    const OFFSET = 12;
    const MAX_TEXT_W = 110;

    // Use CSS-pixel dimensions (chart.width/height), not physical pixels
    // (chart.canvas.width/height), since Chart.js draws in CSS-pixel space.
    const CANVAS_MARGIN = 10;
    const BOTTOM_MARGIN = 30;
    const canvasBounds = {
      top: CANVAS_MARGIN,
      bottom: chart.height - BOTTOM_MARGIN,
      left: CANVAS_MARGIN,
      right: chart.width - CANVAS_MARGIN
    };

    // Collect all label info.
    const labels = [];
    ctx.save();
    ctx.font = '9px sans-serif';

    let labelIndex = 0;
    for (const p of currentProgression) {
      if (p.isRoot || !p.editSummary || p.score === null) continue;
      const px = xScale.getPixelForValue(p.iter);
      const py = yScale.getPixelForValue(p.score);

      const lines = wrapText(ctx, p.editSummary, MAX_TEXT_W);
      const boxW = Math.min(Math.max(...lines.map((l) => ctx.measureText(l).width)), MAX_TEXT_W) + 12;
      const boxH = lines.length * 12 + (lines.length - 1) * 2 + 10;

      // Smart horizontal positioning: use the side with more available space.
      // Add slight horizontal offset variation to spread labels out.
      const horizontalVariation = (labelIndex % 3 - 1) * 15;
      const spaceLeft = px - canvasBounds.left;
      const spaceRight = canvasBounds.right - px;
      let boxX;
      if (spaceRight >= boxW + OFFSET) {
        // Enough space on the right - position to the right of the point.
        boxX = px + OFFSET + Math.max(0, horizontalVariation);
      } else if (spaceLeft >= boxW + OFFSET) {
        // Not enough space on right but enough on left - position to the left.
        boxX = px - boxW - OFFSET + Math.min(0, horizontalVariation);
      } else if (spaceRight > spaceLeft) {
        // Neither side has full space, use the side with more room (right-aligned to edge).
        boxX = canvasBounds.right - boxW;
      } else {
        // Use left side (left-aligned to edge).
        boxX = canvasBounds.left;
      }

      // Prefer labels above the line when minimizing, below when maximizing.
      const spaceAbove = py - canvasBounds.top;
      const spaceBelow = canvasBounds.bottom - py;
      const verticalVariation = (labelIndex % 3 - 1) * 10;
      const preferBelow = maximize;

      let boxY;
      if (preferBelow) {
        if (spaceBelow >= boxH + OFFSET) {
          boxY = py + OFFSET + verticalVariation;
        } else if (spaceAbove >= boxH + OFFSET) {
          boxY = py - OFFSET - boxH + verticalVariation;
        } else {
          boxY = py + OFFSET + verticalVariation;
        }
      } else {
        if (spaceAbove >= boxH + OFFSET) {
          boxY = py - OFFSET - boxH + verticalVariation;
        } else if (spaceBelow >= boxH + OFFSET) {
          boxY = py + OFFSET + verticalVariation;
        } else {
          boxY = py - OFFSET - boxH + verticalVariation;
        }
      }

      const idx = currentProgression.indexOf(p);
      labels.push({ idx, boxX, boxY, boxW, boxH, lines, px, py });
      labelIndex++;
    }
    ctx.restore();

    // --- Phase 1: All-pairs label-label overlap resolution (iterative) ---
    const MIN_GAP = 8;
    for (let iter = 0; iter < 20; iter++) {
      let hadOverlap = false;
      for (let i = 0; i < labels.length; i++) {
        for (let j = i + 1; j < labels.length; j++) {
          const a = { x: labels[i].boxX, y: labels[i].boxY, w: labels[i].boxW, h: labels[i].boxH };
          const b = { x: labels[j].boxX, y: labels[j].boxY, w: labels[j].boxW, h: labels[j].boxH };
          if (_boxesOverlap(a, b)) {
            hadOverlap = true;
            // Push labels apart based on relative position, not maximize direction.
            const aCenterY = labels[i].boxY + labels[i].boxH / 2;
            const bCenterY = labels[j].boxY + labels[j].boxH / 2;
            const overlapAmount = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
            const halfShift = (overlapAmount + MIN_GAP) / 2;

            if (aCenterY <= bCenterY) {
              // a is above b — push a up, push b down
              labels[i].boxY -= halfShift;
              labels[j].boxY += halfShift;
            } else {
              // a is below b — push a down, push b up
              labels[i].boxY += halfShift;
              labels[j].boxY -= halfShift;
            }
            // Clamp both to canvas bounds.
            labels[i].boxX = Math.max(canvasBounds.left, Math.min(labels[i].boxX, canvasBounds.right - labels[i].boxW));
            labels[i].boxY = Math.max(canvasBounds.top, Math.min(labels[i].boxY, canvasBounds.bottom - labels[i].boxH));
            labels[j].boxX = Math.max(canvasBounds.left, Math.min(labels[j].boxX, canvasBounds.right - labels[j].boxW));
            labels[j].boxY = Math.max(canvasBounds.top, Math.min(labels[j].boxY, canvasBounds.bottom - labels[j].boxH));
          }
        }
      }
      if (!hadOverlap) break;
    }

    // --- Phase 2: Avoid data points (path + discarded) ---
    const pointPositions = [];
    for (const p of currentProgression) {
      if (p.score !== null) {
        pointPositions.push({ x: xScale.getPixelForValue(p.iter), y: yScale.getPixelForValue(p.score) });
      }
    }
    for (const n of _chartDiscarded) {
      if (n.score !== null) {
        pointPositions.push({ x: xScale.getPixelForValue(n.step), y: yScale.getPixelForValue(n.score) });
      }
    }
    for (const l of labels) {
      for (const pt of pointPositions) {
        if (_boxOverlapsPoint(l, pt.x, pt.y, 10)) {
          _shiftAway(l, { x: pt.x - 10, y: pt.y - 10, w: 20, h: 20 }, maximize, canvasBounds);
        }
      }
    }

    // --- Phase 3: Avoid step line segments (stepped: 'before') ---
    for (let i = 0; i < currentProgression.length - 1; i++) {
      const p0 = currentProgression[i];
      const p1 = currentProgression[i + 1];
      if (p0.score === null || p1.score === null) continue;
      const x0 = xScale.getPixelForValue(p0.iter);
      const y0 = yScale.getPixelForValue(p0.score);
      const x1 = xScale.getPixelForValue(p1.iter);
      const y1 = yScale.getPixelForValue(p1.score);

      // Vertical segment from (x0, y0) to (x0, y1).
      const vSeg = {
        x: x0 - 4,
        y: Math.min(y0, y1) - 2,
        w: 8,
        h: Math.abs(y1 - y0) + 4,
      };
      // Horizontal segment from (x0, y1) to (x1, y1).
      const hSeg = {
        x: Math.min(x0, x1) - 2,
        y: y1 - 4,
        w: Math.abs(x1 - x0) + 4,
        h: 8,
      };

      for (const l of labels) {
        if (_boxesOverlap({ x: l.boxX, y: l.boxY, w: l.boxW, h: l.boxH }, vSeg)) {
          _shiftAway(l, vSeg, maximize, canvasBounds);
        }
        if (_boxesOverlap({ x: l.boxX, y: l.boxY, w: l.boxW, h: l.boxH }, hSeg)) {
          _shiftAway(l, hSeg, maximize, canvasBounds);
        }
      }
    }

    // Final clamp: ensure all labels are within canvas bounds after all adjustments.
    for (const l of labels) {
      l.boxX = Math.max(canvasBounds.left, Math.min(l.boxX, canvasBounds.right - l.boxW));
      l.boxY = Math.max(canvasBounds.top, Math.min(l.boxY, canvasBounds.bottom - l.boxH));
    }

    _labelBounds = labels.map((l) => ({ x: l.boxX, y: l.boxY, w: l.boxW, h: l.boxH, idx: l.idx }));
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
      ctx.font = '9px sans-serif';
      ctx.fillStyle = 'rgba(249, 168, 212, 0.95)';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      l.lines.forEach((line, i) => {
        ctx.fillText(line, l.boxX + 6, l.boxY + 5 + i * 14 + 6);
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
        ctx.font = '9px sans-serif';
        ctx.fillStyle = 'rgba(249, 168, 212, 0.95)';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        l.lines.forEach((line, i) => {
          ctx.fillText(line, l.boxX + 6, l.boxY + 5 + i * 14 + 6);
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

  // Pink → lavender → sky-blue gradient for the step line.
  const lineGradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
  lineGradient.addColorStop(0, 'rgba(249, 168, 212, 0.9)');
  lineGradient.addColorStop(0.5, 'rgba(196, 168, 224, 0.95)');
  lineGradient.addColorStop(1, 'rgba(135, 206, 235, 1)');

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
          borderColor: lineGradient,
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
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            color: '#f0e0ff',
            usePointStyle: true,
            pointStyleWidth: 10,
            padding: 15,
            font: { size: 11 },
          },
        },
        title: {
          display: true,
          text: `${_chartImprovements.length} improvements along the path`,
          color: '#f0e0ff',
          font: { size: 13, weight: 'normal' },
          padding: { bottom: 12 },
        },
        tooltip: { enabled: false },
      },
      scales: {
        x: {
          type: 'linear',
          min: -1,
          ticks: {
            color: '#f0e0ff',
            font: { size: 10 },
            maxTicksLimit: 20,
          },
          grid: { color: 'rgba(80, 60, 110, 0.25)', drawBorder: false },
          border: { display: false },
        },
        y: {
          min: yMin,
          max: yMax,
          ticks: { color: '#f0e0ff', font: { size: 11 }, padding: 8 },
          grid: { color: 'rgba(80, 60, 110, 0.25)', drawBorder: false },
          border: { display: false },
        },
      },
    },
    plugins: [labelPlugin],
  });
}
