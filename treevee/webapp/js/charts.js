let scoreChart = null;
let currentProgression = [];

function renderScoreChart() {
  currentProgression = StateLoader.getBestProgression();

  if (currentProgression.length === 0) {
    return;
  }

  const allScores = currentProgression.map((p) => p.runningBest).filter((s) => s !== null);
  const yMin = allScores.length > 0 ? Math.min(...allScores) * 0.98 : 0;
  const yMax = allScores.length > 0 ? Math.max(...allScores) * 1.02 : 1;

  if (scoreChart) scoreChart.destroy();

  const canvas = document.getElementById('score-chart');
  const ctx = canvas.getContext('2d');

  // Create gradient fill — kawaii pastel pink
  const gradient = ctx.createLinearGradient(0, 0, 0, 300);
  gradient.addColorStop(0, 'rgba(249, 168, 212, 0.25)');
  gradient.addColorStop(0.5, 'rgba(196, 168, 224, 0.08)');
  gradient.addColorStop(1, 'rgba(196, 168, 224, 0.0)');

  // Create gradient for the line — pink to lavender
  const lineGradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
  lineGradient.addColorStop(0, 'rgba(249, 168, 212, 0.5)');
  lineGradient.addColorStop(0.5, 'rgba(196, 168, 224, 0.85)');
  lineGradient.addColorStop(1, 'rgba(135, 206, 235, 1)');

  scoreChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: 'Best Score',
        data: currentProgression.map((p) => ({
          x: p.datetime ? new Date(p.datetime).getTime() : null,
          y: p.runningBest,
        })),
        borderColor: lineGradient,
        borderWidth: 2.5,
        pointRadius: currentProgression.map((p) => p.isImprovement ? 5 : 2),
        pointHoverRadius: 7,
        pointBackgroundColor: currentProgression.map((p) => {
          if (!p.isImprovement) return 'rgba(224, 196, 240, 0.4)';
          if (p.runningBest === StateLoader.getBestScore()) return '#80f0b0';
          return 'rgba(255, 176, 224, 0.8)';
        }),
        pointBorderColor: 'transparent',
        pointBorderWidth: 0,
        pointRadiusHover: currentProgression.map((p) => p.isImprovement ? 7 : 4),
        pointStyle: currentProgression.map((p) => p.isImprovement ? 'circle' : 'diamond'),
        fill: true,
        tension: 0.3,
        spanGaps: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 800,
        easing: 'easeOutQuart',
      },
      events: ['mousemove', 'mouseout', 'click'],
      interaction: {
        intersect: false,
        mode: 'index',
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: true,
          backgroundColor: 'rgba(35, 21, 53, 0.95)',
          titleColor: '#ffffff',
          bodyColor: '#ffffff',
          borderColor: 'rgba(249, 168, 212, 0.3)',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 14,
          titleFont: { size: 13, weight: '600' },
          bodyFont: { size: 12 },
          displayColors: false,
          callbacks: {
            title: function(items) {
              if (!items || items.length === 0) return 'No data~ (｡•́︿•̀｡)';
              const idx = items[0].dataIndex;
              const p = currentProgression[idx];
              if (!p) return 'Iteration ? ✨';
              if (p.datetime) {
                const d = new Date(p.datetime);
                return d.toLocaleString() + ' ✨';
              }
              return `Iteration ${p.iter} ✨`;
            },
            label: function(item) {
              const p = currentProgression[item.dataIndex];
              if (!p) return '';
              const emoji = p.isImprovement ? '🎀' : '·';
              return `${emoji} Best Score: ${p.runningBest !== null ? p.runningBest.toFixed(6) : 'N/A'}`;
            },
            afterLabel: function(item) {
              const p = currentProgression[item.dataIndex];
              if (!p || !p.isImprovement) return null;
              const lines = [];
              if (p.editSummary) lines.push(`Change: ${p.editSummary}`);
              if (p.score !== p.runningBest) lines.push(`Actual score: ${p.score.toFixed(6)}`);
              lines.push('');
              lines.push('Keep going~ (◕‿◕)');
              return lines;
            },
          },
        },
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'hour',
            tooltipFormat: 'yyyy-MM-dd HH:mm',
            displayFormats: {
              hour: 'HH:mm',
              day: 'MMM dd HH:mm',
            },
          },
          grid: { color: 'rgba(61, 42, 92, 0.4)', drawBorder: false },
          border: { display: false },
          ticks: {
            color: '#ddbdf0',
            font: { size: 10 },
            maxRotation: 45,
            autoSkip: true,
            maxTicksLimit: 30,
          },
        },
        y: {
          grid: { color: 'rgba(61, 42, 92, 0.4)', drawBorder: false },
          border: { display: false },
          ticks: {
            color: '#ddbdf0',
            font: { size: 11 },
            padding: 8,
          },
          min: yMin,
          max: yMax,
        },
      },
    },
  });
}
