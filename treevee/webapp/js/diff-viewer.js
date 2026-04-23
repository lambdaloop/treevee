function initDiffClose() {
  const closeBtn = document.getElementById('close-diff');
  const diffSection = document.getElementById('diff-section');
  if (closeBtn && diffSection) {
    closeBtn.addEventListener('click', () => {
      diffSection.style.display = 'none';
      closeBtn.style.display = 'none';
    });
  }
}

function renderDiffHTML(diffText) {
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

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
