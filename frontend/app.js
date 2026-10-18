/* 数据分析 Agent 控制台 —— 纯原生 JS，无外部依赖 */

const state = {
  datasets: [],
  currentId: null,
  profile: null,
  conversationId: null,
  busy: false,
};

const $ = (id) => document.getElementById(id);

/* ----------------------------- 基础工具 ----------------------------- */
function escapeHtml(text) {
  return String(text == null ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function inlineMd(text) {
  let s = escapeHtml(text);
  s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, '<img alt="$1" src="$2">');
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  return s;
}

function splitRow(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
}

function renderMarkdown(md) {
  const lines = String(md || '').split('\n');
  let html = '';
  let i = 0;
  let listType = null;
  const closeList = () => { if (listType) { html += `</${listType}>`; listType = null; } };

  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      closeList();
      const buf = [];
      i += 1;
      while (i < lines.length && !/^\s*```/.test(lines[i])) { buf.push(lines[i]); i += 1; }
      i += 1;
      html += `<pre><code>${escapeHtml(buf.join('\n'))}</code></pre>`;
      continue;
    }
    if (/^\s*$/.test(line)) { closeList(); i += 1; continue; }
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      closeList();
      const head = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(splitRow(lines[i])); i += 1; }
      html += '<table><thead><tr>' + head.map((c) => `<th>${inlineMd(c)}</th>`).join('') + '</tr></thead><tbody>';
      html += rows.map((r) => '<tr>' + r.map((c) => `<td>${inlineMd(c)}</td>`).join('') + '</tr>').join('');
      html += '</tbody></table>';
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length + 1, 6);
      html += `<h${level}>${inlineMd(heading[2])}</h${level}>`;
      i += 1;
      continue;
    }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { closeList(); html += '<hr>'; i += 1; continue; }
    if (/^\s*>\s?/.test(line)) {
      closeList();
      html += `<blockquote>${inlineMd(line.replace(/^\s*>\s?/, ''))}</blockquote>`;
      i += 1;
      continue;
    }
    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      const type = ul ? 'ul' : 'ol';
      if (listType !== type) { closeList(); html += `<${type}>`; listType = type; }
      html += `<li>${inlineMd((ul || ol)[1])}</li>`;
      i += 1;
      continue;
    }
    closeList();
    html += `<p>${inlineMd(line)}</p>`;
    i += 1;
  }
  closeList();
  return html;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (err) { /* 保持默认信息 */ }
    throw new Error(detail);
  }
  const type = res.headers.get('content-type') || '';
  return type.includes('json') ? res.json() : res.text();
}

/* ----------------------------- 健康检查 ----------------------------- */
async function checkHealth() {
  const badge = $('health-badge');
  try {
    const data = await api('/api/health');
    badge.textContent = `${data.llm_model} · ${data.tool_count} 个工具 · ${data.checkpoint.backend}`;
    badge.className = data.llm_configured ? 'badge ok' : 'badge bad';
    if (!data.llm_configured) badge.textContent = '未配置 LLM_API_KEY';
  } catch (err) {
    badge.textContent = '服务不可用';
    badge.className = 'badge bad';
  }
}

/* ----------------------------- 数据集 ----------------------------- */
async function loadDatasets() {
  const data = await api('/api/datasets/');
  state.datasets = data.items || [];
  renderDatasetList();
}

function renderDatasetList() {
  const list = $('dataset-list');
  list.innerHTML = '';
  if (!state.datasets.length) {
    list.innerHTML = '<li class="empty">还没有数据，先上传一个文件</li>';
    return;
  }
  state.datasets.forEach((ds) => {
    const li = document.createElement('li');
    if (ds.id === state.currentId) li.classList.add('active');
    li.innerHTML = `<span class="name">${escapeHtml(ds.name)}</span>
      <span class="meta">${ds.rows}×${ds.cols} · ${ds.quality_score ?? '-'}分</span>`;
    li.onclick = () => selectDataset(ds.id);
    list.appendChild(li);
  });
}

async function uploadFile(file) {
  const status = $('upload-status');
  status.className = 'status';
  status.textContent = `正在上传并解析 ${file.name} …`;
  const form = new FormData();
  form.append('file', file);
  try {
    const data = await api('/api/datasets/upload', { method: 'POST', body: form });
    status.className = 'status ok';
    status.textContent = `已解析：${data.dataset.rows} 行 × ${data.dataset.cols} 列，质量评分 ${data.dataset.quality_score}`;
    await loadDatasets();
    await selectDataset(data.dataset.id, data.profile, data.warnings);
  } catch (err) {
    status.className = 'status err';
    status.textContent = `上传失败：${err.message}`;
  }
}

async function selectDataset(id, presetProfile, presetWarnings) {
  state.currentId = id;
  state.profile = presetProfile || null;
  state.conversationId = null;
  renderDatasetList();
  const card = $('dataset-card');
  card.hidden = false;
  await applyProfile(presetProfile || (await api(`/api/datasets/${id}/profile`)), presetWarnings);
  await attachLatestConversation(id);
}

async function applyProfile(profile, warnings) {
  if (!profile) return;
  state.profile = profile;
  const quality = profile.quality || {};
  $('dataset-metrics').innerHTML = [
    ['行数', profile.rows], ['列数', profile.cols], ['质量评分', quality.score],
  ].map(([k, v]) => `<div class="metric"><div class="k">${k}</div><div class="v">${v ?? '-'}</div></div>`).join('');

  const box = $('dataset-warnings');
  const messages = [...(warnings || [])];
  if (quality.total_missing) messages.push(`缺失单元格 ${quality.total_missing} 个，重复行 ${quality.duplicate_rows} 行`);
  box.innerHTML = messages.map((m) => `<div>${escapeHtml(m)}</div>`).join('');

  const cols = profile.columns || [];
  $('column-table').innerHTML = '<table><thead><tr><th>字段</th><th>语义类型</th><th>缺失</th><th>缺失率</th><th>唯一值</th><th>示例</th></tr></thead><tbody>'
    + cols.map((c) => `<tr><td>${escapeHtml(c.name)}</td><td>${escapeHtml(c.semantic_type)}</td>`
      + `<td>${c.missing}</td><td>${(100 * (c.missing_ratio || 0)).toFixed(1)}%</td><td>${c.unique}</td>`
      + `<td>${escapeHtml((c.sample_values || []).slice(0, 3).join(' / '))}</td></tr>`).join('')
    + '</tbody></table>';

  $('quality-issues').innerHTML = (quality.issues || []).map((i) => `
    <div class="issue ${escapeHtml(i.level)}">
      <div class="t">[${escapeHtml(i.level)}] ${escapeHtml(i.scope || '')} — ${escapeHtml(i.message || '')}</div>
      <div class="d">建议：${escapeHtml(i.suggestion || '')}</div>
    </div>`).join('') || '<div class="small">未发现明显问题</div>';
}

async function refreshProfile() {
  if (!state.currentId) return;
  const profile = await api(`/api/datasets/${state.currentId}/profile?refresh=true`);
  await applyProfile(profile);
}

async function showPreview() {
  if (!state.currentId) return;
  const data = await api(`/api/datasets/${state.currentId}/preview?limit=50`);
  const head = data.columns.map((c) => `<th>${escapeHtml(c)}<br><span class="small">${escapeHtml(data.dtypes[c] || '')}</span></th>`).join('');
  const rows = data.rows.map((row) => '<tr>' + data.columns.map((c) => {
    const v = row[c];
    return `<td>${v === null || v === undefined ? '<span class="small">-</span>' : escapeHtml(String(v))}</td>`;
  }).join('') + '</tr>').join('');
  $('preview-body').innerHTML = `<div class="small">共 ${data.total_rows} 行，显示前 ${data.rows.length} 行</div>
    <table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
  openModal('modal-preview');
}

async function resetData() {
  if (!state.currentId) return;
  const data = await api(`/api/datasets/${state.currentId}/reset`, { method: 'POST' });
  await applyProfile(data.dataset.profile || (await api(`/api/datasets/${state.currentId}/profile?refresh=true`)));
  $('upload-status').className = 'status ok';
  $('upload-status').textContent = `已恢复到原始数据（${data.rows} 行 × ${data.cols} 列）`;
}

/* ----------------------------- 会话 ----------------------------- */
async function attachLatestConversation(datasetId) {
  clearMessages();
  try {
    const data = await api(`/api/chat/conversations?dataset_id=${encodeURIComponent(datasetId)}`);
    const first = (data.items || [])[0];
    if (!first) return;
    state.conversationId = first.id;
    const history = await api(`/api/chat/history/${encodeURIComponent(first.id)}`);
    (history.messages || []).forEach((m) => {
      if (m.role === 'user') addMessage('user', escapeHtml(m.content));
      else if (m.role === 'assistant' && m.content) addMessage('assistant', renderMarkdown(m.content));
    });
    if ((history.messages || []).length) {
      addMessage('system', `<span class="small">已从本地 Checkpoint 恢复该数据集最近的会话（${history.messages.length} 条消息）</span>`);
    }
  } catch (err) { /* 历史不是关键路径，忽略 */ }
}

function clearMessages() {
  $('messages').innerHTML = '';
}

function addMessage(role, html) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const label = role === 'user' ? '你' : (role === 'assistant' ? 'Agent' : '系统');
  wrap.innerHTML = `<span class="role">${label}</span><div class="bubble">${html}</div>`;
  $('messages').appendChild(wrap);
  $('messages').scrollTop = $('messages').scrollHeight;
  return wrap.querySelector('.bubble');
}

/* ----------------------------- 对话（SSE 流式） ----------------------------- */
function createAssistantShell(question) {
  const wrap = document.createElement('div');
  wrap.className = 'msg assistant';
  wrap.innerHTML = `
    <span class="role">Agent · 分析中</span>
    <div class="bubble">
      <div class="status thinking"><span class="dot"></span><span class="dot"></span><span class="dot"></span>
        <span class="status-text">正在理解你的问题…</span></div>
      <div class="answer"></div>
      <div class="artifacts"></div>
    </div>
    <details class="trace" open>
      <summary>执行过程</summary>
      <div class="steps"></div>
    </details>`;
  wrap.dataset.question = question;
  $('messages').appendChild(wrap);
  $('messages').scrollTop = $('messages').scrollHeight;
  return {
    root: wrap,
    status: wrap.querySelector('.status'),
    statusText: wrap.querySelector('.status-text'),
    answer: wrap.querySelector('.answer'),
    artifacts: wrap.querySelector('.artifacts'),
    trace: wrap.querySelector('.trace'),
    traceSummary: wrap.querySelector('.trace summary'),
    steps: wrap.querySelector('.steps'),
    stepCount: 0,
    headline: '',
  };
}

function appendStep(shell, step) {
  shell.stepCount += 1;
  const node = document.createElement('div');
  node.className = 'step';
  let detail = '';
  if (step.tool_calls) {
    detail = step.tool_calls
      .map((c) => `<div class="detail">⚙ ${escapeHtml(c.name)}(${escapeHtml(JSON.stringify(c.args))})</div>`)
      .join('');
  }
  if (step.results) {
    detail += step.results.map((r) => `<div class="detail ${r.ok ? 'ok' : 'bad'}">${r.ok ? '✓' : '✗'} ${escapeHtml(r.tool)} — ${escapeHtml(r.summary || '')}</div>`).join('');
  }
  if (step.plan && step.plan.length) {
    detail += step.plan.map((p, idx) => `<div class="detail">${idx + 1}. ${escapeHtml(p)}</div>`).join('');
  }
  node.innerHTML = `<div class="head">${escapeHtml(step.label || step.node)}</div>${detail}`;
  shell.steps.appendChild(node);
  shell.traceSummary.textContent = `执行过程 · ${shell.stepCount} 步`;
  $('messages').scrollTop = $('messages').scrollHeight;
}

function renderArtifacts(shell, artifacts) {
  shell.artifacts.innerHTML = (artifacts || []).map((a) =>
    `<a class="chip" href="${a.url}" target="_blank" rel="noreferrer">📊 ${escapeHtml(a.title)}</a>`).join('');
}

async function sendMessage(message) {
  if (!state.currentId) { addMessage('system', '请先上传并选择一个数据集。'); return; }
  if (state.busy) return;
  const text = (message || '').trim();
  if (!text) return;

  state.busy = true;
  $('btn-send').disabled = true;
  $('btn-report').disabled = true;
  addMessage('user', escapeHtml(text));
  const shell = createAssistantShell(text);

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id: state.currentId, message: text, conversation_id: state.conversationId }),
    });
    if (!res.ok || !res.body) throw new Error(`请求失败 HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let final = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx = buffer.indexOf('\n\n');
      while (idx >= 0) {
        const raw = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 2);
        if (raw.startsWith('data:')) {
          let event = null;
          try { event = JSON.parse(raw.slice(5).trim()); } catch (err) { event = null; }
          if (event) {
            if (event.type === 'start') {
              state.conversationId = event.conversation_id;
              shell.statusText.textContent = '正在分析…';
            } else if (event.type === 'step') {
              if (event.label) shell.statusText.textContent = event.label + ' …';
              appendStep(shell, event);
              if (event.report_id) shell.reportId = event.report_id;
            } else if (event.type === 'final') {
              final = event;
            } else if (event.type === 'error') {
              throw new Error(event.message);
            }
          }
        }
        idx = buffer.indexOf('\n\n');
      }
    }

    shell.status.remove();
    if (final) {
      state.conversationId = final.conversation_id;
      shell.answer.innerHTML = renderMarkdown(final.answer || '（没有返回内容）');
      renderArtifacts(shell, final.artifacts);
      const role = shell.root.querySelector('.role');
      role.textContent = `Agent · 意图：${final.intent || '-'} · 工具调用 ${final.tool_rounds} 轮 · 证据 ${final.evidence.length} 条`;
      if (final.report_id) {
        const btn = document.createElement('button');
        btn.className = 'ghost';
        btn.textContent = '查看完整报告';
        btn.onclick = () => openReport(final.report_id);
        shell.answer.appendChild(btn);
      }
    }
  } catch (err) {
    shell.status.remove();
    shell.answer.innerHTML = `<span style="color:var(--danger)">分析失败：${escapeHtml(err.message)}</span>`;
  } finally {
    state.busy = false;
    $('btn-send').disabled = false;
    $('btn-report').disabled = false;
    $('messages').scrollTop = $('messages').scrollHeight;
  }
}

/* ----------------------------- 报告 ----------------------------- */
async function generateReport() {
  if (!state.currentId) { addMessage('system', '请先上传并选择一个数据集。'); return; }
  if (state.busy) return;
  state.busy = true;
  $('btn-report').disabled = true;
  $('btn-send').disabled = true;
  addMessage('user', '生成一份完整的数据分析报告');
  const shell = createAssistantShell('报告');
  shell.statusText.textContent = '正在执行完整分析流程并撰写报告…';

  try {
    const data = await api('/api/reports/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id: state.currentId, conversation_id: state.conversationId }),
    });
    state.conversationId = data.conversation_id;
    shell.status.remove();
    shell.answer.innerHTML = renderMarkdown(data.answer || '报告已生成');
    renderArtifacts(shell, data.artifacts);
    const btn = document.createElement('button');
    btn.className = 'ghost';
    btn.textContent = '查看完整报告';
    btn.onclick = () => openReport(data.report_id);
    shell.answer.appendChild(btn);
  } catch (err) {
    shell.status.remove();
    shell.answer.innerHTML = `<span style="color:var(--danger)">报告生成失败：${escapeHtml(err.message)}</span>`;
  } finally {
    state.busy = false;
    $('btn-report').disabled = false;
    $('btn-send').disabled = false;
  }
}

async function openReport(reportId) {
  const frame = $('report-frame');
  $('report-title').textContent = '报告加载中…';
  frame.src = 'about:blank';
  openModal('modal-report');
  try {
    const data = await api(`/api/reports/${reportId}`);
    $('report-title').textContent = data.title || '分析报告';
    // 内嵌预览用 inline，点「下载 HTML」时才带 download=1 变成附件
    frame.src = `/api/reports/${reportId}/download?fmt=html`;
    $('report-download').href = `/api/reports/${reportId}/download?fmt=html&download=1`;
  } catch (err) {
    $('report-title').textContent = `报告加载失败：${err.message}`;
  }
}

/* ----------------------------- 工具清单 ----------------------------- */
async function showTools() {
  const data = await api('/api/tools');
  const groups = {};
  (data.items || []).forEach((t) => {
    groups[t.group_label] = groups[t.group_label] || [];
    groups[t.group_label].push(t);
  });
  $('tools-body').innerHTML = Object.entries(groups).map(([label, items]) => `
    <div class="tool-group">
      <h4>${escapeHtml(label)}（${items.length}）</h4>
      ${items.map((t) => `<div class="tool-item"><code>${escapeHtml(t.name)}</code>
        <span class="desc"> — ${escapeHtml(t.description)}</span></div>`).join('')}
    </div>`).join('');
  openModal('modal-tools');
}

function openModal(id) { $(id).hidden = false; }
function closeModal(id) { $(id).hidden = true; }

/* ----------------------------- 事件绑定 ----------------------------- */
function bindEvents() {
  const dropzone = $('dropzone');
  const fileInput = $('file-input');

  $('btn-pick').onclick = () => fileInput.click();
  fileInput.onchange = () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); fileInput.value = ''; };

  ['dragenter', 'dragover'].forEach((evt) => dropzone.addEventListener(evt, (e) => {
    e.preventDefault(); dropzone.classList.add('hover');
  }));
  ['dragleave', 'drop'].forEach((evt) => dropzone.addEventListener(evt, (e) => {
    e.preventDefault(); dropzone.classList.remove('hover');
  }));
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  $('btn-preview').onclick = showPreview;
  $('btn-refresh-profile').onclick = refreshProfile;
  $('btn-reset-data').onclick = resetData;
  $('btn-send').onclick = () => { const v = $('input').value; $('input').value = ''; sendMessage(v); };
  $('btn-report').onclick = generateReport;
  $('btn-tools').onclick = showTools;
  $('btn-reset-conv').onclick = () => {
    state.conversationId = null;
    clearMessages();
    addMessage('system', '<span class="small">已开启新会话，旧会话仍保存在本地 Checkpoint 中。</span>');
  };

  $('input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const v = $('input').value;
      $('input').value = '';
      sendMessage(v);
    }
  });

  document.querySelectorAll('.quick button').forEach((btn) => {
    btn.onclick = () => sendMessage(btn.dataset.prompt);
  });
  document.querySelectorAll('[data-close]').forEach((btn) => {
    btn.onclick = () => closeModal(btn.dataset.close);
  });
  document.querySelectorAll('.modal').forEach((modal) => {
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.hidden = true; });
  });
}

/* ----------------------------- 启动 ----------------------------- */
(async function init() {
  bindEvents();
  await checkHealth();
  await loadDatasets();
  if (state.datasets.length) await selectDataset(state.datasets[0].id);
})();
