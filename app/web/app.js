/* CardioScribe v2 — SPA.
 * No frameworks, no bundler, no third-party JS: everything a patient waveform
 * touches stays on this machine. The ECG strip is drawn on a raw 2D canvas.
 */
'use strict';

const $  = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const fmt = (v, digits = 0, unit = '') =>
  (v === null || v === undefined || Number.isNaN(v)) ? null : `${Number(v).toFixed(digits)}${unit}`;

const state = {
  hea: null, dat: null,
  mode: 'demo',
  result: null,
  lead: 1,              // lead II
  view: { start: 0, end: 1 },   // fraction of the strip visible
  drag: null,
  stream: null,
};

/* ───────────────────────── boot ───────────────────────── */

async function boot() {
  for (const sel of ['likert-accuracy', 'likert-completeness', 'likert-safety']) {
    const node = $(sel);
    for (let i = 5; i >= 1; i--) node.append(new Option(`${i}`, String(i)));
    node.value = '5';
  }

  try {
    const health = await (await fetch('/api/health')).json();
    state.mode = health.mode;
    const chip = $('mode-chip');
    chip.textContent = health.mode === 'live' ? 'live model' : 'demo mode';
    chip.className = `chip chip--mode is-${health.mode}`;
    chip.title = health.mode === 'live'
      ? `Runtime: ${health.runtime_dir || 'configured'}`
      : 'No trained weights loaded';
    $('demo-note').classList.toggle('hidden', health.mode !== 'demo');
    if (health.mode === 'live' && !health.runtime_ready) {
      showError('The live runtime is configured but incomplete: ' + health.missing.join(', '));
    }
  } catch {
    showError('Cannot reach the CardioScribe API. Is the server running?');
  }

  try {
    const templates = await (await fetch('/api/templates')).json();
    for (const t of templates) $('opt-template').append(new Option(t.label, t.name));
  } catch { /* the select simply stays empty; the server falls back */ }

  wireUpload();
  wireCanvas();
  $('btn-analyze').addEventListener('click', analyze);
  $('btn-save-review').addEventListener('click', saveReview);
}

/* ─────────────────────── upload ─────────────────────── */

function wireUpload() {
  const dz = $('dropzone'), input = $('file-input');
  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => acceptFiles(input.files));

  ['dragenter', 'dragover'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('is-over'); }));
  dz.addEventListener('drop', (e) => acceptFiles(e.dataTransfer.files));
}

function acceptFiles(fileList) {
  clearError();
  for (const file of Array.from(fileList || [])) {
    const name = file.name.toLowerCase();
    if (/\.(png|jpe?g|gif|bmp|tiff?|webp|pdf)$/.test(name)) {
      showError(`"${file.name}" is an image or PDF. CardioScribe reads the raw 1D ` +
                `waveform, not a picture of one — 2D ECG image input is an explicit ` +
                `non-goal. Upload the WFDB .hea and .dat files.`);
      return;
    }
    if (name.endsWith('.hea')) state.hea = file;
    else if (name.endsWith('.dat')) state.dat = file;
    else { showError(`"${file.name}" is neither a .hea nor a .dat file.`); return; }
  }
  paintFileChips();
}

function paintFileChips() {
  const set = (node, file, fallback) => {
    node.querySelector('.file__name').textContent = file ? file.name : fallback;
    node.classList.toggle('is-set', !!file);
  };
  set($('file-hea'), state.hea, 'no header file');
  set($('file-dat'), state.dat, 'no signal file');

  let ready = !!(state.hea && state.dat);
  if (ready) {
    const hs = state.hea.name.replace(/\.hea$/i, '');
    const ds = state.dat.name.replace(/\.dat$/i, '');
    if (hs !== ds) {
      showError(`The two files must share a name: "${hs}.hea" and "${ds}.dat". ` +
                `WFDB locates the signal through the header, so the stems must match.`);
      ready = false;
    }
  }
  $('btn-analyze').disabled = !ready;
}

const showError = (msg) => { const n = $('upload-error'); n.textContent = msg; n.classList.remove('hidden'); };
const clearError = () => $('upload-error').classList.add('hidden');

/* ─────────────────────── analyse ─────────────────────── */

async function analyze() {
  clearError();
  $('results').classList.add('hidden');
  $('progress').classList.remove('hidden');
  $('btn-analyze').disabled = true;
  renderSteps('queued');
  $('step-message').textContent = 'Uploading…';

  const body = new FormData();
  body.append('hea_file', state.hea);
  body.append('dat_file', state.dat);
  body.append('template_name', $('opt-template').value || 'full_report');
  body.append('use_calibration', String($('opt-calibration').checked));
  if ($('opt-sex').value) body.append('sex', $('opt-sex').value);

  let jobId;
  try {
    const res = await fetch('/api/analyze', { method: 'POST', body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Upload failed (${res.status})`);
    jobId = data.job_id;
    $('queue-note').textContent = data.queue_position > 1
      ? `· position ${data.queue_position} in queue` : '';
  } catch (err) {
    failProgress(err.message);
    return;
  }

  if (state.stream) state.stream.close();
  const stream = new EventSource(`/api/jobs/${jobId}/stream`);
  state.stream = stream;

  stream.onmessage = async (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }
    if (data.stage) renderSteps(data.stage);
    if (data.message) $('step-message').textContent = data.message;
    $('queue-note').textContent = data.queue_position > 1
      ? `· position ${data.queue_position} in queue` : '';

    if (data.status === 'completed') {
      stream.close();
      try {
        const job = await (await fetch(`/api/jobs/${jobId}`)).json();
        renderResult(job.result);
        $('progress').classList.add('hidden');
      } catch (err) { failProgress(String(err)); }
      $('btn-analyze').disabled = false;
    } else if (data.status === 'failed') {
      stream.close();
      failProgress(data.error || 'Processing failed.');
    }
  };
  stream.onerror = () => {
    stream.close();
    failProgress('Lost the connection to the server while processing.');
  };
}

function failProgress(message) {
  $('progress').classList.add('hidden');
  $('btn-analyze').disabled = false;
  showError(message);
}

const STEPS = [
  ['queued',   'Queued for the single-GPU worker'],
  ['qc',       'Signal QC — 500 Hz, 12 leads, NaN / flat / clipping'],
  ['measure',  'Deterministic measurement — R-peaks, HR, RR, intervals'],
  ['encode',   'Frozen encoder + Q-Former, calibrated probabilities'],
  ['generate', '4-bit LLM report decoding'],
  ['audit',    'Evidence gate, validator, saliency, precedent lookup'],
  ['done',     'Audited report ready'],
];

function renderSteps(activeId) {
  const list = $('steps');
  const activeIdx = Math.max(0, STEPS.findIndex(([id]) => id === activeId));
  list.replaceChildren();
  STEPS.forEach(([id, label], i) => {
    const li = el('li', 'step');
    if (i < activeIdx) li.classList.add('is-done');
    if (i === activeIdx) li.classList.add('is-active');
    li.append(el('span', 'step__dot', i < activeIdx ? '✓' : String(i + 1)), el('span', '', label));
    list.append(li);
  });
}

/* ─────────────────────── results ─────────────────────── */

function renderResult(r) {
  state.result = r;
  state.view = { start: 0, end: 1 };
  $('results').classList.remove('hidden');

  renderVerdict(r);
  renderProbabilities(r);
  renderMeasurements(r);
  renderIntervals(r);
  renderSaliency(r);
  renderReport(r);
  renderPrecedents(r);
  renderQC(r);
  renderProvenance(r);

  const sel = $('lead-select');
  sel.replaceChildren();
  r.lead_names.forEach((name, i) => sel.append(new Option(`Lead ${name}`, String(i))));
  state.lead = Math.min(1, r.lead_names.length - 1);
  sel.value = String(state.lead);

  const hasAttention = Array.isArray(r.attention) && r.attention.length > 0;
  $('toggle-attention').disabled = !hasAttention;
  $('toggle-attention').checked = hasAttention;
  $('toggle-attention').parentElement.title = hasAttention
    ? 'Shade where the Q-Former looked' : 'No attention map for this record';
  $('sig-units').textContent = r.provenance.mode === 'demo'
    ? 'per-lead normalized (display only)' : 'standardized mV';

  drawECG();
  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderVerdict(r) {
  const flagged = r.flagged_for_review;
  const box = $('verdict');
  box.className = `verdict verdict--${flagged ? 'flag' : 'ok'}`;
  box.replaceChildren();

  const head = el('div', 'verdict__head');
  head.append(el('span', 'verdict__title',
    flagged ? '⚠ FLAGGED FOR CLINICIAN REVIEW' : '✓ CONFIDENT DRAFT — READY FOR SIGN-OFF'));
  const meta = el('div', 'controls');
  meta.append(el('span', 'chip', `record ${r.record_name}`));
  meta.append(el('span', 'chip', `abstention ≥ ${r.threshold.toFixed(3)}`));
  if (r.elapsed_s) meta.append(el('span', 'chip', `${r.elapsed_s.toFixed(1)} s`));
  head.append(meta);
  box.append(head);

  if (r.issues?.length) {
    const ul = el('ul', 'verdict__issues');
    r.issues.forEach((i) => ul.append(el('li', '', i)));
    box.append(ul);
  }
}

function renderProbabilities(r) {
  const host = $('probs');
  host.replaceChildren();
  const entries = Object.entries(r.probabilities || {});
  if (!entries.length) {
    host.append(el('p', 'empty', r.provenance.mode === 'demo'
      ? 'Withheld in demo mode — no classifier is loaded.'
      : 'No probabilities returned for this record.'));
    return;
  }
  for (const [name, p] of entries) {
    const row  = el('div', 'bar');
    const top  = el('div', 'bar__top');
    top.append(el('span', 'bar__name', name), el('span', 'muted', `${(p * 100).toFixed(1)}%`));
    const track = el('div', 'bar__track');
    const fill  = el('div', `bar__fill${p >= r.threshold ? ' is-over' : ''}`);
    fill.style.width = `${Math.min(100, Math.max(0, p * 100))}%`;
    const thresh = el('div', 'bar__thresh');
    thresh.style.left = `${Math.min(100, r.threshold * 100)}%`;
    thresh.title = `abstention threshold ${r.threshold.toFixed(3)}`;
    track.append(fill, thresh);
    row.append(top, track);
    host.append(row);
  }
}

function metric(label, value) {
  const box = el('div', 'metric');
  box.append(el('div', 'metric__k', label));
  box.append(el('div', `metric__v${value === null ? ' is-na' : ''}`, value === null ? 'n/a' : value));
  return box;
}

function renderMeasurements(r) {
  const m = r.measurements || {};
  $('measurements').replaceChildren(
    metric('heart rate', fmt(m.heart_rate, 0, ' bpm')),
    metric('rr mean',    fmt(m.rr_mean_ms, 0, ' ms')),
    metric('rr sd',      fmt(m.rr_std_ms, 0, ' ms')),
  );
}

function renderIntervals(r) {
  const iv = r.intervals;
  const wrap = $('intervals-wrap');
  if (!iv) { wrap.classList.add('hidden'); return; }
  wrap.classList.remove('hidden');
  $('intervals').replaceChildren(
    metric('PR',   fmt(iv.pr_ms, 0, ' ms')),
    metric('QRS',  fmt(iv.qrs_ms, 0, ' ms')),
    metric('QT',   fmt(iv.qt_ms, 0, ' ms')),
    metric('QTc',  fmt(iv.qtc_bazett_ms, 0, ' ms')),
    metric('axis', fmt(iv.axis_deg, 0, '°')),
  );
  const flags = $('interval-flags');
  flags.replaceChildren();
  (iv.flags || []).forEach((f) => flags.append(el('span', 'flag', f)));
}

function renderSaliency(r) {
  const host = $('saliency');
  host.replaceChildren();
  const entries = Object.entries(r.lead_importance || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    host.append(el('p', 'empty', r.provenance.mode === 'demo'
      ? 'Withheld in demo mode — saliency needs the trained encoder.'
      : 'No saliency returned for this record.'));
    return;
  }
  const max = entries[0][1] || 1;
  for (const [lead, val] of entries.slice(0, 6)) {
    const row = el('div', 'bar');
    const top = el('div', 'bar__top');
    top.append(el('span', 'bar__name', lead), el('span', 'muted', `${(val * 100).toFixed(1)}%`));
    const track = el('div', 'bar__track');
    const fill  = el('div', 'bar__fill is-over');
    fill.style.width = `${(val / max) * 100}%`;
    track.append(fill);
    row.append(top, track);
    host.append(row);
  }
}

function renderReport(r) {
  const note = $('report-provenance');
  if (!r.provenance.report_is_model_output) {
    note.textContent = 'This text is a placeholder, not model output. ' +
      'No report has been generated because no trained weights are loaded.';
    note.classList.remove('hidden');
  } else { note.classList.add('hidden'); }

  const fields = $('report-fields');
  fields.replaceChildren();
  const entries = Object.entries(r.report_fields || {});
  if (entries.length) {
    for (const [k, v] of entries) {
      const row = el('div', 'fieldrow');
      row.append(el('div', 'fieldrow__k', k.replace(/_/g, ' ')), el('div', '', v || '—'));
      fields.append(row);
    }
  } else {
    fields.append(el('p', 'empty', 'The report did not parse into the 8-field schema.'));
  }

  $('report-raw').textContent = r.report || '(empty)';
  $('edited-report').value = r.provenance.report_is_model_output ? (r.report || '') : '';
  $('review-status').textContent = '';

  const wrap = $('repair-wrap');
  if (r.repair_actions?.length) {
    wrap.classList.remove('hidden');
    const ul = $('repair-actions');
    ul.replaceChildren();
    r.repair_actions.forEach((a) => ul.append(el('li', '', a)));
  } else { wrap.classList.add('hidden'); }
}

function renderPrecedents(r) {
  const host = $('precedents');
  host.replaceChildren();
  if (!r.retrieved?.length) {
    host.append(el('p', 'empty', r.provenance.mode === 'demo'
      ? 'Withheld in demo mode — retrieval needs the PTB-XL training embeddings.'
      : 'No precedent reports retrieved.'));
    return;
  }
  for (const p of r.retrieved) {
    const card = el('div', 'prec');
    const top  = el('div', 'prec__top');
    top.append(el('span', '', `ecg_id ${p.ecg_id}`), el('span', '', `sim ${p.similarity.toFixed(3)}`));
    card.append(top, el('div', 'prec__text', p.report_text || '(no report text)'));
    host.append(card);
  }
}

function renderQC(r) {
  const q = r.qc;
  $('qc').replaceChildren(
    metric('rate',     `${q.sampling_rate} Hz`),
    metric('leads',    String(q.n_leads)),
    metric('duration', `${q.duration_s.toFixed(1)} s`),
    metric('units',    q.units || 'mV'),
  );
  const host = $('qc-warnings');
  host.replaceChildren();
  (q.warnings || []).forEach((w) => {
    const a = el('div', 'alert alert--warn', w);
    a.style.marginTop = '10px';
    host.append(a);
  });
}

function renderProvenance(r) {
  const ul = $('provenance');
  ul.replaceChildren();
  ul.append(el('li', '', `Mode: ${r.provenance.mode}`));
  ul.append(el('li', '', `Measurements from: ${r.provenance.measurement_source}`));
  ul.append(el('li', '', `Report is model output: ${r.provenance.report_is_model_output ? 'yes' : 'no'}`));
  (r.provenance.notes || []).forEach((n) => ul.append(el('li', '', n)));
}

/* ────────────────────── canvas ECG ────────────────────── */

function wireCanvas() {
  const c = $('ecg');
  $('lead-select').addEventListener('change', (e) => { state.lead = Number(e.target.value); drawECG(); });
  $('toggle-attention').addEventListener('change', drawECG);
  $('toggle-grid').addEventListener('change', drawECG);
  $('btn-zoom-reset').addEventListener('click', () => { state.view = { start: 0, end: 1 }; drawECG(); });
  window.addEventListener('resize', drawECG);

  c.addEventListener('wheel', (e) => {
    if (!state.result) return;
    e.preventDefault();
    const rect = c.getBoundingClientRect();
    const anchor = state.view.start +
      ((e.clientX - rect.left) / rect.width) * (state.view.end - state.view.start);
    const factor = e.deltaY > 0 ? 1.18 : 1 / 1.18;
    let span = Math.min(1, Math.max(0.01, (state.view.end - state.view.start) * factor));
    let start = Math.min(Math.max(0, anchor - (anchor - state.view.start) * factor), 1 - span);
    state.view = { start, end: start + span };
    drawECG();
  }, { passive: false });

  c.addEventListener('pointerdown', (e) => {
    if (!state.result) return;
    state.drag = { x: e.clientX, view: { ...state.view } };
    c.setPointerCapture(e.pointerId);
  });
  c.addEventListener('pointermove', (e) => {
    if (!state.drag) return;
    const rect = c.getBoundingClientRect();
    const span = state.drag.view.end - state.drag.view.start;
    const shift = -((e.clientX - state.drag.x) / rect.width) * span;
    const start = Math.min(Math.max(0, state.drag.view.start + shift), 1 - span);
    state.view = { start, end: start + span };
    drawECG();
  });
  ['pointerup', 'pointercancel'].forEach((ev) =>
    c.addEventListener(ev, () => { state.drag = null; }));
}

function drawECG() {
  const r = state.result;
  const canvas = $('ecg');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 800;
  const cssH = canvas.clientHeight || 340;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  if (!r || !r.signal?.length) return;

  const pad = { l: 46, r: 12, t: 12, b: 26 };
  const W = cssW - pad.l - pad.r;
  const H = cssH - pad.t - pad.b;
  const n = r.signal.length;
  const fs = r.sampling_rate / (r.signal_decimation || 1);

  const i0 = Math.floor(state.view.start * (n - 1));
  const i1 = Math.max(i0 + 2, Math.ceil(state.view.end * (n - 1)));
  const lead = Math.min(state.lead, r.lead_names.length - 1);

  // y-range from the visible window, with a little headroom
  let lo = Infinity, hi = -Infinity;
  for (let i = i0; i <= i1 && i < n; i++) {
    const v = r.signal[i][lead];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const margin = (hi - lo) * 0.12;
  lo -= margin; hi += margin;

  const xOf = (i) => pad.l + ((i - i0) / (i1 - i0)) * W;
  const yOf = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * H;

  // ── ECG paper grid ──
  if ($('toggle-grid').checked) {
    const tStart = i0 / fs, tEnd = i1 / fs;
    ctx.lineWidth = 1;
    for (const [step, color] of [[0.04, 'var-small'], [0.2, 'var-big']]) {
      ctx.strokeStyle = step === 0.04
        ? getComputedStyle(document.body).getPropertyValue('--grid').trim()
        : getComputedStyle(document.body).getPropertyValue('--grid-bold').trim();
      ctx.beginPath();
      for (let t = Math.ceil(tStart / step) * step; t <= tEnd; t += step) {
        const x = Math.round(xOf(t * fs)) + 0.5;
        ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + H);
      }
      const yStep = (hi - lo) / (step === 0.04 ? 25 : 5);
      for (let v = lo; v <= hi; v += yStep) {
        const y = Math.round(yOf(v)) + 0.5;
        ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + W, y);
      }
      ctx.stroke();
    }
  }

  // ── attention shading ──
  const att = r.attention;
  if ($('toggle-attention').checked && Array.isArray(att) && att.length === n) {
    // Shade the top decile of the VISIBLE window, not everything above a fixed
    // fraction of the peak. A percentile is self-normalising: a peaked map and a
    // flat one both highlight the same amount of strip, so the overlay always
    // answers "where did it look most", never "the model looked everywhere".
    const win = [];
    for (let i = i0; i <= i1 && i < n; i++) win.push(att[i]);
    if (win.length) {
      const sorted = Float64Array.from(win).sort();
      const cut  = sorted[Math.floor(sorted.length * 0.90)];
      const peak = sorted[sorted.length - 1];
      if (peak > cut) {
        for (let i = i0; i < i1 && i < n - 1; i++) {
          if (att[i] < cut) continue;
          const k = (att[i] - cut) / (peak - cut);
          ctx.fillStyle = `rgba(245, 180, 69, ${(0.10 + 0.26 * k).toFixed(3)})`;
          ctx.fillRect(xOf(i), pad.t, Math.max(1, xOf(i + 1) - xOf(i)), H);
        }
      }
    }
  }

  // ── waveform: min/max per pixel column, so 5000 points stay honest ──
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--teal').trim() || '#2dd4bf';
  ctx.lineWidth = 1.35;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  const perPx = (i1 - i0) / W;
  if (perPx > 2) {
    for (let px = 0; px < W; px++) {
      const s = i0 + Math.floor(px * perPx);
      const e = Math.min(n - 1, i0 + Math.floor((px + 1) * perPx));
      let mn = Infinity, mx = -Infinity;
      for (let i = s; i <= e; i++) {
        const v = r.signal[i][lead];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      if (!Number.isFinite(mn)) continue;
      const x = pad.l + px + 0.5;
      ctx.moveTo(x, yOf(mx)); ctx.lineTo(x, yOf(mn));
    }
  } else {
    let started = false;
    for (let i = i0; i <= i1 && i < n; i++) {
      const x = xOf(i), y = yOf(r.signal[i][lead]);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  // ── axes ──
  const muted = getComputedStyle(document.body).getPropertyValue('--muted').trim() || '#8da2bb';
  ctx.fillStyle = muted;
  ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of [hi, (hi + lo) / 2, lo]) ctx.fillText(v.toFixed(2), pad.l - 7, yOf(v));
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const ticks = 6;
  for (let k = 0; k <= ticks; k++) {
    const i = i0 + ((i1 - i0) * k) / ticks;
    ctx.fillText(`${(i / fs).toFixed(2)}s`, xOf(i), pad.t + H + 7);
  }
  ctx.textAlign = 'left';
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--teal').trim() || '#2dd4bf';
  ctx.font = '600 12px system-ui, sans-serif';
  ctx.fillText(`Lead ${r.lead_names[lead]}`, pad.l + 6, pad.t + 4);
}

/* ─────────────────────── review ─────────────────────── */

async function saveReview() {
  const r = state.result;
  if (!r) return;
  const status = $('review-status');
  status.textContent = 'Saving…';
  try {
    const res = await fetch('/api/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        record_name: r.record_name,
        generated_report: r.report,
        edited_report: $('edited-report').value,
        measured_hr: r.measurements?.heart_rate ?? null,
        flagged_for_review: r.flagged_for_review,
        mode: r.provenance.mode,
        likert_accuracy: Number($('likert-accuracy').value),
        likert_completeness: Number($('likert-completeness').value),
        likert_safety: Number($('likert-safety').value),
        reviewer_notes: $('reviewer-notes').value,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Save failed');
    status.textContent = `Saved — ${data.rows} row(s) in ${data.path}`;
  } catch (err) {
    status.textContent = `Could not save: ${err.message}`;
  }
}

document.addEventListener('DOMContentLoaded', boot);
