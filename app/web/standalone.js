/* CardioScribe v2 — in-browser engine for static hosting (GitHub Pages).
 *
 * GitHub Pages serves files, not processes: there is no Python, no GPU and no
 * model. But DEMO mode never needed any of those — it is pure signal
 * processing — so the whole demo path is reimplemented here and runs in the
 * tab. The page produces exactly the same result schema as the FastAPI backend,
 * so app.js renders it without knowing the difference.
 *
 * What runs here for real: WFDB parsing, the 0.5-40 Hz zero-phase band-pass,
 * R-peak detection, heart rate and RR variability.
 * What is withheld, exactly as server-side demo mode withholds it: the report,
 * diagnosis probabilities, attention, saliency and precedent retrieval.
 */
'use strict';

const Standalone = (() => {

  const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
  const SEQ_LEN = 5000, CLIP_MV = 4.0, MIN_SECONDS = 5.0;

  class UploadError extends Error {}

  /* ───────── WFDB header ───────── */

  function parseHeader(text) {
    const lines = text.split(/\r?\n/)
      .map(l => l.trim())
      .filter(l => l && !l.startsWith('#'));
    if (!lines.length) throw new UploadError('The .hea file is empty.');

    const head = lines[0].split(/\s+/);
    const nSig = parseInt(head[1], 10);
    const fs = parseFloat(head[2]);
    const nSamp = parseInt(head[3], 10);
    if (!Number.isFinite(nSig) || !Number.isFinite(fs)) {
      throw new UploadError('Could not read the record line of the .hea file.');
    }

    const signals = [];
    for (let i = 1; i <= nSig && i < lines.length; i++) {
      const f = lines[i].split(/\s+/);
      // e.g.  demo_ecg.dat 16 1000.0(0)/mV 16 0 1 46492 0 II
      const fmt = parseInt(f[1], 10);
      const gainField = f[2] || '200';
      const gain = parseFloat(gainField) || 200;
      const baseMatch = gainField.match(/\(([-\d.]+)\)/);
      const unitsMatch = gainField.match(/\/(\S+)$/);
      signals.push({
        file: f[0],
        format: fmt,
        gain,
        baseline: baseMatch ? parseFloat(baseMatch[1]) : 0,
        units: unitsMatch ? unitsMatch[1] : 'mV',
        name: f[8] || LEAD_NAMES[i - 1] || `ch${i}`,
      });
    }
    return { nSig, fs, nSamp, signals };
  }

  /* ───────── WFDB signal file ───────── */

  function parseDat(buffer, header) {
    const formats = new Set(header.signals.map(s => s.format));
    if (formats.size !== 1) {
      throw new UploadError('Mixed WFDB storage formats in one record are not supported here.');
    }
    const format = header.signals[0].format;
    if (format !== 16) {
      throw new UploadError(
        `This record uses WFDB format ${format}. The in-browser demo reads format 16 ` +
        `(16-bit little-endian), which is what PTB-XL ships. Run the app locally for ` +
        `the full reader.`
      );
    }

    const n = header.nSig;
    const raw = new Int16Array(buffer, 0, Math.floor(buffer.byteLength / 2));
    const nSamples = Math.floor(raw.length / n);
    if (nSamples < 1) throw new UploadError('The .dat file holds no complete samples.');

    // physical = (adc - baseline) / gain     [WFDB convention]
    const out = [];
    for (let s = 0; s < nSamples; s++) {
      const row = new Float64Array(n);
      for (let c = 0; c < n; c++) {
        const sig = header.signals[c];
        row[c] = (raw[s * n + c] - sig.baseline) / sig.gain;
      }
      out.push(row);
    }
    return out;
  }

  /* ───────── 0.5-40 Hz zero-phase band-pass ─────────
   *
   * These are the EXACT second-order sections SciPy produces for
   *   butter(4, [0.5, 40.0], btype='bandpass', fs=500, output='sos')
   * which is what backend/wfdb_io.py applies. Embedding them, rather than
   * designing a filter in JS, is what makes the browser agree with the server:
   * a hand-rolled biquad cascade drifted by up to 0.23 mV, enough to move
   * detected R-peaks by a few samples and visibly corrupt RR variability.
   *
   * They are fixed for fs = 500 Hz, which the QC step already requires.
   */
  const SOS_500HZ = [
    [0.0021387987326912015, 0.004277597465382403, 0.0021387987326912015, 1.0, -1.22787587909828, 0.39352306024517103],
    [1.0, 2.0, 1.0, 1.0, -1.486663673168146, 0.6949675580253452],
    [1.0, -2.0, 1.0, 1.0, -1.9882154714982394, 0.9882564156624591],
    [1.0, -2.0, 1.0, 1.0, -1.995246749028118, 0.9952864068099667],
  ];

  /** One pass of a cascaded SOS filter, transposed direct form II (as SciPy).
   *  `zi` optionally seeds each section's delay line. Verified bit-exact
   *  against scipy.signal.sosfilt on real records. */
  function sosfilt(x, sos, zi) {
    const y = Float64Array.from(x);
    for (let s = 0; s < sos.length; s++) {
      const [b0, b1, b2, , a1, a2] = sos[s];
      let z1 = zi ? zi[s][0] : 0;
      let z2 = zi ? zi[s][1] : 0;
      for (let i = 0; i < y.length; i++) {
        const xn = y[i];
        const yn = b0 * xn + z1;
        z1 = b1 * xn - a1 * yn + z2;
        z2 = b2 * xn - a2 * yn;
        y[i] = yn;
      }
    }
    return y;
  }

  /** Steady-state delay values for a unit step, per section — scipy.sosfilt_zi.
   *
   * For one transposed-DF2 biquad at constant input x = 1:
   *   y  = (b0+b1+b2) / (1+a1+a2)
   *   z1 = (b1+b2) - (a1+a2) * y
   *   z2 =  b2     -  a2     * y
   * Each later section sees the DC gain of the ones before it, so the running
   * `scale` carries that through exactly as SciPy does.
   */
  function sosfiltZi(sos) {
    const zi = [];
    let scale = 1;
    for (const [b0, b1, b2, , a1, a2] of sos) {
      const denom = 1 + a1 + a2;
      const y = (b0 + b1 + b2) / denom;
      zi.push([((b1 + b2) - (a1 + a2) * y) * scale, (b2 - a2 * y) * scale]);
      scale *= (b0 + b1 + b2) / denom;
    }
    return zi;
  }

  const reverse = (x) => {
    const y = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) y[i] = x[x.length - 1 - i];
    return y;
  };

  const scaleZi = (zi, x0) => zi.map(([a, b]) => [a * x0, b * x0]);

  /** Zero-phase filtering, following scipy.signal.sosfiltfilt step for step:
   *  odd-reflect by 3*(2*n_sections+1), seed each pass with the steady state
   *  for its first sample, filter forward then backward, trim.
   *
   *  Seeding matters more than padding here. Padding alone left a ~0.1 mV
   *  interior error, because the 0.5 Hz high-pass sections have poles at
   *  |p| ≈ 0.995 and take thousands of samples to settle.
   */
  function sosfiltfilt(x, sos) {
    const pad = Math.min(3 * (2 * sos.length + 1), Math.max(0, x.length - 1));
    let ext;
    if (pad < 1) {
      ext = Float64Array.from(x);
    } else {
      ext = new Float64Array(x.length + 2 * pad);
      for (let i = 0; i < pad; i++) {
        ext[i] = 2 * x[0] - x[pad - i];
        ext[ext.length - 1 - i] = 2 * x[x.length - 1] - x[x.length - 1 - (pad - i)];
      }
      ext.set(x, pad);
    }

    const zi = sosfiltZi(sos);
    const forward = sosfilt(ext, sos, scaleZi(zi, ext[0]));
    const flipped = reverse(forward);
    const backward = sosfilt(flipped, sos, scaleZi(zi, flipped[0]));
    const done = reverse(backward);
    return pad < 1 ? done : done.slice(pad, pad + x.length);
  }

  function bandpass(signal, fs) {
    if (Math.abs(fs - 500) > 1e-9) {
      throw new UploadError(
        `The in-browser filter is fixed at 500 Hz and this record is ${fs} Hz.`
      );
    }
    const nLeads = signal[0].length;
    const out = signal.map(() => new Float64Array(nLeads));
    for (let c = 0; c < nLeads; c++) {
      const col = new Float64Array(signal.length);
      for (let i = 0; i < signal.length; i++) col[i] = signal[i][c];
      const filtered = sosfiltfilt(col, SOS_500HZ);
      for (let i = 0; i < signal.length; i++) out[i][c] = filtered[i];
    }
    return out;
  }

  /* ───────── R-peaks (port of backend/runtime_demo.py) ───────── */

  /** numpy.percentile with the default linear interpolation. */
  function percentile(sortedAsc, q) {
    const n = sortedAsc.length;
    if (!n) return 0;
    const idx = (n - 1) * q;
    const lo = Math.floor(idx), frac = idx - lo;
    return lo + 1 < n ? sortedAsc[lo] + frac * (sortedAsc[lo + 1] - sortedAsc[lo]) : sortedAsc[lo];
  }

  /** scipy.signal._local_maxima_1d — plateaus report their midpoint. */
  function localMaxima(x) {
    const peaks = [];
    let i = 1;
    const last = x.length - 1;
    while (i < last) {
      if (x[i - 1] < x[i]) {
        let ahead = i + 1;
        while (ahead < last && x[ahead] === x[i]) ahead++;
        if (x[ahead] < x[i]) {
          peaks.push((i + ahead - 1) >> 1);
          i = ahead;
          continue;
        }
      }
      i++;
    }
    return peaks;
  }

  /** scipy.signal._select_by_peak_distance — keep tallest first, not leftmost.
   *  Greedy left-to-right picks different beats, which is what made RR
   *  variability disagree with the server even on an identical signal. */
  function selectByDistance(peaks, heights, distance) {
    const n = peaks.length;
    const keep = new Uint8Array(n).fill(1);
    const order = Array.from({ length: n }, (_, i) => i)
      .sort((a, b) => heights[a] - heights[b]);      // ascending, walked in reverse
    for (let k = n - 1; k >= 0; k--) {
      const j = order[k];
      if (!keep[j]) continue;
      for (let m = j - 1; m >= 0 && peaks[j] - peaks[m] < distance; m--) keep[m] = 0;
      for (let m = j + 1; m < n && peaks[m] - peaks[j] < distance; m++) keep[m] = 0;
    }
    return peaks.filter((_, i) => keep[i]);
  }

  function detectRPeaks(leadII, fs) {
    const n = leadII.length;
    if (n < fs) return [];

    const sq = new Float64Array(n);
    for (let i = 1; i < n; i++) { const d = leadII[i] - leadII[i - 1]; sq[i] = d * d; }

    // Centred moving average == numpy.convolve(sq, ones(win)/win, 'same').
    const win = Math.max(1, Math.round(0.12 * fs));
    const back = Math.floor(win / 2), fwd = Math.floor((win - 1) / 2);
    const prefix = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) prefix[i + 1] = prefix[i] + sq[i];
    const integ = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i - back), hi = Math.min(n - 1, i + fwd);
      integ[i] = (prefix[hi + 1] - prefix[lo]) / win;
    }

    let any = false;
    for (let i = 0; i < n; i++) if (integ[i] > 0) { any = true; break; }
    if (!any) return [];

    const threshold = Math.max(0.35 * percentile(Float64Array.from(integ).sort(), 0.99), 1e-12);
    const candidates = localMaxima(integ).filter(i => integ[i] >= threshold);
    const peaks = selectByDistance(candidates, candidates.map(i => integ[i]),
                                   Math.round(0.25 * fs));

    // Snap onto the true local extremum of the filtered signal.
    const half = Math.round(0.05 * fs);
    const snapped = [];
    for (const p of peaks) {
      let best = p, bestAbs = -1;
      for (let i = Math.max(0, p - half); i < Math.min(n, p + half + 1); i++) {
        const a = Math.abs(leadII[i]);
        if (a > bestAbs) { bestAbs = a; best = i; }
      }
      if (!snapped.length || best !== snapped[snapped.length - 1]) snapped.push(best);
    }
    return snapped;
  }

  function measure(filtered, fs) {
    const leadII = new Float64Array(filtered.length);
    for (let i = 0; i < filtered.length; i++) leadII[i] = filtered[i][1];
    const peaks = detectRPeaks(leadII, fs);

    const out = { heart_rate: null, rr_mean_ms: null, rr_std_ms: null };
    if (peaks.length < 3) return out;

    const rr = [];
    for (let i = 1; i < peaks.length; i++) {
      const ms = ((peaks[i] - peaks[i - 1]) / fs) * 1000;
      if (ms > 200 && ms < 3000) rr.push(ms);
    }
    if (!rr.length) return out;

    const sorted = [...rr].sort((a, b) => a - b);
    const median = sorted.length % 2
      ? sorted[(sorted.length - 1) / 2]
      : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
    const mean = rr.reduce((a, b) => a + b, 0) / rr.length;
    const variance = rr.reduce((a, b) => a + (b - mean) ** 2, 0) / rr.length;

    out.heart_rate = Math.round((60000 / median) * 10) / 10;
    out.rr_mean_ms = Math.round(mean * 10) / 10;
    out.rr_std_ms = Math.round(Math.sqrt(variance) * 10) / 10;
    return out;
  }

  /* ───────── QC + assembly ───────── */

  function qualityControl(raw, header) {
    const nSamples = raw.length, nLeads = header.nSig;
    if (nLeads !== 12) throw new UploadError(`Expected 12 leads, got ${nLeads}.`);
    if (header.fs !== 500) {
      throw new UploadError(
        `Expected 500 Hz, got ${header.fs} Hz. The pipeline reads PTB-XL at its native ` +
        `500 Hz and does not resample on upload.`
      );
    }
    if (nSamples < MIN_SECONDS * header.fs) {
      throw new UploadError(
        `Signal is ${(nSamples / header.fs).toFixed(1)} s; at least ${MIN_SECONDS} s is required.`
      );
    }

    const warnings = [], flat = [], clipped = [];
    for (let c = 0; c < nLeads; c++) {
      let mn = Infinity, mx = -Infinity, sum = 0, sum2 = 0;
      for (let i = 0; i < nSamples; i++) {
        const v = raw[i][c];
        if (!Number.isFinite(v)) throw new UploadError('Signal QC failed: non-finite samples.');
        if (v < mn) mn = v; if (v > mx) mx = v;
        sum += v; sum2 += v * v;
      }
      const sd = Math.sqrt(Math.max(0, sum2 / nSamples - (sum / nSamples) ** 2));
      if (sd < 1e-6) flat.push(LEAD_NAMES[c]);
      if (Math.max(Math.abs(mn), Math.abs(mx)) >= CLIP_MV) clipped.push(LEAD_NAMES[c]);
    }
    if (flat.length === nLeads) throw new UploadError('Signal QC failed: every lead is flat.');
    if (flat.length) warnings.push(`Flat (zero-variance) lead(s): ${flat.join(', ')}.`);
    if (clipped.length) warnings.push(`Possible ADC clipping (|x| >= ${CLIP_MV} mV) in: ${clipped.join(', ')}.`);
    if (nSamples !== SEQ_LEN) warnings.push(`Signal is ${nSamples} samples; cropped or padded to ${SEQ_LEN}.`);

    return {
      sampling_rate: header.fs,
      n_leads: nLeads,
      n_samples: nSamples,
      duration_s: Math.round((nSamples / header.fs) * 1000) / 1000,
      units: header.signals[0]?.units || 'mV',
      clipped: clipped.length > 0,
      clipped_leads: clipped,
      flat_leads: flat,
      warnings,
    };
  }

  function padOrCrop(sig) {
    if (sig.length === SEQ_LEN) return sig;
    if (sig.length > SEQ_LEN) return sig.slice(0, SEQ_LEN);
    const out = sig.slice();
    while (out.length < SEQ_LEN) out.push(new Float64Array(sig[0].length));
    return out;
  }

  function standardize(sig) {
    const nLeads = sig[0].length;
    const out = sig.map(() => new Array(nLeads));
    for (let c = 0; c < nLeads; c++) {
      let sum = 0, sum2 = 0;
      for (let i = 0; i < sig.length; i++) { sum += sig[i][c]; sum2 += sig[i][c] ** 2; }
      const mean = sum / sig.length;
      const sd = Math.max(Math.sqrt(Math.max(0, sum2 / sig.length - mean * mean)), 1e-6);
      for (let i = 0; i < sig.length; i++) out[i][c] = Math.round(((sig[i][c] - mean) / sd) * 1e4) / 1e4;
    }
    return out;
  }

  const PLACEHOLDER_REPORT = `DEMO — THIS IS NOT MODEL OUTPUT.

This page runs entirely in your browser. There is no model here: GitHub Pages
serves files, not a GPU, so no report has been generated.

The measurements below ARE real. They come from the same 0.5-40 Hz zero-phase
band-pass and R-peak detection the pipeline uses, reimplemented in JavaScript.

For an actual report, clone the repository and run the app locally against
trained weights. See app/README.md.`;

  const PLACEHOLDER_FIELDS = {
    rhythm: '(no model loaded)', rate: '(see measured heart rate)',
    axis: '(no model loaded)', intervals: '(see measurements)',
    conduction: '(no model loaded)', st_t: '(no model loaded)',
    hypertrophy: '(no model loaded)', impression: '(no model loaded — browser demo)',
  };

  /** Mirrors backend AnalysisResult so app.js cannot tell the difference. */
  async function analyze(heaFile, datFile) {
    const started = performance.now();
    const stem = heaFile.name.replace(/\.hea$/i, '');

    const header = parseHeader(await heaFile.text());
    const raw = parseDat(await datFile.arrayBuffer(), header);
    const qc = qualityControl(raw, header);

    const filtered = padOrCrop(bandpass(raw, header.fs));
    const measurements = measure(filtered, header.fs);

    return {
      record_name: stem,
      provenance: {
        mode: 'demo',
        report_is_model_output: false,
        probabilities_are_model_output: false,
        measurement_source: 'browser-detector',
        notes: [
          'This page is static: everything ran in your browser, nothing was uploaded.',
          'Band-pass, R-peak detection, heart rate and RR variability are really computed.',
          'No report, no probabilities, no attention, no saliency and no precedent ' +
            'retrieval — all four need the trained model and PTB-XL.',
          'The band-pass uses the exact second-order sections SciPy produces for the '
            + 'server, and the R-peak detector follows scipy.find_peaks semantics, so '
            + 'heart rate and RR variability match the server exactly.',
        ],
      },
      qc,
      report: PLACEHOLDER_REPORT,
      report_fields: { ...PLACEHOLDER_FIELDS },
      diagnoses: [],
      probabilities: {},
      threshold: 0.852,
      measurements,
      intervals: null,
      signal: standardize(filtered),
      signal_decimation: 1,
      lead_names: LEAD_NAMES.slice(),
      sampling_rate: header.fs,
      attention: null,
      lead_importance: {},
      retrieved: [],
      confident: false,
      issues: ['Browser demo: no model output to validate.'],
      flagged_for_review: true,
      repair_actions: [],
      elapsed_s: Math.round((performance.now() - started)) / 1000,
    };
  }

  return { analyze, UploadError, _internals: { parseHeader, parseDat, bandpass, measure, sosfilt, sosfiltZi, sosfiltfilt, detectRPeaks, SOS_500HZ } };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = Standalone;
