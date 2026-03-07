/**
 * Interactive Difference-in-Differences Simulation
 * Lets readers adjust parameters and see how naive/TWFE/event-study estimates change.
 */
(function () {
  'use strict';

  // --- Seeded RNG (Mulberry32) ---
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // Box-Muller for normal distribution
  function normalRandom(rng) {
    var u1 = rng(), u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  // --- Simulation ---
  function runSimulation(nTreatment, nControl, trueEffect, staggerRange, nMonths) {
    var rng = mulberry32(42);
    var locations = [];

    // Treatment locations
    for (var i = 0; i < nTreatment; i++) {
      var adoptionStart = 3 + Math.floor(rng() * staggerRange);
      var nEmp = 1 + Math.floor(rng() * 5);
      var starts = [];
      if (rng() < 0.4) {
        for (var e = 0; e < nEmp; e++) starts.push(adoptionStart);
      } else {
        var spread = 1 + Math.floor(rng() * 3);
        for (var e = 0; e < nEmp; e++) {
          starts.push(adoptionStart + Math.floor(rng() * spread));
        }
      }
      var completions = starts.map(function (s) { return s + 6; });
      var fc = Math.min.apply(null, completions);
      var lc = Math.max.apply(null, completions);
      if (lc - fc > 6) continue;

      var months = [], metrics = [];
      for (var m = 1; m <= nMonths; m++) {
        var baseline = 50 + 0.3 * m;
        var noise = normalRandom(rng) * 2;
        var effect = 0;
        if (m >= fc) {
          var monthsPost = Math.min(m - fc + 1, 3);
          effect = monthsPost * (trueEffect / 3);
        }
        months.push(m);
        metrics.push(baseline + effect + noise);
      }
      locations.push({ id: i, treatment: 1, fc: fc, lc: lc, months: months, metrics: metrics, adoptionStart: adoptionStart });
    }

    // Control locations
    for (var i = 0; i < nControl; i++) {
      var months = [], metrics = [];
      for (var m = 1; m <= nMonths; m++) {
        var baseline = 50 + 0.3 * m;
        var noise = normalRandom(rng) * 2;
        months.push(m);
        metrics.push(baseline + noise);
      }
      locations.push({ id: nTreatment + i, treatment: 0, fc: null, lc: null, months: months, metrics: metrics });
    }

    return locations;
  }

  // --- Naive Cutoff DiD ---
  function naiveCutoff(locations, nMonths) {
    var fcs = locations.filter(function (l) { return l.treatment === 1; }).map(function (l) { return l.fc; });
    fcs.sort(function (a, b) { return a - b; });
    var cutoff = fcs[Math.floor(fcs.length / 2)];

    var treatPre = 0, treatPost = 0, ctrlPre = 0, ctrlPost = 0;
    var nTP = 0, nTPo = 0, nCP = 0, nCPo = 0;

    locations.forEach(function (loc) {
      for (var j = 0; j < loc.months.length; j++) {
        var m = loc.months[j], v = loc.metrics[j];
        if (loc.treatment === 1) {
          if (m < cutoff) { treatPre += v; nTP++; }
          else { treatPost += v; nTPo++; }
        } else {
          if (m < cutoff) { ctrlPre += v; nCP++; }
          else { ctrlPost += v; nCPo++; }
        }
      }
    });

    var treatChange = (treatPost / nTPo) - (treatPre / nTP);
    var ctrlChange = (ctrlPost / nCPo) - (ctrlPre / nCP);
    return treatChange - ctrlChange;
  }

  // --- TWFE (simplified) ---
  function twfeEstimate(locations) {
    var treatPre = 0, treatPost = 0, ctrlPre = 0, ctrlPost = 0;
    var nTP = 0, nTPo = 0, nCP = 0, nCPo = 0;

    var fcs = locations.filter(function (l) { return l.treatment === 1; }).map(function (l) { return l.fc; });
    fcs.sort(function (a, b) { return a - b; });
    var medianFc = fcs[Math.floor(fcs.length / 2)];

    locations.forEach(function (loc) {
      for (var j = 0; j < loc.months.length; j++) {
        var m = loc.months[j], v = loc.metrics[j];
        if (loc.treatment === 1) {
          if (m < loc.fc) { treatPre += v; nTP++; }
          else { treatPost += v; nTPo++; }
        } else {
          if (m < medianFc) { ctrlPre += v; nCP++; }
          else { ctrlPost += v; nCPo++; }
        }
      }
    });

    var treatChange = (treatPost / nTPo) - (treatPre / nTP);
    var ctrlChange = (ctrlPost / nCPo) - (ctrlPre / nCP);
    return treatChange - ctrlChange;
  }

  // --- Event-Study (period-by-period DiD means) ---
  function eventStudyEstimate(locations) {
    var periodData = {};
    for (var p = -6; p <= 6; p++) {
      periodData[p] = { treatSum: 0, treatN: 0, ctrlSum: 0, ctrlN: 0 };
    }

    var ctrlLocs = locations.filter(function (l) { return l.treatment === 0; });
    var treatLocs = locations.filter(function (l) { return l.treatment === 1; });
    var medianFc = treatLocs.map(function (l) { return l.fc; }).sort(function (a, b) { return a - b; })[Math.floor(treatLocs.length / 2)];

    treatLocs.forEach(function (loc) {
      for (var j = 0; j < loc.months.length; j++) {
        var m = loc.months[j];
        var eventTime = m - loc.fc;
        if (eventTime >= -6 && eventTime <= 6) {
          var p = Math.round(eventTime);
          periodData[p].treatSum += loc.metrics[j];
          periodData[p].treatN++;
        }
      }
    });

    ctrlLocs.forEach(function (loc) {
      for (var j = 0; j < loc.months.length; j++) {
        var m = loc.months[j];
        var eventTime = m - medianFc;
        if (eventTime >= -6 && eventTime <= 6) {
          var p = Math.round(eventTime);
          periodData[p].ctrlSum += loc.metrics[j];
          periodData[p].ctrlN++;
        }
      }
    });

    var coefficients = {};
    var refDiff = 0;
    if (periodData[-1].treatN > 0 && periodData[-1].ctrlN > 0) {
      refDiff = (periodData[-1].treatSum / periodData[-1].treatN) -
                (periodData[-1].ctrlSum / periodData[-1].ctrlN);
    }

    for (var p = -6; p <= 6; p++) {
      if (periodData[p].treatN > 0 && periodData[p].ctrlN > 0) {
        var diff = (periodData[p].treatSum / periodData[p].treatN) -
                   (periodData[p].ctrlSum / periodData[p].ctrlN);
        coefficients[p] = diff - refDiff;
      } else {
        coefficients[p] = 0;
      }
    }

    // Average post-period coefficient
    var postSum = 0, postN = 0;
    for (var p = 1; p <= 6; p++) {
      if (coefficients[p] !== undefined) { postSum += coefficients[p]; postN++; }
    }
    var avgPost = postN > 0 ? postSum / postN : 0;

    return { coefficients: coefficients, avgPost: avgPost };
  }

  // --- Build UI ---
  function buildUI() {
    var container = document.getElementById('did-interactive');
    if (!container) return;

    container.innerHTML = '';

    // Controls wrapper
    var controls = document.createElement('div');
    controls.style.cssText = 'display:flex;flex-wrap:wrap;gap:24px;align-items:flex-end;margin-bottom:24px;padding:20px;background:#f8f9fa;border:1px solid #ddd;border-radius:8px;';

    function makeSlider(label, id, min, max, step, value, unit, helpText) {
      var wrap = document.createElement('div');
      wrap.style.cssText = 'flex:1;min-width:180px;';
      var lbl = document.createElement('label');
      lbl.style.cssText = 'display:block;font-weight:600;font-size:0.9em;margin-bottom:4px;color:#333;';
      lbl.textContent = label;
      var valSpan = document.createElement('span');
      valSpan.id = id + '-val';
      valSpan.style.cssText = 'font-weight:700;color:#4A90E2;';
      valSpan.textContent = value + (unit || '');
      lbl.appendChild(document.createTextNode(' '));
      lbl.appendChild(valSpan);
      var slider = document.createElement('input');
      slider.type = 'range'; slider.id = id;
      slider.min = min; slider.max = max; slider.step = step; slider.value = value;
      slider.style.cssText = 'width:100%;cursor:pointer;';
      slider.addEventListener('input', function () {
        valSpan.textContent = this.value + (unit || '');
      });
      wrap.appendChild(lbl);
      wrap.appendChild(slider);
      if (helpText) {
        var help = document.createElement('div');
        help.style.cssText = 'font-size:0.78em;color:#777;margin-top:2px;line-height:1.3;';
        help.textContent = helpText;
        wrap.appendChild(help);
      }
      return wrap;
    }

    controls.appendChild(makeSlider('Treatment Locations:', 'did-n-treat', 20, 200, 10, 120, '',
      'Number of locations that receive the software'));
    controls.appendChild(makeSlider('True Effect (units):', 'did-effect', 0, 10, 0.5, 4.5, '',
      'The real treatment effect built into the simulation'));
    controls.appendChild(makeSlider('Adoption Spread:', 'did-stagger', 1, 15, 1, 12, ' mo',
      'How many months between the earliest and latest adopters. 1 = everyone starts together. 15 = widely staggered.'));

    var btnWrap = document.createElement('div');
    btnWrap.style.cssText = 'min-width:140px;';
    var btn = document.createElement('button');
    btn.id = 'did-run-btn';
    btn.textContent = 'Run Simulation';
    btn.style.cssText = 'padding:10px 20px;background:#4A90E2;color:white;border:none;border-radius:6px;font-size:1em;font-weight:600;cursor:pointer;transition:background 0.2s;';
    btn.addEventListener('mouseenter', function () { this.style.background = '#357ABD'; });
    btn.addEventListener('mouseleave', function () { this.style.background = '#4A90E2'; });
    btnWrap.appendChild(btn);
    controls.appendChild(btnWrap);

    container.appendChild(controls);

    // Results text
    var resultsDiv = document.createElement('div');
    resultsDiv.id = 'did-results';
    resultsDiv.style.cssText = 'margin-bottom:20px;padding:16px 20px;background:#fff;border:1px solid #ddd;border-radius:8px;font-size:0.95em;display:none;';
    container.appendChild(resultsDiv);

    // Chart containers
    var chartsRow = document.createElement('div');
    chartsRow.style.cssText = 'display:flex;flex-wrap:wrap;gap:16px;';

    var chart1 = document.createElement('div');
    chart1.id = 'did-chart-event-study';
    chart1.style.cssText = 'flex:1;min-width:300px;min-height:350px;';
    chartsRow.appendChild(chart1);

    var chart2 = document.createElement('div');
    chart2.id = 'did-chart-comparison';
    chart2.style.cssText = 'flex:1;min-width:300px;min-height:350px;';
    chartsRow.appendChild(chart2);

    container.appendChild(chartsRow);

    // Wire up button
    btn.addEventListener('click', runAndPlot);

    // Run once on load
    runAndPlot();
  }

  function runAndPlot() {
    var nTreat = parseInt(document.getElementById('did-n-treat').value);
    var effect = parseFloat(document.getElementById('did-effect').value);
    var stagger = parseInt(document.getElementById('did-stagger').value);
    var nControl = 80;
    var nMonths = 24;

    var locations = runSimulation(nTreat, nControl, effect, stagger, nMonths);
    var naive = naiveCutoff(locations, nMonths);
    var twfe = twfeEstimate(locations);
    var es = eventStudyEstimate(locations);

    // Update results text
    var resultsDiv = document.getElementById('did-results');
    resultsDiv.style.display = 'block';
    var bias = function (est) { return effect > 0 ? ((est - effect) / effect * 100).toFixed(0) : 'N/A'; };
    resultsDiv.innerHTML =
      '<strong>Results</strong> (true effect = ' + effect.toFixed(1) + ')&nbsp;&nbsp;|&nbsp;&nbsp;' +
      'Naive: <strong>' + naive.toFixed(2) + '</strong> (' + bias(naive) + '% bias)&nbsp;&nbsp;|&nbsp;&nbsp;' +
      'TWFE: <strong>' + twfe.toFixed(2) + '</strong> (' + bias(twfe) + '% bias)&nbsp;&nbsp;|&nbsp;&nbsp;' +
      'Event-Study: <strong>' + es.avgPost.toFixed(2) + '</strong> (' + bias(es.avgPost) + '% bias)';

    // Event-study chart
    var periods = [], coefs = [];
    for (var p = -6; p <= 6; p++) {
      periods.push(p);
      coefs.push(es.coefficients[p] || 0);
    }

    var esTrace = {
      x: periods, y: coefs,
      type: 'scatter', mode: 'lines+markers',
      name: 'Coefficient',
      line: { color: '#4A90E2', width: 2.5 },
      marker: { size: 7 }
    };
    var zeroLine = {
      x: [-6, 6], y: [0, 0],
      type: 'scatter', mode: 'lines',
      line: { color: '#999', dash: 'dash', width: 1 },
      showlegend: false
    };
    var trueEffectLine = {
      x: [-6, 6], y: [effect, effect],
      type: 'scatter', mode: 'lines',
      name: 'True Effect (' + effect.toFixed(1) + ')',
      line: { color: '#E8827C', dash: 'dot', width: 2 }
    };

    var esLayout = {
      title: { text: 'Event-Study Coefficients', font: { size: 14 } },
      xaxis: { title: 'Relative Period', dtick: 1 },
      yaxis: { title: 'Treatment Effect' },
      margin: { t: 50, b: 50, l: 60, r: 20 },
      legend: { x: 0.02, y: 0.98 },
      shapes: [{
        type: 'rect', x0: -0.5, x1: 0.5,
        y0: 0, y1: 1, yref: 'paper',
        fillcolor: '#999', opacity: 0.1, line: { width: 0 }
      }],
      font: { family: 'system-ui, sans-serif' }
    };

    Plotly.newPlot('did-chart-event-study', [zeroLine, esTrace, trueEffectLine], esLayout, { responsive: true, displayModeBar: false });

    // Comparison bar chart
    var barTrace = {
      x: ['Naive Cutoff', 'TWFE', 'Event-Study'],
      y: [naive, twfe, es.avgPost],
      type: 'bar',
      marker: {
        color: ['#E8827C', '#999999', '#4A90E2'],
        line: { width: 1, color: '#333' }
      },
      text: [naive.toFixed(2), twfe.toFixed(2), es.avgPost.toFixed(2)],
      textposition: 'outside',
      textfont: { size: 13, color: '#333' }
    };

    var barLayout = {
      title: { text: 'Method Comparison', font: { size: 14 } },
      yaxis: { title: 'Estimated Effect', range: [0, Math.max(effect * 1.5, naive * 1.3, 2)] },
      margin: { t: 50, b: 50, l: 60, r: 20 },
      shapes: [{
        type: 'line', x0: -0.5, x1: 2.5,
        y0: effect, y1: effect,
        line: { color: '#E8827C', dash: 'dot', width: 2 }
      }],
      annotations: [{
        x: 2.4, y: effect,
        text: 'True Effect (' + effect.toFixed(1) + ')',
        showarrow: false, font: { size: 11, color: '#E8827C' },
        xanchor: 'right', yanchor: 'bottom'
      }],
      font: { family: 'system-ui, sans-serif' }
    };

    Plotly.newPlot('did-chart-comparison', [barTrace], barLayout, { responsive: true, displayModeBar: false });
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', buildUI);
  } else {
    buildUI();
  }
})();
