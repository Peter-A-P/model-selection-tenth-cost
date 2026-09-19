// Every chart on this page, drawn by hand into SVG and one canvas.
//
// No chart library. The page is four small files and a folder of JSON, it loads nothing from
// anywhere else, and a reader who wants to know how a line got onto the screen can read the
// function that put it there.

"use strict";

const NS = "http://www.w3.org/2000/svg";

export function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    node.setAttribute(key, String(value));
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

function scale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (value) => r0 + ((value - d0) / span) * (r1 - r0);
}

// A log scale for the item-count axis, because the interesting part of every curve on this
// page happens between ten items and a hundred and a linear axis spends four fifths of its
// width on the part where nothing moves.
function logScale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const l0 = Math.log(d0);
  const span = Math.log(d1) - l0 || 1;
  return (value) => r0 + ((Math.log(Math.max(value, d0)) - l0) / span) * (r1 - r0);
}

// ------------------------------------------------------------------ the live adaptive test

// Two posteriors on one ability axis, with the credible interval each one currently holds.
// This is the picture the project is about: the width of those bars is what a budget buys.
export function drawPosteriors(svg, series, options = {}) {
  const width = 640;
  const height = options.height || 220;
  const pad = { top: 14, right: 16, bottom: 34, left: 16 };
  clear(svg);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");

  const domain = options.domain || [-4.5, 4.5];
  const x = scale(domain, [pad.left, width - pad.right]);
  const floor = height - pad.bottom;
  const top = pad.top;
  const barBand = 30;

  let peak = 0;
  for (const item of series) for (const value of item.density) peak = Math.max(peak, value);
  const y = scale([0, peak * 1.05 || 1], [floor - barBand, top]);

  for (let tick = Math.ceil(domain[0]); tick <= domain[1]; tick += 1) {
    svg.appendChild(
      svgEl("line", { class: "grid", x1: x(tick), x2: x(tick), y1: top, y2: floor })
    );
    svg.appendChild(
      svgEl("text", { class: "tick", x: x(tick), y: height - 16, "text-anchor": "middle" },
        tick > 0 ? `+${tick}` : String(tick))
    );
  }
  svg.appendChild(
    svgEl("text", { class: "axis", x: width / 2, y: height - 2, "text-anchor": "middle" },
      "ability, on the bank's scale")
  );

  // The ends of the scale, drawn rather than left implicit. The prior this estimator carries
  // is a grid from -4.5 to +4.5, so a model the bank cannot place inside that range piles up
  // against the edge and gets an interval that looks tight because it has a wall on one side.
  // Two frontier models do exactly that on this suite, and a chart that does not show the wall
  // makes it look like precision.
  for (const edge of domain) {
    svg.appendChild(
      svgEl("line", {
        class: "edge", x1: x(edge), x2: x(edge), y1: top, y2: floor - barBand,
      })
    );
  }
  svg.appendChild(
    svgEl("text", { class: "tick edge-label", x: x(domain[1]) - 4, y: top + 10, "text-anchor": "end" },
      "edge of the scale")
  );

  series.forEach((item, index) => {
    const nodes = item.nodes;
    let path = "";
    for (let i = 0; i < nodes.length; i += 1) {
      path += `${i === 0 ? "M" : "L"}${x(nodes[i]).toFixed(2)},${y(item.density[i]).toFixed(2)}`;
    }
    const area = `${path}L${x(nodes[nodes.length - 1]).toFixed(2)},${(floor - barBand).toFixed(2)}L${x(nodes[0]).toFixed(2)},${(floor - barBand).toFixed(2)}Z`;
    svg.appendChild(svgEl("path", { d: area, fill: item.colour, class: "density-fill" }));
    svg.appendChild(
      svgEl("path", { d: path, fill: "none", stroke: item.colour, "stroke-width": 2, class: "series" })
    );
    const barY = floor - barBand + 10 + index * 12;
    svg.appendChild(
      svgEl("line", {
        x1: x(item.interval[0]), x2: x(item.interval[1]), y1: barY, y2: barY,
        stroke: item.colour, "stroke-width": 5, "stroke-linecap": "round", opacity: 0.9,
      })
    );
    svg.appendChild(
      svgEl("circle", { cx: x(item.theta), cy: barY, r: 4, fill: item.colour, class: "dot" })
    );
  });
  return svg;
}

// ------------------------------------------------------------------ the headline curve

export function drawTauCurve(svg, panel, options = {}) {
  const width = 720;
  const height = 380;
  const pad = { top: 18, right: 108, bottom: 46, left: 52 };
  clear(svg);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const rows = panel.curve;
  const counts = [...new Set(rows.map((r) => r.items))].sort((a, b) => a - b);
  const x = logScale([counts[0], counts[counts.length - 1]], [pad.left, width - pad.right]);
  const lows = rows.map((r) => r.lo);
  const y = scale([Math.min(0.3, Math.min(...lows) - 0.03), 1], [height - pad.bottom, pad.top]);

  for (let value = 0.3; value <= 1.0001; value += 0.1) {
    const at = y(value);
    svg.appendChild(svgEl("line", { class: "grid", x1: pad.left, x2: width - pad.right, y1: at, y2: at }));
    svg.appendChild(
      svgEl("text", { class: "tick", x: pad.left - 8, y: at + 4, "text-anchor": "end" }, value.toFixed(1))
    );
  }
  for (const count of counts) {
    svg.appendChild(
      svgEl("text", { class: "tick", x: x(count), y: height - pad.bottom + 16, "text-anchor": "middle" },
        count >= 1000 ? `${count / 1000}k` : String(count))
    );
  }
  svg.appendChild(
    svgEl("text", { class: "axis", x: (pad.left + width - pad.right) / 2, y: height - 10, "text-anchor": "middle" },
      "questions asked per model")
  );
  svg.appendChild(
    svgEl("text", {
      class: "axis", x: 14, y: (height - pad.bottom + pad.top) / 2,
      "text-anchor": "middle", transform: `rotate(-90 14 ${(height - pad.bottom + pad.top) / 2})`,
    }, "agreement with the full ranking")
  );

  // Labels go in the right margin rather than on the lines, and they are pushed apart before
  // they are drawn: three curves that converge, which is the whole point of the chart, would
  // otherwise print their three names on top of each other.
  const labels = [];
  if (panel.tau_full_fit) {
    const at = y(panel.tau_full_fit);
    svg.appendChild(svgEl("line", { class: "zero", x1: pad.left, x2: width - pad.right, y1: at, y2: at }));
    labels.push({ text: `ceiling ${panel.tau_full_fit.toFixed(2)}`, y: at, colour: null });
  }

  const methods = options.methods || ["adaptive", "stratified", "random"];
  for (const method of methods) {
    const series = rows.filter((r) => r.method === method).sort((a, b) => a.items - b.items);
    if (!series.length) continue;
    const colour = options.colours[method];
    const hidden = options.hidden && options.hidden.has(method);
    let band = "";
    series.forEach((point, i) => {
      band += `${i === 0 ? "M" : "L"}${x(point.items).toFixed(1)},${y(point.hi).toFixed(1)}`;
    });
    for (let i = series.length - 1; i >= 0; i -= 1) {
      band += `L${x(series[i].items).toFixed(1)},${y(series[i].lo).toFixed(1)}`;
    }
    svg.appendChild(
      svgEl("path", { d: `${band}Z`, fill: colour, class: `band${hidden ? " hidden" : ""}` })
    );
    let line = "";
    series.forEach((point, i) => {
      line += `${i === 0 ? "M" : "L"}${x(point.items).toFixed(1)},${y(point.tau).toFixed(1)}`;
    });
    svg.appendChild(
      svgEl("path", {
        d: line, class: `series${hidden ? " hidden" : ""}`, stroke: colour,
        "stroke-width": method === "adaptive" ? 3 : 2,
        "stroke-dasharray": method === "random" ? "7 5" : null,
      })
    );
    const last = series[series.length - 1];
    labels.push({ text: method, y: y(last.tau), colour, hidden });
  }

  labels.sort((first, second) => first.y - second.y);
  for (let i = 1; i < labels.length; i += 1) {
    const gap = labels[i].y - labels[i - 1].y;
    if (gap < 14) labels[i].y = labels[i - 1].y + 14;
  }
  for (const label of labels) {
    svg.appendChild(
      svgEl("text", {
        class: `end-label${label.hidden ? " hidden" : ""}${label.colour ? "" : " tick"}`,
        x: width - pad.right + 6,
        y: label.y + 4,
        fill: label.colour,
      }, label.text)
    );
  }
  return svg;
}

// ------------------------------------------------------------------ the item cloud

// Difficulty across, discrimination up, one dot per item, drawn to canvas because a scatter of
// several thousand points in SVG is several thousand DOM nodes and a page that stutters.
export function drawItemCloud(canvas, groups, options = {}) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 640;
  const height = options.height || 420;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pad = { top: 12, right: 14, bottom: 36, left: 54 };
  const x = scale(options.domainX || [-4, 6], [pad.left, width - pad.right]);
  const y = scale(options.domainY || [-1, 3.5], [height - pad.bottom, pad.top]);
  const style = getComputedStyle(document.body);
  const line = style.getPropertyValue("--line").trim() || "#ddd";
  const faint = style.getPropertyValue("--ink-faint").trim() || "#888";
  const warn = style.getPropertyValue("--warn").trim() || "#c0392b";

  // The dead zone: everything below a discrimination of 0.3, where an item's answer barely
  // depends on ability at all. A fifth of this bank sits in it.
  ctx.fillStyle = warn;
  ctx.globalAlpha = 0.07;
  ctx.fillRect(pad.left, y(0.3), width - pad.left - pad.right, height - pad.bottom - y(0.3));
  ctx.globalAlpha = 1;

  ctx.strokeStyle = line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, y(0));
  ctx.lineTo(width - pad.right, y(0));
  ctx.stroke();
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = warn;
  ctx.beginPath();
  ctx.moveTo(pad.left, y(0.3));
  ctx.lineTo(width - pad.right, y(0.3));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.fillStyle = faint;
  ctx.textAlign = "right";
  for (const value of [-1, 0, 1, 2, 3]) ctx.fillText(value.toFixed(0), pad.left - 10, y(value) + 4);
  ctx.textAlign = "center";
  for (const value of [-4, -2, 0, 2, 4, 6]) {
    ctx.fillText(value > 0 ? `+${value}` : String(value), x(value), height - pad.bottom + 16);
  }
  ctx.fillText("difficulty: the ability an item sits at", width / 2, height - 6);
  ctx.save();
  ctx.translate(14, (height - pad.bottom + pad.top) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("discrimination", 0, 0);
  ctx.restore();

  for (const group of groups) {
    ctx.fillStyle = group.colour;
    ctx.globalAlpha = group.dim ? 0.06 : 0.5;
    for (let i = 0; i < group.a.length; i += 1) {
      const b = group.b[i];
      if (b === null || b === undefined) continue;
      ctx.beginPath();
      ctx.arc(x(b), y(group.a[i]), group.dim ? 1.2 : 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
  return canvas;
}

// ------------------------------------------------------------------ per-model bars

export function drawBars(svg, rows, options = {}) {
  const width = 640;
  const rowHeight = 26;
  // The gutter is sized to the longest name rather than fixed. These rows used to be aliases,
  // all of a similar length; a model identifier is not, and a fixed gutter either clipped
  // "Llama-3.3-70B-Instruct-Turbo" or wasted a third of the width on "qwen2.5:3b".
  const longest = rows.reduce((most, row) => Math.max(most, row.label.length), 0);
  const pad = { top: 8, right: 60, bottom: 26, left: Math.min(232, Math.max(110, longest * 6.6 + 14)) };
  const height = pad.top + pad.bottom + rows.length * rowHeight;
  clear(svg);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const top = options.max || Math.max(...rows.map((r) => r.hi ?? r.value)) * 1.1 || 1;
  const x = scale([0, top], [pad.left, width - pad.right]);

  for (let i = 0; i <= 4; i += 1) {
    const value = (top * i) / 4;
    svg.appendChild(
      svgEl("line", { class: "grid", x1: x(value), x2: x(value), y1: pad.top, y2: height - pad.bottom })
    );
    svg.appendChild(
      svgEl("text", { class: "tick", x: x(value), y: height - 8, "text-anchor": "middle" },
        options.format ? options.format(value) : value.toFixed(2))
    );
  }

  rows.forEach((row, index) => {
    const y = pad.top + index * rowHeight + rowHeight / 2;
    svg.appendChild(
      svgEl("text", { class: "row-label", x: pad.left - 10, y: y + 4, "text-anchor": "end" }, row.label)
    );
    svg.appendChild(
      svgEl("rect", {
        x: pad.left, y: y - 7, width: Math.max(x(row.value) - pad.left, 1), height: 14,
        rx: 3, fill: row.colour || "var(--accent)", opacity: 0.85,
      })
    );
    if (row.lo !== undefined && row.hi !== undefined) {
      svg.appendChild(
        svgEl("line", {
          x1: x(row.lo), x2: x(row.hi), y1: y, y2: y,
          stroke: "var(--ink)", "stroke-width": 1.5, opacity: 0.55,
        })
      );
    }
    svg.appendChild(
      svgEl("text", { class: "row-value", x: x(row.value) + 8, y: y + 4 },
        options.format ? options.format(row.value) : row.value.toFixed(2))
    );
  });
  return svg;
}
