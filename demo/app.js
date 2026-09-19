// The page: load the data this repository wrote, wire up five things a reader can move, and
// draw the result. No framework and no build step; what is served is what was written.

"use strict";

import { Posterior, Selector, RandomSelector, quadrature, rng, separated, unpackBits } from "./irt.js";
import { drawPosteriors, drawTauCurve, drawItemCloud, drawBars, clear, svgEl } from "./charts.js";

const MAX_QUESTIONS = 300;

// What the same two models cost when every question was put to them, which is the comparison
// the short test exists to beat. Both currencies, because one of them is zero for a model on a
// laptop and that is the case where a dollar figure undersells the method rather than proving
// it: the laptop's bill is hours of a machine nobody else can use while it runs.
function comparedToEverything(arena) {
  const usd = arena.models.reduce((total, model) => total + model.full_usd, 0);
  const seconds = arena.models.reduce((total, model) => total + (model.full_seconds || 0), 0);
  const everything = `all ${state.panel.n_items.toLocaleString("en-CA")} questions of both`;
  if (usd < 0.0005) {
    return (
      `Nothing in dollars, because these run on a laptop: what a short test saves here is the ` +
      `machine. It took <strong>${duration(arena.seconds)}</strong> against ` +
      `<strong>${duration(seconds)}</strong> to ask ${everything}.`
    );
  }
  const timed = arena.timed === 2 && seconds > 0;
  return (
    `Asking ${everything} cost <strong>${money(usd)}</strong>` +
    (timed ? ` and took <strong>${duration(seconds)}</strong>` : "") +
    ` when it was run.`
  );
}

// Said whenever an estimate reaches the end of the scale, which is a property of the suite
// rather than of the run: the questions stop before these models do.
function ceilingNote(arena) {
  const named = arena.models
    .filter((_, side) => Math.abs(arena.posteriors[side].theta) > 4.0)
    .map((model) => model.model);
  const subject = named.length > 1 ? "Both estimates have" : `<code>${named[0]}</code> has`;
  return (
    `<strong>${subject} run off the end of the bank's scale.</strong> The selector asks for ` +
    "questions it believes are a coin flip at this level, and this panel's strongest models get " +
    "about nine in ten of those right, so the estimate climbs until the scale stops it. Only 253 " +
    "of the 2,816 questions are harder than +3. The suite has run out of questions hard enough, " +
    "and an interval measured against that wall is not precision."
  );
};

const state = { panel: null, curves: null, items: null, experiments: null, power: null, run: null };

const colour = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();

// Dollars, to as many places as the number deserves and no more. A page about measurement
// error should not print four decimal places on a figure whose third one is noise.
const money = (usd) =>
  usd === 0
    ? "$0"
    : usd >= 0.1
      ? `$${usd.toFixed(2)}`
      : usd >= 0.01
        ? `$${usd.toFixed(3)}`
        : `$${usd.toFixed(4)}`;

// The other currency. A model on a laptop costs nothing in dollars and hours of a machine
// nobody else can use while it runs, which is the cost a team with its own hardware actually
// pays. Both come from the same run: the ledger recorded a price and a latency per call.
function duration(seconds) {
  if (seconds === null || seconds === undefined) return "not recorded";
  if (seconds < 90) return `${seconds.toFixed(0)}s`;
  if (seconds < 5400) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

async function load(name) {
  const response = await fetch(`data/${name}.json`);
  if (!response.ok) throw new Error(`${name}: ${response.status}`);
  return response.json();
}

// ------------------------------------------------------------------ the live adaptive test

function prepare() {
  const panel = state.panel;
  const items = {
    a: Float64Array.from(panel.items.a),
    b: Float64Array.from(panel.items.b.map((v) => (v === null ? 0 : v))),
    c: Float64Array.from(panel.items.c),
    stratum: Int32Array.from(panel.items.stratum),
    usable: unpackBits(panel.items.usable, panel.n_items),
  };
  const models = panel.models.map((model) => ({
    ...model,
    responses: unpackBits(model.correct, panel.n_items),
  }));
  return { items, models, weights: Float64Array.from(panel.weights) };
}

class Arena {
  constructor(key, kind, prepared, a, b, seed) {
    this.key = key;
    this.kind = kind;
    this.prepared = prepared;
    this.models = [a, b];
    this.posteriors = [new Posterior(), new Posterior()];
    this.selectors = this.models.map(() =>
      kind === "adaptive"
        ? new Selector(prepared.items, prepared.weights)
        : new RandomSelector(prepared.items)
    );
    this.draw = rng(seed);
    this.asked = 0;
    this.usd = 0;
    this.seconds = 0;
    this.timed = this.models.filter((model) => model.latency_ms).length;
    this.decidedAt = null;
    this.decision = null;
    this.trail = [];
  }

  step() {
    if (this.decidedAt !== null || this.asked >= MAX_QUESTIONS) return false;
    const chosen = [];
    for (let side = 0; side < 2; side += 1) {
      const posterior = this.posteriors[side];
      const item = this.selectors[side].next(posterior.theta, this.draw);
      if (item < 0) return false;
      const model = this.models[side];
      const correct = model.responses[item] === 1;
      posterior.update(
        this.prepared.items.a[item],
        this.prepared.items.b[item],
        this.prepared.items.c[item],
        correct
      );
      this.usd += model.cost_micro[item] / 1e6;
      if (model.latency_ms && model.latency_ms[item] !== null) {
        this.seconds += model.latency_ms[item] / 1000;
      }
      chosen.push({ item, correct });
    }
    this.asked += 1;
    this.trail.push({ n: this.asked, first: chosen[0], second: chosen[1] });
    if (this.trail.length > 14) this.trail.shift();
    const verdict = separated(this.posteriors[0], this.posteriors[1]);
    if (verdict.separated && this.asked >= 10) {
      this.decidedAt = this.asked;
      this.decision = verdict;
    }
    return true;
  }

  finished() {
    return this.decidedAt !== null || this.asked >= MAX_QUESTIONS;
  }

  widest() {
    return Math.max(
      ...this.posteriors.map((p) => {
        const [lo, hi] = p.interval();
        return hi - lo;
      })
    );
  }

  // Is either estimate pressed against the end of the bank's scale? The estimator carries a
  // grid from -4.5 to +4.5, so a model the bank cannot place inside that range stops there and
  // its interval gets narrow for the wrong reason: it has a wall on one side rather than
  // evidence. It happens to the two strongest models on this suite, and an interval measured
  // against a wall should not be read as precision.
  atTheEdge() {
    return this.posteriors.some((p) => Math.abs(p.theta) > 4.0);
  }
}

function drawArena(arena) {
  const svg = document.getElementById(`chart-${arena.key}`);
  const tones = [colour("--accent"), colour("--sim")];
  const series = arena.posteriors.map((posterior, index) => ({
    nodes: quadrature.nodes,
    density: posterior.count ? posterior.density() : quadrature.weights,
    interval: posterior.interval(),
    theta: posterior.theta,
    colour: tones[index],
  }));
  drawPosteriors(svg, series);

  // Classes, never a style attribute. The hosting policy is style-src 'self', which blocks an
  // inline style attribute in markup, and the failure is silent: the swatches simply come out
  // blank on the live site and correct in every local check that sends no headers.
  const legend = document.getElementById(`legend-${arena.key}`);
  legend.innerHTML = arena.models
    .map(
      (model, index) =>
        `<span class="pair-key"><span class="pair-swatch ${index === 0 ? "first" : "second"}">` +
        `</span>${model.model}</span>`
    )
    .join("");

  document.getElementById(`${arena.key}-items`).textContent = String(arena.asked);
  document.getElementById(`${arena.key}-cost`).textContent = money(arena.usd);
  document.getElementById(`${arena.key}-time`).textContent =
    arena.timed === 0 ? "-" : duration(arena.seconds);
  document.getElementById(`${arena.key}-time-note`).textContent =
    arena.timed === 0
      ? "both ran as a batch, so no call was timed"
      : arena.timed === 1
        ? "of machine time; one ran as a batch and is not counted"
        : "of machine time these calls took";
  document.getElementById(`${arena.key}-width`).textContent =
    arena.asked === 0 ? "-" : arena.widest().toFixed(2);

  const target = document.getElementById(`${arena.key}-verdict`);
  const [first, second] = arena.models;
  if (arena.decision) {
    const leader = arena.decision.leader === "a" ? first : second;
    const other = arena.decision.leader === "a" ? second : first;
    // Whether the short test agreed with the full suite is checked rather than asserted. It
    // usually does, and where it does not the pair is a fraction of a point apart and the
    // disagreement is the construct gap this page's curve is about, not a fault in the run.
    const agrees = leader.accuracy > other.accuracy;
    target.className = `verdict ${agrees ? "good" : "warn"}`;
    target.innerHTML =
      `<strong>${leader.model} is ahead, decided after ${arena.decidedAt} questions</strong> ` +
      `for ${money(arena.usd)}. Their intervals no longer overlap, so nothing further is bought ` +
      `by asking again. ` +
      `${comparedToEverything(arena)} ` +
      (agrees
        ? `Asking all ${state.panel.n_items.toLocaleString("en-CA")} questions ranks them the same way.`
        : `Asking all ${state.panel.n_items.toLocaleString("en-CA")} questions ranks them the ` +
          `other way round, by ` +
          `${(Math.abs(leader.accuracy - other.accuracy) * 100).toFixed(1)} of a point. Ability ` +
          `and a count of correct answers are not the same quantity, and at a gap this small ` +
          `they can disagree.`) +
      (arena.atTheEdge() ? ` ${ceilingNote(arena)}` : "");
  } else if (arena.asked >= MAX_QUESTIONS) {
    target.className = "verdict warn";
    target.innerHTML =
      `<strong>Not separated after ${MAX_QUESTIONS} questions</strong>, for ${money(arena.usd)}. ` +
      `That is an answer: these two are indistinguishable at this budget. It is never reported ` +
      `as "the same", because the evidence does not support that. ${comparedToEverything(arena)}` +
      (arena.atTheEdge() ? ` ${ceilingNote(arena)}` : "");
  } else if (arena.asked > 0) {
    target.className = "verdict";
    target.innerHTML =
      `${arena.asked} questions in, the intervals still overlap.` +
      (arena.atTheEdge() ? ` ${ceilingNote(arena)}` : "");
  } else {
    target.className = "verdict";
    target.textContent = "Not started.";
  }
}

function drawLog(arena) {
  const body = document.getElementById("log-body");
  clear(body);
  const benchmarks = state.panel.benchmarks;
  for (const row of arena.trail) {
    const item = row.first.item;
    const tr = document.createElement("tr");
    const cells = [
      String(row.n),
      benchmarks[state.panel.items.stratum[item]].title,
      state.panel.items.b[item] === null ? "-" : state.panel.items.b[item].toFixed(2),
      state.panel.items.a[item].toFixed(2),
    ];
    for (const [index, value] of cells.entries()) {
      const td = document.createElement("td");
      td.textContent = value;
      if (index >= 2) td.className = "num";
      tr.appendChild(td);
    }
    for (const side of [row.first, row.second]) {
      const td = document.createElement("td");
      td.textContent = side.correct ? "right" : "wrong";
      td.className = side.correct ? "right" : "wrong";
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

function setupRace() {
  const prepared = prepare();
  const selectA = document.getElementById("model-a");
  const selectB = document.getElementById("model-b");
  const ordered = [...state.panel.models].sort((x, y) => y.accuracy - x.accuracy);
  // Named, and grouped by where they ran. The panel is addressed by alias in the code so that
  // a vendor renaming a model cannot break anything, and an alias is the wrong thing to show a
  // reader: "local-small-a against local-small-b" is two strings, where "llama3.2:3b against
  // qwen2.5:3b" is a comparison somebody might actually want the answer to. These are the
  // identifiers the vendors themselves returned on the call, out of the run records.
  for (const select of [selectA, selectB]) {
    for (const hosted of [true, false]) {
      const group = document.createElement("optgroup");
      group.label = hosted ? "Hosted, billed by the token" : "On a laptop, billed in hours";
      for (const model of ordered.filter((m) => m.hosted === hosted)) {
        const option = document.createElement("option");
        option.value = model.alias;
        option.textContent = `${model.model} - ${(model.accuracy * 100).toFixed(1)}%`;
        group.appendChild(option);
      }
      select.appendChild(group);
    }
  }
  // Four accuracy points apart, which is the case the method is for: close enough that a
  // leaderboard reader would call it too close to call, far enough that it is really there.
  // Adaptive selection settles this pair in about a hundred questions; random selection does
  // not settle it at all inside three hundred.
  selectA.value = "anthropic-haiku";
  selectB.value = "google-mid";

  const runButton = document.getElementById("run-test");
  const resetButton = document.getElementById("reset");
  let timer = null;

  function stop() {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
    runButton.textContent = "Run the test";
    runButton.disabled = false;
  }

  function reset() {
    stop();
    const find = (alias) => prepared.models.find((m) => m.alias === alias);
    const a = find(selectA.value);
    const b = find(selectB.value);
    const seed = (a.alias.length * 31 + b.alias.length * 17 + 7) >>> 0;
    state.run = {
      adaptive: new Arena("adaptive", "adaptive", prepared, a, b, seed),
      random: new Arena("random", "random", prepared, a, b, seed),
    };
    document.getElementById("log-a").textContent = a.model;
    document.getElementById("log-b").textContent = b.model;
    drawArena(state.run.adaptive);
    drawArena(state.run.random);
    drawLog(state.run.adaptive);
  }

  function advance() {
    const { adaptive, random } = state.run;
    let moved = false;
    if (!adaptive.finished()) moved = adaptive.step() || moved;
    if (!random.finished()) moved = random.step() || moved;
    drawArena(adaptive);
    drawArena(random);
    drawLog(adaptive);
    if (!moved || (adaptive.finished() && random.finished())) stop();
  }

  runButton.addEventListener("click", () => {
    if (timer !== null) {
      stop();
      return;
    }
    if (state.run.adaptive.finished() && state.run.random.finished()) reset();
    const speed = Number(document.getElementById("speed").value);
    runButton.textContent = "Pause";
    if (speed === 0) {
      let guard = 0;
      while (
        guard < MAX_QUESTIONS * 2 &&
        !(state.run.adaptive.finished() && state.run.random.finished())
      ) {
        const { adaptive, random } = state.run;
        if (!adaptive.finished()) adaptive.step();
        if (!random.finished()) random.step();
        guard += 1;
      }
      drawArena(state.run.adaptive);
      drawArena(state.run.random);
      drawLog(state.run.adaptive);
      stop();
      return;
    }
    timer = setInterval(advance, speed);
  });

  resetButton.addEventListener("click", reset);
  setupSweep(prepared);
  for (const select of [selectA, selectB]) {
    select.addEventListener("change", () => {
      if (selectA.value === selectB.value) {
        const other = ordered.find((m) => m.alias !== selectA.value);
        if (select === selectA) selectB.value = other.alias;
        else selectA.value = other.alias;
      }
      reset();
    });
  }
  reset();
}

// Every pair of the twelve models, both methods. One run is an anecdote and a reader is right
// to treat it as one, so the page offers to do the whole matrix rather than asking to be
// believed. It runs in chunks so the page keeps answering while it works.
function setupSweep(prepared) {
  const button = document.getElementById("sweep");
  const progress = document.getElementById("sweep-progress");
  const stats = document.getElementById("sweep-stats");
  const verdict = document.getElementById("sweep-verdict");

  button.addEventListener("click", () => {
    button.disabled = true;
    const models = prepared.models;
    const pairs = [];
    for (let i = 0; i < models.length; i += 1) {
      for (let j = i + 1; j < models.length; j += 1) pairs.push([models[i], models[j]]);
    }
    const done = [];
    let at = 0;

    function chunk() {
      const until = Math.min(at + 3, pairs.length);
      for (; at < until; at += 1) {
        const [a, b] = pairs[at];
        const seed = (at * 2654435761) >>> 0;
        const result = {};
        for (const kind of ["adaptive", "random"]) {
          const arena = new Arena(kind, kind, prepared, a, b, seed);
          while (!arena.finished()) if (!arena.step()) break;
          result[kind] = arena;
        }
        const leader = result.adaptive.decision
          ? result.adaptive.decision.leader === "a" ? a : b
          : null;
        done.push({
          a, b,
          adaptive: result.adaptive.decidedAt,
          random: result.random.decidedAt,
          usdAdaptive: result.adaptive.usd,
          usdRandom: result.random.usd,
          agrees: leader
            ? (leader === a ? a.accuracy > b.accuracy : b.accuracy > a.accuracy)
            : null,
        });
      }
      progress.textContent = `${at} of ${pairs.length} pairs`;
      if (at < pairs.length) {
        setTimeout(chunk, 0);
        return;
      }
      render(done);
    }

    function render(rows) {
      const settled = rows.filter((r) => r.adaptive !== null);
      const settledRandom = rows.filter((r) => r.random !== null);
      const bothSettled = rows.filter((r) => r.adaptive !== null && r.random !== null);
      const sooner = bothSettled.filter((r) => r.adaptive < r.random).length;
      const agree = settled.filter((r) => r.agrees).length;
      const usdA = rows.reduce((total, r) => total + r.usdAdaptive, 0);
      const median = (values) => {
        const sorted = [...values].sort((x, y) => x - y);
        return sorted[Math.floor(sorted.length / 2)];
      };
      const cards = [
        [`${settled.length} of ${rows.length}`, "pairs the adaptive test settled inside 300 questions"],
        [`${settledRandom.length} of ${rows.length}`, "pairs random selection settled, same budget"],
        [String(median(settled.map((r) => r.adaptive))), "questions the adaptive test needed, median"],
        [money(usdA), "to settle every pair it could, in total"],
      ];
      clear(stats);
      for (const [figure, caption] of cards) {
        const card = document.createElement("div");
        card.className = "stat";
        card.innerHTML = `<span class="figure">${figure}</span><span class="caption">${caption}</span>`;
        stats.appendChild(card);
      }
      stats.hidden = false;
      verdict.hidden = false;
      verdict.className = "verdict";
      verdict.innerHTML =
        `<strong>Where both methods settled a pair, the adaptive test got there first in ` +
        `${sooner} of ${bothSettled.length}.</strong> It agreed with the full-suite ranking in ` +
        `${agree} of the ${settled.length} pairs it settled. The disagreements are the pairs a ` +
        `tenth of an accuracy point apart, and they are the same construct gap the curve below ` +
        `shows: the test measures ability, a leaderboard reports an average of correct answers, ` +
        `and those two rankings do not agree perfectly even when every question has been asked.`;
      button.disabled = false;
      button.textContent = "Run all 66 pairs again";
    }

    chunk();
  });
}

// ------------------------------------------------------------------ hero

function setupHero() {
  const own = state.curves.panels.find((p) => p.key === "own");
  const v1 = state.curves.panels.find((p) => p.key === "v1");
  const hundred = own.cost.find((c) => c.items === 100);
  document.getElementById("hero-items").textContent = String(v1.equivalent.adaptive_items);
  document.getElementById("hero-items-label").textContent =
    `rank a shortlist as well as ${Math.round(v1.equivalent.random_items)} chosen at random`;
  document.getElementById("hero-cost").textContent = money(hundred.usd_adaptive);
  document.getElementById("hero-cost-label").textContent =
    `against ${money(own.full_suite_usd)} to ask all twelve every question`;
  const dead = state.items.benchmarks.reduce(
    (total, group) => total + group.share_low * group.n_items,
    0
  );
  const share = dead / state.items.n_items;
  document.getElementById("hero-dead").textContent = `${(share * 100).toFixed(0)}%`;
  document.getElementById("hero-suite").textContent = own.n_items.toLocaleString("en-CA");
  document.getElementById("hero-spend").textContent = money(own.full_suite_usd);
  const laptop = state.panel.models
    .filter((model) => !model.hosted && model.full_seconds)
    .sort((x, y) => y.full_seconds - x.full_seconds)[0];
  if (laptop) {
    document.getElementById("local-hours").textContent =
      `${(laptop.full_seconds / 3600).toFixed(1)} hours`;
  }
  document.getElementById("footer-provenance").textContent =
    `Item bank ${state.items.bank_version} (${state.items.bank_hash}): ` +
    `${state.items.n_models} models by ${state.items.n_items.toLocaleString("en-CA")} questions, ` +
    `fitted by marginal maximum a posteriori. The twelve models on this page were held out of ` +
    `every fit. Data written by "mselect demo build", ` +
    `${(state.sizes / 1024).toFixed(0)} KB of JSON in total, and nothing on this page is fetched ` +
    `from anywhere else.`;
}

// ------------------------------------------------------------------ the headline curve

function setupCurve() {
  const holder = document.getElementById("curve-switch");
  const colours = {
    adaptive: colour("--accent"),
    stratified: colour("--sim"),
    random: colour("--warn"),
  };
  let current = state.curves.panels[0].key;

  function render() {
    const panel = state.curves.panels.find((p) => p.key === current);
    drawTauCurve(document.getElementById("chart-curve"), panel, { colours });
    document.getElementById("curve-caption").textContent = panel.note;
    const verdict = document.getElementById("curve-verdict");
    if (panel.equivalent) {
      verdict.innerHTML =
        `<strong>${panel.equivalent.adaptive_items} adaptively chosen questions rank this panel ` +
        `as well as ${Math.round(panel.equivalent.random_items)} chosen at random</strong>, ` +
        `agreement ${panel.equivalent.tau.toFixed(3)}. That is the screening budget, and it is ` +
        `where the saving is.`;
    } else {
      verdict.innerHTML =
        `<strong>The bank ranks twelve models it never saw at ${panel.transfer.point.toFixed(3)}</strong> ` +
        `(${panel.transfer.lo.toFixed(2)} to ${panel.transfer.hi.toFixed(2)}), and 100 adaptively ` +
        `chosen questions reach that same figure for ` +
        `${money(panel.cost.find((c) => c.items === 100).usd_adaptive)} against ` +
        `${money(panel.full_suite_usd)} to ask everything. Twelve models is a wide interval and it ` +
        `is published wide.`;
    }
    for (const button of holder.children) {
      button.classList.toggle("on", button.dataset.key === current);
    }
  }

  for (const panel of state.curves.panels) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "legend-item";
    button.dataset.key = panel.key;
    button.textContent = panel.label;
    button.addEventListener("click", () => {
      current = panel.key;
      render();
    });
    holder.appendChild(button);
  }
  render();
}

// ------------------------------------------------------------------ the power calculator

function setupPower() {
  const { effects, abilities, powers, grid } = state.power;
  const effect = document.getElementById("effect");
  const ability = document.getElementById("ability");
  const confidence = document.getElementById("confidence");
  effect.max = String(effects.length - 1);
  ability.max = String(abilities.length - 1);
  confidence.max = String(powers.length - 1);

  const own = state.curves.panels.find((p) => p.key === "own");
  const perQuestion = own.full_suite_usd / (own.n_items * state.panel.n_models);
  const words = ["near the bottom of the panel", "well below the middle", "below the middle",
    "in the middle of the panel", "above the middle", "well above the middle",
    "near the top of the panel"];

  function render() {
    const e = effects[Number(effect.value)];
    const t = abilities[Number(ability.value)];
    const p = powers[Number(confidence.value)];
    const row = grid.find((g) => g.effect === e && g.ability === t && g.power === p);
    document.getElementById("effect-out").textContent = `${e} accuracy points`;
    document.getElementById("ability-out").textContent = words[Number(ability.value)];
    document.getElementById("confidence-out").textContent = `${(p * 100).toFixed(0)}% of the time`;
    document.getElementById("power-items").textContent = row.items.toLocaleString("en-CA");
    document.getElementById("power-five").textContent = (row.items * 5).toLocaleString("en-CA");
    document.getElementById("power-price").textContent = money(row.items * 2 * perQuestion);
    document.getElementById("power-share").textContent =
      `${((row.items / own.n_items) * 100).toFixed(row.items < 30 ? 1 : 0)}%`;
    const sentence = document.getElementById("power-sentence");
    if (row.capped) {
      sentence.innerHTML =
        `<strong>More questions than the bank holds.</strong> A difference this small cannot be ` +
        `established from this benchmark at all, at any budget, and that is worth knowing before ` +
        `the first call rather than after the last one.`;
    } else if (row.items > 1000) {
      sentence.innerHTML =
        `A difference of ${e} points is <strong>expensive to prove</strong>. If the decision ` +
        `turns on a gap this small, the honest move is usually to decide on price, latency or ` +
        `context window instead.`;
    } else {
      sentence.innerHTML =
        `<strong>${row.items} questions per model</strong> to catch a ${e}-point drop ${p * 100}% ` +
        `of the time. Fewer than that, and "no regression found" means only that you did not look ` +
        `hard enough.`;
    }
  }

  for (const control of [effect, ability, confidence]) control.addEventListener("input", render);
  render();
}

// ------------------------------------------------------------------ the item cloud

function setupCloud() {
  const holder = document.getElementById("cloud-switch");
  const canvas = document.getElementById("chart-cloud");
  const tones = [colour("--accent"), colour("--sim"), colour("--good"), colour("--warn")];
  let current = "all";

  function render() {
    const groups = state.items.benchmarks.map((group, index) => ({
      a: group.a,
      b: group.b,
      colour: tones[index % tones.length],
      dim: current !== "all" && current !== group.name,
    }));
    drawItemCloud(canvas, groups);
    const shown =
      current === "all"
        ? null
        : state.items.benchmarks.find((group) => group.name === current);
    document.getElementById("cloud-caption").textContent = shown
      ? `${shown.title}: ${shown.n_items.toLocaleString("en-CA")} questions in the bank, ` +
        `${shown.sampled} of them drawn here.`
      : `A seeded sample of up to ${state.items.sample_per_benchmark} questions per benchmark, ` +
        `out of ${state.items.n_items.toLocaleString("en-CA")}. The shares below are counted over ` +
        `every question, not over the sample.`;

    const stats = clear(document.getElementById("cloud-stats"));
    const rows = shown
      ? [
          ["measure nothing", shown.share_low],
          ["run backwards", shown.share_negative],
        ]
      : [
          [
            "measure nothing",
            state.items.benchmarks.reduce((t, g) => t + g.share_low * g.n_items, 0) /
              state.items.n_items,
          ],
          [
            "run backwards",
            state.items.benchmarks.reduce((t, g) => t + g.share_negative * g.n_items, 0) /
              state.items.n_items,
          ],
        ];
    for (const [label, value] of rows) {
      const card = document.createElement("div");
      card.className = "stat";
      card.innerHTML =
        `<span class="figure">${(value * 100).toFixed(1)}%</span>` +
        `<span class="caption">of ${shown ? shown.title : "the whole bank"} ${label}</span>`;
      stats.appendChild(card);
    }
    for (const button of holder.children) button.classList.toggle("on", button.dataset.key === current);
  }

  const entries = [{ name: "all", title: "Every benchmark" }, ...state.items.benchmarks];
  for (const entry of entries) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "legend-item";
    button.dataset.key = entry.name;
    button.textContent = entry.title;
    button.addEventListener("click", () => {
      current = entry.name;
      render();
    });
    holder.appendChild(button);
  }
  render();
  window.addEventListener("resize", render);
}

// ------------------------------------------------------------------ the three experiments

function setupHidden() {
  const holder = document.getElementById("hidden-switch");
  const svg = document.getElementById("chart-hidden");
  const experiments = state.experiments;
  let current = "noise";

  const views = {
    noise: {
      label: "Noise",
      lead:
        "The same model, the same 500 questions, asked twice a day apart with randomness turned " +
        "down as far as each vendor allows. Whatever moves between the two runs is noise, and it " +
        "is the floor under every claim that a new version dropped two points.",
      rows: () =>
        Object.entries(experiments.retest.agreement)
          .map(([alias, agreement]) => ({ label: alias, value: 1 - agreement }))
          .sort((x, y) => y.value - x.value),
      format: (v) => `${(v * 100).toFixed(1)}%`,
      caption: "Share of answers that changed between two identical administrations.",
      verdict: () =>
        `<strong>A benchmark score moves by up to two points with nothing changed at all</strong>, ` +
        `and it moves in both directions equally, so it is a random walk rather than drift. ` +
        `Pooled across twelve models the flip rate is ` +
        `${(experiments.retest.pooled_flip_rate * 100).toFixed(1)}%, but the worst hosted model is ` +
        `${(experiments.retest.worst_hosted_flip_rate * 100).toFixed(1)}%, and a release gate sized ` +
        `on the pooled figure would understate its own noise floor by half.`,
    },
    order: {
      label: "Option order",
      lead:
        "The same 300 multiple-choice questions, asked four times with the correct answer moved " +
        "to a different position each time. Anything that changes is a formatting artefact, not " +
        "capability. The bars are net of each model's own instability, measured above.",
      rows: () =>
        experiments.order
          .map((model) => ({ label: model.alias, value: model.net, lo: model.lo, hi: model.hi }))
          .sort((x, y) => y.value - x.value),
      format: (v) => `${(v * 100).toFixed(1)}%`,
      caption:
        "Share of questions whose answer flips when the right option moves, net of noise, with a " +
        "bootstrap interval.",
      verdict: () =>
        `<strong>Every model in the panel does worse when the right answer sits at A.</strong> ` +
        `On the weakest model, 43 percent of questions change answer on option order alone: what ` +
        `a leaderboard would report as its score is substantially a fact about formatting. On the ` +
        `strongest, it is under one percent. Both of those are in the same published table, at the ` +
        `same apparent precision.`,
    },
    framing: {
      label: "Prompt framing",
      lead:
        "The same questions under three prompt templates: answer only, the letter alone, and a " +
        "short chain of reasoning. This puts a number on what this project's own answer-only " +
        "format costs, which is the honest way to report a design decision made to save money.",
      rows: () =>
        experiments.framing.models
          .map((model) => ({ label: model.alias, value: model.cost, lo: model.lo, hi: model.hi }))
          .sort((x, y) => y.value - x.value),
      format: (v) => `${(v * 100).toFixed(1)}%`,
      caption:
        "Accuracy given up by asking for an answer rather than letting the model reason, with a " +
        "bootstrap interval. An interval that spans zero is no cost at all.",
      verdict: () =>
        `<strong>For ten of the twelve models the cost is indistinguishable from zero</strong>, and ` +
        `the template explains at most 0.3 percent of the variance in whether an answer is right. ` +
        `Which question was asked explains most of it. Two models are exceptions and they are ` +
        `named rather than averaged away.`,
    },
  };

  function render() {
    const view = views[current];
    document.getElementById("hidden-lead").textContent = view.lead;
    drawBars(svg, view.rows().map((row) => ({ ...row, colour: colour("--accent") })), {
      format: view.format,
    });
    document.getElementById("hidden-caption").textContent = view.caption;
    document.getElementById("hidden-verdict").innerHTML = view.verdict();
    for (const button of holder.children) button.classList.toggle("on", button.dataset.key === current);
  }

  for (const [key, view] of Object.entries(views)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "legend-item";
    button.dataset.key = key;
    button.textContent = view.label;
    button.addEventListener("click", () => {
      current = key;
      render();
    });
    holder.appendChild(button);
  }
  render();
}

// ------------------------------------------------------------------ start

async function start() {
  const [panel, curves, items, experiments, power, index] = await Promise.all(
    ["panel", "curves", "items", "experiments", "power", "index"].map(load)
  );
  Object.assign(state, { panel, curves, items, experiments, power });
  state.sizes = index.files.reduce((total, file) => total + file.bytes, 0);
  setupHero();
  setupRace();
  setupCurve();
  setupPower();
  setupCloud();
  setupHidden();
}

start().catch((error) => {
  const note = document.createElement("p");
  note.className = "verdict warn";
  note.textContent = `The data files did not load: ${error.message}. This page reads them with fetch, so it needs to be served rather than opened from disk: "uv run mselect demo serve".`;
  document.querySelector("main").prepend(note);
});
