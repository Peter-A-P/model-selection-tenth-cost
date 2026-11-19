// The film's data: one adaptive and one random run of a pair, replayed exactly as the page
// replays it (demo/irt.js, the page's seed for the pair), plus the same pair rerun at forty
// other seeds so the film can say how typical its one run is. Reads demo/data/*.json, which
// `mselect demo build` writes, and writes video/trace.json for video/film.html to draw.
//
//     node video/trace.mjs [first-alias] [second-alias]
//
// The pair defaults to DeepSeek-V4.1-Flash against Jev 1.14. A pair with a decision model in
// it is replayed on the multiple-choice frame (decisions.json), as the page does.

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  Posterior,
  RandomSelector,
  Selector,
  quadrature,
  rng,
  separated,
  unpackBits,
} from "../demo/irt.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, "..", "demo", "data");
const MAX_QUESTIONS = 300; // demo/app.js's budget
const SEEDS = 40;

const [firstAlias = "together-open-c", secondAlias = "decision-jev-1.14"] = process.argv.slice(2);
const load = (name) => JSON.parse(readFileSync(join(DATA, `${name}.json`), "utf8"));
const text = load("panel");
const decisions = load("decisions");
const isDecision = new Set(decisions.models.filter((m) => m.kind === "decision").map((m) => m.alias));
const panel = isDecision.has(firstAlias) || isDecision.has(secondAlias) ? decisions : text;

const items = {
  a: Float64Array.from(panel.items.a),
  b: Float64Array.from(panel.items.b.map((v) => (v === null ? 0 : v))),
  c: Float64Array.from(panel.items.c),
  stratum: Int32Array.from(panel.items.stratum),
  usable: unpackBits(panel.items.usable, panel.n_items),
};
const weights = Float64Array.from(panel.weights);
const model = (alias) => {
  const m = panel.models.find((x) => x.alias === alias);
  if (!m) throw new Error(`${alias} is not in the ${panel.kind ?? "text"} frame`);
  return { ...m, responses: unpackBits(m.correct, panel.n_items) };
};
const pair = [model(firstAlias), model(secondAlias)];

// demo/app.js's Arena.step, recording what the film draws at every step.
function run(kind, seed, record) {
  const posteriors = [new Posterior(), new Posterior()];
  const selectors = pair.map(() => (kind === "adaptive" ? new Selector(items, weights) : new RandomSelector(items)));
  const draw = rng(seed);
  const steps = [];
  let usd = 0;
  let decidedAt = null;
  let leader = null;
  for (let n = 1; n <= MAX_QUESTIONS; n += 1) {
    const asked = [];
    for (let side = 0; side < 2; side += 1) {
      const item = selectors[side].next(posteriors[side].theta, draw);
      if (item < 0) return { steps, decidedAt, leader, usd };
      const correct = pair[side].responses[item] === 1;
      posteriors[side].update(items.a[item], items.b[item], items.c[item], correct);
      usd += pair[side].cost_micro[item] / 1e6;
      asked.push({ item, correct });
    }
    const verdict = separated(posteriors[0], posteriors[1]);
    if (record) {
      steps.push({
        n,
        usd: Number(usd.toFixed(6)),
        sides: posteriors.map((p, side) => ({
          item: asked[side].item,
          correct: asked[side].correct,
          theta: Number(p.theta.toFixed(4)),
          interval: p.interval().map((v) => Number(v.toFixed(4))),
          density: Array.from(p.density(), (v) => Number(v.toFixed(5))),
        })),
      });
    }
    if (verdict.separated && n >= 10) {
      decidedAt = n;
      leader = verdict.leader === "a" ? 0 : 1;
      break;
    }
  }
  return { steps, decidedAt, leader, usd };
}

// The page's seed for a pair (demo/app.js reset()), and the sweep's seeds for the reruns.
const pageSeed = (pair[0].alias.length * 31 + pair[1].alias.length * 17 + 7) >>> 0;
const arms = Object.fromEntries(
  ["adaptive", "random"].map((kind) => [kind, run(kind, pageSeed, true)])
);

// Wilson score interval for a proportion, 95%.
function wilson(k, n) {
  const z = 1.959964;
  const p = k / n;
  const d = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / d;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [centre - half, centre + half];
}
const median = (xs) => {
  const s = [...xs].sort((x, y) => x - y);
  return s.length === 0 ? null : s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
const reruns = {};
for (const kind of ["adaptive", "random"]) {
  const runs = [];
  for (let at = 1; at <= SEEDS; at += 1) runs.push(run(kind, (at * 2654435761) >>> 0, false));
  const done = runs.filter((r) => r.decidedAt !== null);
  reruns[kind] = {
    seeds: SEEDS,
    settled: done.length,
    interval: wilson(done.length, SEEDS),
    median: median(done.map((r) => r.decidedAt)),
    leaders: [...new Set(done.map((r) => pair[r.leader].label))],
  };
}

// The page's headline figures (demo/app.js setupHero), from the same files the page reads.
const curves = load("curves");
const bank = load("items");
const own = curves.panels.find((p) => p.key === "own");
const v1 = curves.panels.find((p) => p.key === "v1");
const hundred = own.cost.find((c) => c.items === 100);
const adaptive100 = own.curve.find((p) => p.method === "adaptive" && p.items === 100);
const ten = v1.curve.find((p) => p.method === "adaptive" && p.items === v1.equivalent.adaptive_items);
const dead = bank.benchmarks.reduce((total, g) => total + g.share_low * g.n_items, 0) / bank.n_items;
const headline = {
  models: own.n_models,
  usd_hundred: hundred.usd_adaptive,
  usd_full: own.full_suite_usd,
  tau_hundred: [adaptive100.tau, adaptive100.lo, adaptive100.hi],
  tau_full: own.transfer.point,
  adaptive_items: v1.equivalent.adaptive_items,
  random_items: Math.round(v1.equivalent.random_items),
  archive_models: v1.n_models,
  tau_ten: ten ? [ten.tau, ten.lo, ten.hi] : null,
  dead_share: dead,
};

const usable = [];
for (let i = 0; i < panel.n_items; i += 1) if (items.usable[i]) usable.push(i);
const out = {
  frame: panel === decisions ? "multiple-choice" : "text",
  n_items: panel.n_items,
  benchmarks: panel.benchmarks.map((b) => b.title),
  nodes: Array.from(quadrature.nodes),
  items: {
    index: usable,
    a: usable.map((i) => Number(items.a[i].toFixed(3))),
    b: usable.map((i) => Number(items.b[i].toFixed(3))),
    stratum: usable.map((i) => items.stratum[i]),
  },
  models: pair.map((m) => ({
    alias: m.alias,
    label: m.label,
    kind: m.kind ?? "text",
    accuracy: m.accuracy,
    full_usd: m.full_usd,
    full_seconds: m.full_seconds ?? null,
  })),
  seed: pageSeed,
  adaptive: arms.adaptive,
  random: arms.random,
  reruns,
  headline,
};
writeFileSync(join(HERE, "trace.json"), JSON.stringify(out));
console.log(
  `${pair[0].label} against ${pair[1].label}, ${out.frame} frame of ${panel.n_items}: ` +
    `adaptive ${arms.adaptive.decidedAt ?? "undecided"} for US$${arms.adaptive.usd.toFixed(4)}, ` +
    `random ${arms.random.decidedAt ?? "undecided"} after ${arms.random.steps.length}; ` +
    `reruns ${JSON.stringify(reruns)}`
);
