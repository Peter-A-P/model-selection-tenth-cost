// The adaptive test itself, in the browser.
//
// This file is a port of three modules in the repository: mselect/irt/model.py (the response
// model and its information function), mselect/cat/estimate.py (the running posterior) and
// mselect/cat/select.py (which item to ask next). It is short on purpose. The claim the whole
// project makes is that choosing the next question well is worth more than asking a thousand
// of them, and the arithmetic behind that claim fits on two screens, so it is shipped where a
// reader can see it rather than described.
//
// Nothing here invents an answer. Every response the page scores was recorded when the twelve
// models were administered the suite, and is shipped in data/panel.json one bit per cell.

"use strict";

// The prior the fit used and the estimator reads: 61 points from -4.5 to +4.5, weighted
// standard normal. Same grid as mselect/irt/fit.py Quadrature.normal.
const NODES = 61;
const SPAN = 4.5;

export const quadrature = (() => {
  const nodes = new Float64Array(NODES);
  const weights = new Float64Array(NODES);
  let total = 0;
  for (let i = 0; i < NODES; i += 1) {
    const x = -SPAN + (2 * SPAN * i) / (NODES - 1);
    nodes[i] = x;
    weights[i] = Math.exp(-0.5 * x * x);
    total += weights[i];
  }
  for (let i = 0; i < NODES; i += 1) weights[i] /= total;
  return { nodes, weights, logWeights: weights.map(Math.log) };
})();

function sigmoid(z) {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

// P(correct) for one item at one ability. Two parameters: a is how sharply the item separates
// ability just above its threshold from ability just below, b is where that threshold sits.
export function probability(theta, a, b, c) {
  const s = sigmoid(a * (theta - b));
  return c + (1 - c) * s;
}

// Fisher information: how much asking this item at this ability would tell you. This is the
// whole selection rule. An item everybody gets right carries none, an item nobody gets right
// carries none, and an item that splits models at exactly this ability carries the most.
export function information(theta, a, b, c) {
  const p = Math.min(Math.max(probability(theta, a, b, c), 1e-12), 1 - 1e-12);
  const ratio = (p - c) / (1 - c);
  return a * a * ((1 - p) / p) * ratio * ratio;
}

// A running posterior over ability, updated one item at a time. Carrying the log likelihood
// over the grid is what makes this cheap: one answered item is one addition per grid point,
// so a 300-item test costs 300 additions rather than 300 rescorings.
export class Posterior {
  constructor() {
    this.logLik = new Float64Array(NODES);
    this.count = 0;
  }

  update(a, b, c, correct) {
    const { nodes } = quadrature;
    for (let i = 0; i < NODES; i += 1) {
      const p = Math.min(Math.max(probability(nodes[i], a, b, c), 1e-10), 1 - 1e-10);
      this.logLik[i] += correct ? Math.log(p) : Math.log1p(-p);
    }
    this.count += 1;
  }

  density() {
    const { logWeights } = quadrature;
    let top = -Infinity;
    const raw = new Float64Array(NODES);
    for (let i = 0; i < NODES; i += 1) {
      raw[i] = this.logLik[i] + logWeights[i];
      if (raw[i] > top) top = raw[i];
    }
    let total = 0;
    for (let i = 0; i < NODES; i += 1) {
      raw[i] = Math.exp(raw[i] - top);
      total += raw[i];
    }
    for (let i = 0; i < NODES; i += 1) raw[i] /= total;
    return raw;
  }

  // Expected a posteriori: the point estimate the next item is chosen against.
  get theta() {
    const post = this.density();
    const { nodes } = quadrature;
    let mean = 0;
    for (let i = 0; i < NODES; i += 1) mean += post[i] * nodes[i];
    return mean;
  }

  get se() {
    const post = this.density();
    const { nodes } = quadrature;
    let mean = 0;
    let second = 0;
    for (let i = 0; i < NODES; i += 1) {
      mean += post[i] * nodes[i];
      second += post[i] * nodes[i] * nodes[i];
    }
    return Math.sqrt(Math.max(second - mean * mean, 0));
  }

  // An equal-tailed credible interval read straight off the posterior rather than a normal
  // approximation to it. With ten items the posterior is visibly skewed and the approximation
  // would hand a model near the top of the panel an interval the bank cannot measure inside.
  interval(level = 0.95) {
    const post = this.density();
    const { nodes } = quadrature;
    const cumulative = new Float64Array(NODES);
    let running = 0;
    for (let i = 0; i < NODES; i += 1) {
      running += post[i];
      cumulative[i] = running;
    }
    const tail = (1 - level) / 2;
    return [interpolate(cumulative, nodes, tail), interpolate(cumulative, nodes, 1 - tail)];
  }
}

function interpolate(xs, ys, target) {
  if (target <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i += 1) {
    if (xs[i] >= target) {
      const span = xs[i] - xs[i - 1];
      const share = span > 0 ? (target - xs[i - 1]) / span : 0;
      return ys[i - 1] + share * (ys[i] - ys[i - 1]);
    }
  }
  return ys[ys.length - 1];
}

// A small seeded generator, so the same run repeats and two methods can be given the same
// stream. The repository's own runs use numpy's generator; this one cannot reproduce that
// stream bit for bit, which is why the page describes a run as a fresh draw rather than as a
// replay of the published figure.
export function rng(seed) {
  let state = seed >>> 0;
  return function next() {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Maximum Fisher information, inside the benchmark that is furthest behind its share of the
// suite, with a random pick among the best few.
//
// The content balancing is not decoration. Without it the selector takes whichever benchmark
// happens to hold the sharpest items near this model's ability, and the short test then
// measures a different thing from the full suite it is being compared against.
export class Selector {
  constructor(items, weights, { randomesque = 5, balance = true } = {}) {
    this.items = items;
    this.weights = weights;
    this.randomesque = randomesque;
    this.balance = balance;
    this.counts = new Float64Array(weights.length);
    this.used = new Uint8Array(items.a.length);
    this.asked = 0;
  }

  next(theta, draw) {
    const { a, b, c, stratum, usable } = this.items;
    let restrict = -1;
    if (this.balance) {
      const total = Math.max(this.asked, 1);
      let worst = -Infinity;
      for (let s = 0; s < this.weights.length; s += 1) {
        const deficit = this.weights[s] * total - this.counts[s];
        if (deficit > worst && this.hasLeft(s)) {
          worst = deficit;
          restrict = s;
        }
      }
    }
    let best = [];
    let cutoff = -Infinity;
    for (let i = 0; i < a.length; i += 1) {
      if (this.used[i] || !usable[i]) continue;
      if (restrict >= 0 && stratum[i] !== restrict) continue;
      const info = information(theta, a[i], b[i], c[i]);
      if (best.length < this.randomesque) {
        best.push([info, i]);
        best.sort((x, y) => y[0] - x[0]);
        cutoff = best[best.length - 1][0];
      } else if (info > cutoff) {
        best[best.length - 1] = [info, i];
        best.sort((x, y) => y[0] - x[0]);
        cutoff = best[best.length - 1][0];
      }
    }
    if (best.length === 0) return -1;
    const pick = best[Math.floor(draw() * best.length) % best.length][1];
    this.used[pick] = 1;
    this.counts[stratum[pick]] += 1;
    this.asked += 1;
    return pick;
  }

  hasLeft(s) {
    const { stratum, usable } = this.items;
    for (let i = 0; i < stratum.length; i += 1) {
      if (!this.used[i] && usable[i] && stratum[i] === s) return true;
    }
    return false;
  }
}

// The baseline worth beating: items drawn uniformly from what the model answered. It has no
// principled way of saying when it has seen enough, which is half of what is being compared.
export class RandomSelector {
  constructor(items) {
    this.items = items;
    this.used = new Uint8Array(items.a.length);
    this.pool = [];
    for (let i = 0; i < items.a.length; i += 1) if (items.usable[i]) this.pool.push(i);
    this.asked = 0;
  }

  next(_theta, draw) {
    let guard = 0;
    while (guard < 10000) {
      const pick = this.pool[Math.floor(draw() * this.pool.length) % this.pool.length];
      if (!this.used[pick]) {
        this.used[pick] = 1;
        this.asked += 1;
        return pick;
      }
      guard += 1;
    }
    return -1;
  }
}

// Are these two models separated on the evidence so far? Overlapping intervals are reported
// as "not separated at this budget", never as "the same": the second is a claim the evidence
// does not support, and it is the claim a benchmark table invites people to make.
export function separated(first, second, level = 0.95) {
  const [loA, hiA] = first.interval(level);
  const [loB, hiB] = second.interval(level);
  const apart = loA > hiB || loB > hiA;
  return {
    separated: apart,
    leader: apart ? (first.theta > second.theta ? "a" : "b") : null,
    intervalA: [loA, hiA],
    intervalB: [loB, hiB],
  };
}

// Unpack a base64 bit string written by mselect/demo/build.py.
export function unpackBits(encoded, length) {
  const binary = atob(encoded);
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    const byte = binary.charCodeAt(i >> 3);
    out[i] = (byte >> (7 - (i & 7))) & 1;
  }
  return out;
}
