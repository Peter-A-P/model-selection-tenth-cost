import { readFileSync } from "node:fs";
const mod = await import("./irt.js");
const { Posterior, Selector, RandomSelector, rng, separated, unpackBits } = mod;
const panel = JSON.parse(readFileSync("data/panel.json", "utf8"));
const n = panel.n_items;
const items = {
  a: Float64Array.from(panel.items.a),
  b: Float64Array.from(panel.items.b.map(v => v === null ? 0 : v)),
  c: Float64Array.from(panel.items.c),
  stratum: Int32Array.from(panel.items.stratum),
  usable: unpackBits(panel.items.usable, n),
};
const weights = Float64Array.from(panel.weights);
const models = panel.models.map(m => ({ ...m, responses: unpackBits(m.correct, n) }));
for (const m of models) {
  let s = 0; for (let i=0;i<n;i++) s += m.responses[i];
  console.log(m.alias.padEnd(18), (s/n).toFixed(3), "json", m.accuracy.toFixed(3));
}
function run(a, b, kind, seed, cap=300) {
  const post = [new Posterior(), new Posterior()];
  const sel = [0,1].map(() => kind==="adaptive" ? new Selector(items, weights) : new RandomSelector(items));
  const draw = rng(seed);
  let usd = 0;
  for (let step=1; step<=cap; step++) {
    for (let side=0; side<2; side++) {
      const it = sel[side].next(post[side].theta, draw);
      if (it < 0) return {decided:null, usd};
      const model = side===0?a:b;
      post[side].update(items.a[it], items.b[it], items.c[it], model.responses[it]===1);
      usd += model.cost_micro[it]/1e6;
    }
    const v = separated(post[0], post[1]);
    if (v.separated && step>=10) return {decided:step, leader:v.leader, usd, thetas:[post[0].theta, post[1].theta]};
  }
  return {decided:null, usd, thetas:[post[0].theta, post[1].theta]};
}
const pairs = [["together-open-b","anthropic-haiku"],["openai-mid","together-open-a"],["anthropic-opus","google-frontier"],["local-mid-a","local-small-a"],["anthropic-haiku","local-mid-a"]];
const find = x => models.find(m=>m.alias===x);
for (const [x,y] of pairs) {
  const A=find(x), B=find(y);
  const ad = run(A,B,"adaptive",7);
  const rd = run(A,B,"random",7);
  console.log(`${x} vs ${y}: adaptive ${ad.decided ?? "none"} ($${ad.usd.toFixed(3)}) leader=${ad.leader ?? "-"} | random ${rd.decided ?? "none"} ($${rd.usd.toFixed(3)}) truth ${A.accuracy.toFixed(3)} vs ${B.accuracy.toFixed(3)}`);
}
