import { readFileSync } from "node:fs";
const { Posterior, Selector, RandomSelector, rng, separated, unpackBits } = await import("./irt.js");
const panel = JSON.parse(readFileSync("data/panel.json", "utf8"));
const n = panel.n_items;
const items = { a: Float64Array.from(panel.items.a), b: Float64Array.from(panel.items.b.map(v=>v??0)), c: Float64Array.from(panel.items.c), stratum: Int32Array.from(panel.items.stratum), usable: unpackBits(panel.items.usable, n) };
const weights = Float64Array.from(panel.weights);
const models = panel.models.map(m => ({ ...m, responses: unpackBits(m.correct, n) }));
function run(a,b,kind,seed,cap=300){
  const post=[new Posterior(),new Posterior()];
  const sel=[0,1].map(()=>kind==="adaptive"?new Selector(items,weights):new RandomSelector(items));
  const draw=rng(seed); let usd=0;
  for(let step=1;step<=cap;step++){
    for(let s=0;s<2;s++){const it=sel[s].next(post[s].theta,draw); if(it<0)return{decided:null,usd};
      const m=s===0?a:b; post[s].update(items.a[it],items.b[it],items.c[it],m.responses[it]===1); usd+=m.cost_micro[it]/1e6;}
    const v=separated(post[0],post[1]); if(v.separated&&step>=10)return{decided:step,leader:v.leader,usd};
  }
  return {decided:null,usd};
}
const rows=[];
for(let i=0;i<models.length;i++)for(let j=i+1;j<models.length;j++){
  const A=models[i],B=models[j];
  const ad=run(A,B,"adaptive",7), rd=run(A,B,"random",7);
  const correct = ad.decided ? ((ad.leader==="a"?A:B).accuracy > (ad.leader==="a"?B:A).accuracy) : null;
  rows.push({p:`${A.alias} vs ${B.alias}`, gap:Math.abs(A.accuracy-B.accuracy), ad:ad.decided, rd:rd.decided, correct, usd:ad.usd});
}
rows.sort((x,y)=>x.gap-y.gap);
for(const r of rows) console.log(`${r.p.padEnd(40)} gap=${(r.gap*100).toFixed(1)}pt ad=${String(r.ad??"-").padStart(4)} rand=${String(r.rd??"-").padStart(4)} ok=${r.correct} $${r.usd.toFixed(3)}`);
const dec=rows.filter(r=>r.ad);
console.log("\nseparated by adaptive:",dec.length,"of",rows.length,"| leader correct:",dec.filter(r=>r.correct).length);
console.log("random separated:",rows.filter(r=>r.rd).length);
console.log("adaptive earlier than random:",rows.filter(r=>r.ad&&(!r.rd||r.ad<r.rd)).length,"of",dec.length);
