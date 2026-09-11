# Public per-item results: what was verified, on 2026-09-10

PLAN.md section 3.1 named two public sources and said, in the risk table, that week 1 verifies
availability and format before anything depends on them. This is that verification. It was done
first, and it changed the plan.

## Open LLM Leaderboard "details" datasets: gated, not usable unattended

The plan's primary source was the Hugging Face `open-llm-leaderboard/<model>-details` datasets.
They still exist and still contain per-sample results:

- `GET https://huggingface.co/api/datasets?author=open-llm-leaderboard` returns the datasets.
- `GET https://huggingface.co/api/datasets/open-llm-leaderboard/gpt2-details/tree/main?recursive=true`
  lists `samples_<task>_<timestamp>.json` files per task, so the per-sample data is there.
- The auto-converted Parquet is also listed, per task and per run, under
  `/api/datasets/<id>/parquet`.

But every route that returns bytes answers **HTTP 401** without a token:

| Request | Result |
|---|---|
| `huggingface.co/datasets/open-llm-leaderboard/gpt2-details/resolve/main/<samples file>` | 401 |
| `huggingface.co/api/datasets/open-llm-leaderboard/gpt2-details/parquet/.../0.parquet` | 401 |
| `datasets-server.huggingface.co/rows?dataset=open-llm-leaderboard%2Fgpt2-details...` | 401 |
| `datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main&split=test` | 200 |

The dataset metadata says why: `"gated": "auto"`. Access is granted automatically, but only to a
signed-in account, so the data needs a Hugging Face account and a read token in the environment.
That is free and takes a few minutes, but it is a credential, and nothing in this repository
takes a credential it has not been given deliberately.

**Decision.** HELM becomes the primary source and the leaderboard details become a documented
follow-up. When a token exists, `mselect/data/` gains an `oll.py` loader beside `helm.py`, the
bank version goes to v2, and the extra models widen the panel. The method does not change.

## HELM public buckets: open, and richer than expected

`https://storage.googleapis.com/crfm-helm-public/<project>/benchmark_output/` needs no account,
no token and no rate limit worth mentioning. Releases found on 2026-09-10:

| Project | Latest release | Runs | Models |
|---|---|---:|---:|
| `lite` | v1.13.0 (2025-01-10) | 2,546 | 91 |
| `mmlu` | v1.13.0 | 4,503 | 79 |
| `capabilities` | v1.15.0 | 340 | 68 |
| `safety` | v1.17.0 | not used: not binary-scored per item | |

Per run, the bucket gives `per_instance_stats.json` (one record per instance, with named
statistics including the binary correctness metric), `instances.json` (the item text, its
references and which reference is the answer key), and at the release level `schema.json` (model
metadata: creator, access, release date, parameter count) and `runs_to_run_suites.json`.

### Which scenarios are in, and why

Binary-scored per instance, not model-judged:

| Benchmark | Project | Metric | Kind |
|---|---|---|---|
| MMLU | `mmlu` | `exact_match` | multiple choice |
| MMLU-Pro | `capabilities` | `chain_of_thought_correctness` | multiple choice |
| GPQA | `capabilities` | `chain_of_thought_correctness` | multiple choice |
| MATH | `lite` | `math_equiv_chain_of_thought` | free response |
| GSM8K | `lite` | `final_number_exact_match` | free response |
| MedQA | `lite` | `exact_match` | multiple choice |
| LegalBench | `lite` | `quasi_exact_match` | multiple choice |
| OpenBookQA | `lite` | `exact_match` | multiple choice |

Excluded, with the reason:

- `ifeval`: the per-instance score is the fraction of instructions satisfied, not a binary
  outcome. PLAN.md section 2 keeps the bank binary-scored.
- `omni_math`, `wildbench`: model-judged. Judges are out of scope by the same section.
- `narrative_qa`, `natural_qa`, `wmt_14`: scored with F1 or BLEU against references.

### What the fetch actually costs

7,074 files, 318 MB on disk gzipped, a few minutes over a home connection, and nothing in
money. Every file is cached under `data/raw/helm/` keyed by the SHA-256 of its URL, and every
fetch appends a line to `data/raw/helm/provenance.jsonl` with the URL, the SHA-256 of the bytes
and the date. A rerun of the bank build costs nothing and produces the same bytes.

## What this changed about the plan

1. **Source order.** HELM first, Open LLM Leaderboard deferred to a token. PLAN.md section 3.1
   amended.
2. **Size.** The plan targeted at least 100 models by 3,000 to 5,000 items. HELM gives 150
   models by 20,365 items after the response floor, which is four times the item target.
3. **Shape.** The matrix is not rectangular: models were run on different HELM projects, so
   54 percent of cells are observed. Item response theory handles that, but "the full-suite
   ranking" is only defined over a nearly complete block, so the headline claim is measured on
   one: 74 models by 18,921 items, 99.1 percent complete. PLAN.md section 13 records this.
4. **A free reliability number.** Because HELM re-ran some models across suites, 8,431 model-item
   cells appear twice. They agree 95.3 percent of the time. That is not the test-retest
   experiment the plan specifies (which needs own runs at temperature 0), but it is a real
   measured floor on how much of a benchmark score is noise, obtained for nothing.
