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
follow-up. When a token exists, `mselect/data/` gains a loader beside `helm.py`, the bank version
goes to v2, and the extra models widen the panel. The method does not change.

> Superseded on 2026-09-11, and one sentence above is wrong: the sentence reading "access is
> granted automatically, but only to a signed-in account" is right, and the conclusion drawn
> from `gated: auto` in the first draft of the loader, that each repository has to be asked for,
> is not. A token is the whole gate. The loader is `mselect/data/ollm.py`, bank v2 exists, and
> the measurements are in the next section. This section is kept as the record of what was known
> on 2026-09-10.

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

## The Open LLM Leaderboard, reached: what a token changed, on 2026-09-11

A Hugging Face read token now exists, so the deferred source above was verified properly. Three
things came out of it, and two of them corrected something this document got wrong.

### The gate wants a token, not a request

`"gated": "auto"` was read, when bank v1 was built, as "each repository has to be asked for".
It does not. The same range request answers **401 without a token and 206 with one**, for a
repository the account has never touched, and nothing is recorded against the account:

```
GET .../yasserrmd__Text2SQL-1.5B-details/resolve/refs%2Fconvert%2Fparquet/...   401
GET the same URL with Authorization: Bearer hf_...                             206
```

This matters more than tidiness. The loader in `mselect/data/ollm.py` **writes nothing, to
Hugging Face or anywhere else**. It has no POST in it. There was an `ask-access` call in an
early draft; it was removed once the 401/206 pair above was measured, because it was asking for
something the token already had.

For the record, the `POST /datasets/<id>/ask-access` endpoint does exist and does return 200,
but it is rate-limited by IP rather than by token and its 429 body says "make sure you pass a
HF_TOKEN" even when one is passed. It is a web form, not an API, and this project does not use
it.

### Python could not verify the certificate; the browser could

Every request from Python to `huggingface.co` failed with `CERTIFICATE_VERIFY_FAILED` while
PowerShell and the browser on the same machine were fine. The network inspects TLS: the proxy
presents its own certificate, signed by a root that the Windows certificate store trusts and
that certifi's bundle, which Python uses by default, does not.

The fix is `truststore`, which verifies against the operating system's trust store instead, and
it is a dependency for this reason alone (`ollm.ssl_context`). It is not a way of skipping
verification: certificates are still verified, against the trust store the machine actually
has. HELM's Google Cloud Storage host is not intercepted, which is why bank v1 built without
noticing any of this.

Two consequences worth writing down. Anything fetched here is visible in clear to whatever does
the inspection, including the token in the `Authorization` header, which is an argument for the
token being read-only and short-lived. And any machine on a network that does *not* inspect TLS
will also work, because the system trust store is a superset of certifi's in that case.

### Reading two columns of a seventy-megabyte file

The details repositories are auto-converted to Parquet under `refs/convert/parquet`, with a
`latest` split per task, so "the most recent run of each task" is a URL rather than a search.
The files are mostly prompt and response text:

| One model's run | Source JSON | Parquet | Columns this project keeps |
|---|---:|---:|---:|
| MMLU-Pro | 328 MB | 71.2 MB | 1.8 MB |
| GPQA main | 6.1 MB | 2.7 MB | 0.17 MB |
| All 36 tasks | 486 MB | 102 MB | about 2 MB |

So every read is a column projection over HTTP range requests (`ollm._RemoteFile`). Parquet
puts its schema in the footer, so a reader that can seek pulls the footer and then only the
column chunks it asked for. Across the 400-model panel that is **under a gigabyte transferred
instead of about forty**. polars cannot do this over a file-like object, which is why pyarrow is
a dependency: it reads a file-like object straight through, which measured 2.7 MB pulled for a
2.7 MB file in 333 requests. pyarrow pulled 166 KB in six.

**What the cache stores is a departure from the HELM cache, deliberately.** `data/raw/helm/`
stores the source bytes. `data/raw/ollm/` stores the projected columns, because the source is a
hundred times larger than the information taken from it and keeping it would cost 40 GB to save
a download this project makes once. Every entry still carries a provenance line with the source
URL, the remote file's size in bytes, the columns taken and the row count, so the extract can be
re-derived and checked.

### Which tasks are in, and why

| Benchmark | Tasks | Metric | Items |
|---|---:|---|---:|
| BBH | 24 | `acc_norm` | 6,511 |
| MMLU-Pro | 1 | `acc` | 12,032 |
| MuSR | 3 | `acc_norm` | 756 |
| MATH level 5 | 7 | `exact_match` | 1,324 |
| GPQA main | 1 | `acc_norm` | 448 |

Excluded, with the reason:

- `leaderboard_ifeval`: the per-item score is the fraction of instructions a response satisfied.
  Binary items only, the same rule that excluded HELM's ifeval.
- `leaderboard_arc_challenge`: present in the repositories, but not part of the v2 average and
  not re-run for every submission. Including it would put holes in a matrix whose advantage over
  bank v1 is that it has none.
- `leaderboard_bbh_fewshot_*` and the non-hard `leaderboard_math_*` configs: an earlier naming
  that only some submissions carry. Same reason.
- `leaderboard_gpqa_diamond` and `leaderboard_gpqa_extended`: subsets of and overlapping with
  `gpqa_main`, so taking all three would put the same question in the bank up to three times and
  call it independent evidence.

### How the panel was chosen

4,485 submissions have a details repository. The panel is 400 of them, and the two rules that
pick it are there to make item parameters identifiable, not to be fair to anyone:

1. **Forty equal-width bands of the leaderboard's own average, ten from each.** A random sample
   of the leaderboard is mostly seven-billion-parameter fine-tunes in a narrow band, and an item
   bank calibrated on one cannot tell a hard item from an impossible one. The panel runs from
   0.7 to 51.2 with quartiles at 12.0, 23.9 and 36.1, which is close to uniform.
2. **At most eight submissions per hub organisation.** Forty merges of one base model are close
   to one model repeated, and the effective panel size is what every standard error depends on.
   The panel has 211 organisations in it.

Each candidate is then probed on the smallest file in the set before anything large is
downloaded; six could not be read and were replaced. The chosen panel is written to
`data/raw/ollm/panel.json` with the seed, so the selection is reproducible.
