# Matryoshka Dense Retrieval (MS MARCO Pipeline + Reproducible Smoke Run)

## Problem Statement
Dense dual-encoders give fast retrieval via vector similarity, but production systems often need multiple embedding sizes (quality vs latency/memory). This repo implements **Matryoshka representation learning**: train one bi-encoder such that **prefix-truncated embeddings** (e.g., 768/384/192/96 dims) remain usable at inference time, and compare against a single-resolution ablation.

The canonical, machine-readable outputs are `artifacts/results.json` and `artifacts/report.md` produced by `src/eval/run_eval.py` and `src/report/make_report.py`.

## Dataset Provenance
### 1) Default Repro Run: Synthetic “toy” IR dataset
The checked-in results (`artifacts/results.json`, `artifacts/report.md`) are from the default config `configs/smoke.yaml`:
- Passages: `200`
- Train queries: `64`
- Dev queries: `32`
- Positives per query: `1`

Data are generated locally by `src/data/make_toy.py` into `data/processed/toy/` as:
- `passages.jsonl`: `{"docid": int, "text": str}`
- `queries_{train,dev}.jsonl`: `{"qid": str, "text": str}`
- `qrels_{train,dev}.tsv`: `qid <tab> docid <tab> rel`

### 2) MS MARCO Passage Ranking (download + prep implemented)
For full-scale experiments, `src/data/download_msmarco.py` downloads (and extracts) the official MS MARCO ranking artifacts:
- `collectionandqueries.tar.gz` (yields `collection.tsv`, `queries.train.tsv`, `queries.dev.tsv`)
- `qrels.train.tsv`, `qrels.dev.tsv`
- `top1000.dev.tar.gz` (yields `top1000.dev` candidates)

`src/data/prepare_msmarco.py` converts raw TSVs into the same processed layout as the toy dataset, cleaning ASCII control chars and normalizing whitespace.
MS MARCO is distributed under its upstream dataset terms; review those terms before downloading/using the corpus.

## Methodology (As Implemented)
### Model
`src/model/encoder.py`:
- Shared-weight bi-encoder using `transformers.AutoModel`
- Mean pooling over token embeddings
- Linear projection to `projection_dim=768`
- Optional L2 normalization (enabled in both `configs/smoke.yaml` and `configs/matryoshka.yaml`)

### Training Objective (Baseline vs Matryoshka)
`src/model/matryoshka.py`, `src/train/biencoder.py`:
- In-batch InfoNCE over a batch of (query, positive passage) pairs:
  - logits = `(q @ p^T) / temperature`
  - targets = diagonal (each query matches its paired passage)
- **Ablation (`dense_baseline`)**: train only at full dim (single resolution).
- **Matryoshka (`matryoshka`)**: compute InfoNCE on multiple **prefix dims** (configured as `[768, 384, 192, 96]`), renormalize per prefix, average losses across dims.

### Evaluation
`src/eval/run_eval.py`:
- For small corpora (toy/smoke): exact dense retrieval by full dot-product search over all passages (NumPy), per dim via prefix truncation + renorm.
- For very large corpora (heuristic: `passages.jsonl` > 2GB): falls back to **candidate reranking** using `top1000.dev` (not full-corpus ANN).
- Metrics (`src/eval/metrics.py`): `MRR@10`, `nDCG@10`, and “Recall@K” implemented as **binary hit@K** (1.0 if any relevant appears in top K for the query).

## Baselines / Ablations
- **BM25 baseline** (toy/small corpora): `rank-bm25` in `src/eval/bm25_eval.py`
- **Dense ablation**: `dense_baseline` (remove Matryoshka loss; train at 768 only; still evaluated at truncated dims)
- **Dense Matryoshka**: `matryoshka` (multi-dim loss)

## Exact Results (This Repo Snapshot)
The numbers below are taken verbatim from `artifacts/report.md` (generated from `artifacts/results.json`).

Run metadata (`artifacts/results.json`):
- run_name: `smoke`
- dataset: `toy`
- timestamp (UTC): `2026-02-10 11:04:04`
- python: `3.12.3`
- torch: `2.2.2+cu121` (cuda_available=`True`)
- transformers: `4.40.2`

Run config highlights (`configs/smoke.yaml`, as used to produce the artifacts):
- model: `hf-internal-testing/tiny-random-bert` + mean pooling + 768-d projection + L2 norm
- train: batch_size=`16`, max_steps=`25`, lr=`2e-5`, weight_decay=`0.01`, temperature=`0.05`
- eval dims: `[768, 384, 192, 96]` (BM25 enabled; FAISS disabled)

### Baseline (Table: `artifacts/report.md` -> “Baseline”)
| Method | MRR@10 | nDCG@10 | Recall@50 | Recall@100 | Recall@1000 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.1225 | 0.1880 | 1.0000 | 1.0000 | 1.0000 |

### Dense Retrieval: Matryoshka vs Ablation (Table: `artifacts/report.md` -> “Dense Retrieval (Matryoshka vs Ablation)”)
| Model | Dim | MRR@10 | nDCG@10 | Recall@50 | Recall@100 | Recall@1000 | Search ms/q | Passage bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_baseline | 768 | 0.0985 | 0.1261 | 0.7812 | 0.8438 | 1.0000 | 0.05 | 614400 |
| dense_baseline | 384 | 0.1001 | 0.1276 | 0.7812 | 0.8438 | 1.0000 | 0.03 | 307200 |
| dense_baseline | 192 | 0.1000 | 0.1211 | 0.8125 | 0.8438 | 1.0000 | 0.03 | 153600 |
| dense_baseline | 96 | 0.0920 | 0.1144 | 0.6875 | 0.8438 | 1.0000 | 0.03 | 76800 |
| matryoshka | 768 | 0.0990 | 0.1265 | 0.7812 | 0.8438 | 1.0000 | 0.07 | 614400 |
| matryoshka | 384 | 0.1001 | 0.1276 | 0.7812 | 0.8438 | 1.0000 | 0.09 | 307200 |
| matryoshka | 192 | 0.1052 | 0.1252 | 0.8125 | 0.8438 | 1.0000 | 0.08 | 153600 |
| matryoshka | 96 | 0.0920 | 0.1144 | 0.6875 | 0.8438 | 1.0000 | 0.08 | 76800 |

Interpretation: this is a tiny synthetic dataset, so absolute quality/latency numbers are not meaningful; it mainly validates that the **end-to-end pipeline runs** and that truncation mechanics + reporting are correct.

## Reproduction
Prereqs:
- `python3` available on PATH (the Makefile bootstraps a local `.venv/`)
- Internet access to fetch `get-pip.py` and Hugging Face model weights/tokenizers (and MS MARCO files if you run the full config)

### Smoke (matches current `artifacts/*`)
```bash
make all
```
This runs: `make setup data train eval report` using `CONFIG=configs/smoke.yaml`.

### MS MARCO (real downloads; long-running)
```bash
make setup
CONFIG=configs/matryoshka.yaml make data
CONFIG=configs/matryoshka.yaml make train
CONFIG=configs/matryoshka.yaml make eval report
```

Recommended for MS MARCO-scale passage access:
```bash
.venv/bin/python -m src.data.build_passage_db \
  --passages_jsonl data/processed/msmarco/passages.jsonl \
  --out_db data/processed/msmarco/passages.sqlite
```

Optional: pre-encode passage embeddings and build FAISS flat inner-product indexes (not wired into `src/eval/run_eval.py` yet):
```bash
.venv/bin/python -m src.index.encode_passages \
  --ckpt checkpoints/msmarco/matryoshka \
  --passages data/processed/msmarco/passages.jsonl \
  --out data/embeddings/msmarco/passages

.venv/bin/python -m src.index.build_faiss \
  --emb_dir data/embeddings/msmarco/passages \
  --dims 768 384 192 96 \
  --out data/index/faiss
```

## Limitations (Current Implementation)
- The checked-in metrics are on a **synthetic toy dataset**, not MS MARCO.
- “Recall@K” is **hit@K**, not set-based recall; with one positive per query it is especially easy to saturate at 1.0.
- Large-corpus evaluation currently supports **candidate reranking** via `top1000.dev`; full-corpus dense retrieval with ANN is not integrated into evaluation.
- No hard-negative mining, distillation, or cross-batch negatives; the training loop uses only in-batch negatives from the current step.
- Latency numbers in `artifacts/results.json` come from the current host and a tiny corpus; they should not be used for capacity planning.

## Next Research Steps
1. Wire FAISS indexing/search into evaluation for true full-corpus MS MARCO retrieval (and report ANN recall/latency).
2. Add MS MARCO-specific training improvements: mined hard negatives, larger batch via gradient accumulation, and/or cross-batch memory.
3. Explore Matryoshka loss variants: dim-weighting schedules, non-prefix subspaces, and truncation-aware regularization.
4. Add deployment-oriented compression: float16/int8 storage, PQ/IVF, and calibrated truncation policies per latency budget.
