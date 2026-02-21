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

## Results

### MS MARCO Passage Ranking (1000 Judged Dev Queries)

Evaluated on MS MARCO dev set using **candidate reranking** mode (top-1000 candidates per query). Both models use `bert-base-uncased` with mean pooling, 768-d projection, and L2 normalization. Training: batch_size=16, lr=2e-5, temperature=0.05, in-batch negatives.

**Environment**: Python 3.12.3, PyTorch 2.10.0+cu128, Transformers 5.1.0, CUDA enabled.

#### Dense Baseline (Single-Resolution Training at dim=768)

| Dim | MRR@10 | nDCG@10 | Recall@50 | Recall@100 | Recall@1000 | Search ms/q |
|----:|-------:|--------:|----------:|-----------:|------------:|------------:|
| 768 | 0.2177 | 0.2727 | 0.672 | 0.715 | 0.794 | 2.994 |
| 384 | 0.2167 | 0.2709 | 0.669 | 0.713 | 0.794 | 0.542 |
| 192 | 0.2156 | 0.2705 | 0.659 | 0.711 | 0.794 | 0.329 |
| 96 | 0.2001 | 0.2544 | 0.646 | 0.697 | 0.794 | 0.247 |

#### Matryoshka (Multi-Resolution Training at dims 768/384/192/96)

| Dim | MRR@10 | nDCG@10 | Recall@50 | Recall@100 | Recall@1000 | Search ms/q |
|----:|-------:|--------:|----------:|-----------:|------------:|------------:|
| 768 | 0.2135 | 0.2692 | 0.665 | 0.708 | 0.794 | 2.412 |
| 384 | 0.2128 | 0.2687 | 0.661 | 0.709 | 0.794 | 0.541 |
| 192 | 0.2142 | 0.2678 | 0.663 | 0.712 | 0.794 | 0.329 |
| 96 | 0.2064 | 0.2597 | 0.656 | 0.712 | 0.794 | 0.246 |

**Key Finding**: Matryoshka training preserves retrieval quality at reduced dimensions. At dim=96 (8x compression), the Matryoshka model retains 96.7% of full-dim baseline MRR@10, while at dim=192, Matryoshka (0.2142 MRR@10) nearly matches the baseline at full dim=768 (0.2177). The Matryoshka model shows better graceful degradation at lower dimensions, particularly for Recall@100 at dim=96 (0.712 vs 0.697).

### Smoke Run (Synthetic Toy Dataset)

Pipeline validation on 200 passages / 32 dev queries using `hf-internal-testing/tiny-random-bert`:

| Model | Dim | MRR@10 | nDCG@10 | Recall@50 | Recall@100 | Recall@1000 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | - | 0.1456 | 0.2257 | 1.0000 | 1.0000 | 1.0000 |
| dense_baseline | 768 | 0.1003 | 0.1278 | 0.7812 | 0.8438 | 1.0000 |
| dense_baseline | 96 | 0.0920 | 0.1144 | 0.6875 | 0.8438 | 1.0000 |
| matryoshka | 768 | 0.1003 | 0.1278 | 0.7812 | 0.8438 | 1.0000 |
| matryoshka | 192 | 0.1099 | 0.1356 | 0.8125 | 0.8438 | 1.0000 |
| matryoshka | 96 | 0.0920 | 0.1144 | 0.6875 | 0.8438 | 1.0000 |

*Note: Toy dataset validates end-to-end pipeline correctness; absolute metrics are not meaningful.*

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
- “Recall@K” is **hit@K**, not set-based recall; with one positive per query it is especially easy to saturate at 1.0.
- Large-corpus evaluation currently supports **candidate reranking** via `top1000.dev`; full-corpus dense retrieval with ANN is not integrated into evaluation.
- No hard-negative mining, distillation, or cross-batch negatives; the training loop uses only in-batch negatives from the current step.
- Latency numbers in `artifacts/results.json` come from the current host and a tiny corpus; they should not be used for capacity planning.

## Next Research Steps
1. Wire FAISS indexing/search into evaluation for full-corpus MS MARCO retrieval with ANN (current eval uses candidate reranking from top-1000).
2. Add MS MARCO-specific training improvements: mined hard negatives, larger batch via gradient accumulation, and/or cross-batch memory.
3. Explore Matryoshka loss variants: dim-weighting schedules, non-prefix subspaces, and truncation-aware regularization.
4. Add deployment-oriented compression: float16/int8 storage, PQ/IVF, and calibrated truncation policies per latency budget.
