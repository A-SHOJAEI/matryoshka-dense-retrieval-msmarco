from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def make_report(results_path: str | Path, out_path: str | Path) -> None:
    results_path = Path(results_path)
    out_path = Path(out_path)
    obj = json.loads(results_path.read_text(encoding="utf-8"))

    dims = [str(d) for d in obj.get("dims", [])]
    exps = obj.get("experiments", {})

    lines: list[str] = []
    lines.append("# Retrieval Report")
    lines.append("")
    lines.append(f"- run_name: `{obj.get('run_name')}`")
    lines.append(f"- dataset: `{obj.get('dataset')}`")
    env = obj.get("environment", {})
    lines.append(f"- python: `{env.get('python')}`")
    lines.append(f"- torch: `{env.get('torch')}` (cuda_available={env.get('cuda_available')})")
    lines.append(f"- transformers: `{env.get('transformers')}`")
    lines.append("")

    lines.append("## Baseline")
    lines.append("")
    if "bm25" in exps:
        m = exps["bm25"]["metrics"]
        lines.append("| Method | MRR@10 | nDCG@10 | Recall@50 | Recall@100 | Recall@1000 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append(
            f"| BM25 | {_fmt(m['mrr_10'])} | {_fmt(m['ndcg_10'])} | {_fmt(m['recall_50'])} | {_fmt(m['recall_100'])} | {_fmt(m['recall_1000'])} |"
        )
    else:
        lines.append("BM25 baseline disabled in config.")
    lines.append("")

    lines.append("## Dense Retrieval (Matryoshka vs Ablation)")
    lines.append("")
    lines.append(
        "Ablation (per plan): **remove Matryoshka loss** (train only single-resolution full-dimension embeddings) and compare truncation at inference."
    )
    lines.append("")
    for model_key in ["dense_baseline", "matryoshka"]:
        if model_key in exps and exps[model_key].get("evaluation_mode") == "candidate_rerank":
            lines.append(
                f"- `{model_key}` evaluated in `candidate_rerank` mode (reranking BM25 top1000 candidates; not full-corpus dense search)."
            )
    lines.append("")

    lines.append("| Model | Dim | MRR@10 | nDCG@10 | Recall@50 | Recall@100 | Recall@1000 | Search ms/q | Passage bytes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model_key in ["dense_baseline", "matryoshka"]:
        if model_key not in exps:
            continue
        per_dim = exps[model_key].get("per_dim", {})
        for d in dims:
            if d not in per_dim:
                continue
            m = per_dim[d]["metrics"]
            lat = per_dim[d].get("latency", {})
            mem = per_dim[d].get("memory", {})
            lines.append(
                f"| {model_key} | {d} | {_fmt(m['mrr_10'])} | {_fmt(m['ndcg_10'])} | {_fmt(m['recall_50'])} | {_fmt(m['recall_100'])} | {_fmt(m['recall_1000'])} | "
                f"{lat.get('search_avg_ms_per_query', 0.0):.2f} | {int(mem.get('passage_matrix_bytes', 0))} |"
            )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    make_report(args.results, args.out)


if __name__ == "__main__":
    main()
