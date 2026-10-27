"""Run the PromptVault retention benchmark.

Usage (from the repository root, venv activated):

    python benchmarks/run_eval.py
    python benchmarks/run_eval.py --methods full_context,last_n,head_tail   # no models needed
    python benchmarks/run_eval.py --budgets 0.1,0.25,0.5 --by-category
    python benchmarks/run_eval.py --limit 5                                 # quick smoke run

What it measures (no LLM, no network except the first-time model downloads):
  For every case, method and budget, how many tokens survive and whether
  the gold answer text still reaches the model. Budgets are FRACTIONS of
  each case's original token count, so the sweep traces a
  rate-distortion curve (compression vs. retained evidence).

Outputs (in benchmarks/report/):
  results.csv   one row per (case, method, budget)
  summary.csv   aggregates per (method, budget, category)
  rate_distortion.png   (only if matplotlib is installed)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from baselines import BASELINES
from metrics import (
    answer_present,
    compression_ratio,
    count_messages_tokens,
    evidence_recall,
    token_reduction_pct,
)

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "datasets" / "controlled_qa.jsonl"
DEFAULT_OUT = HERE / "report"
DEFAULT_BUDGETS = [0.1, 0.2, 0.3, 0.5, 0.75]
ALL_METHODS = ["full_context", "last_n", "head_tail", "promptvault"]

RESULT_FIELDS = [
    "case_id", "category", "method", "budget_frac", "budget_tokens", "effective_budget",
    "original_tokens", "compressed_tokens", "token_reduction_pct", "compression_ratio",
    "evidence_recall", "answer_present", "within_budget", "latency_ms", "error",
]


def words_counter(text: str) -> int:
    """Offline stand-in for smoke tests ONLY. Never report numbers from it."""
    return len(text.split())


def load_dataset(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON ({exc}). Run validate_dataset.py first.")
    return cases


def normalize_category(raw: object) -> str:
    text = str(raw).strip().lower()
    return text[3:].strip() if len(text) > 2 and text[1] == ")" else text


def build_methods(names: list[str], safety_margin: float) -> dict[str, Callable]:
    """name -> fn(history, query, budget, effective_budget, count) -> messages."""
    methods: dict[str, Callable] = {}
    for name in names:
        if name in BASELINES:
            fn = BASELINES[name]
            methods[name] = lambda h, q, b, eff, count, fn=fn: fn(h, q, eff, count)
        elif name == "promptvault":
            from promptvault import PromptVault  # heavy: loads models lazily on first use

            vault = PromptVault(budget_safety_margin=safety_margin)

            def run_pv(h, q, b, eff, count, vault=vault):
                vault.token_budget = b  # vault applies the safety margin itself
                return vault.compress(h, q).messages

            methods[name] = run_pv
            # Warm up so one-time model loading is NOT counted in latency.
            warm_history = [
                {"role": "user", "content": "My order number is ZX-1. Please remember it."},
                {"role": "assistant", "content": "Noted. Anything else about the order today?"},
                {"role": "user", "content": "No, thanks. I also like hiking on weekends."},
            ]
            vault.token_budget = 40
            vault.compress(warm_history, "What is my order number?")
        else:
            raise SystemExit(f"Unknown method {name!r}. Choose from {ALL_METHODS}.")
    return methods


def run(args: argparse.Namespace) -> list[dict]:
    if args.counter == "words":
        count = words_counter
        print("WARNING: using the whitespace word counter; numbers are for smoke tests only.\n")
    else:
        from promptvault import count_tokens as count

    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = build_methods(method_names, args.safety_margin)

    rows: list[dict] = []
    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        history, query, gold = case["history"], case["query"], case["gold_answer"]
        category = normalize_category(case["category"])
        original = count_messages_tokens(history, count)

        for name, fn in methods.items():
            fracs = [1.0] if name == "full_context" else args.budgets
            for frac in fracs:
                budget = max(1, int(frac * original))
                effective = int(budget * args.safety_margin)
                error = ""
                messages: list[dict] = []
                start = time.perf_counter()
                try:
                    messages = fn(history, query, budget, effective, count)
                except Exception as exc:  # keep going; record the failure
                    error = f"{type(exc).__name__}: {exc}"
                latency_ms = (time.perf_counter() - start) * 1000.0

                compressed = count_messages_tokens(messages, count)
                ratio = compression_ratio(original, compressed)
                rows.append({
                    "case_id": case["case_id"],
                    "category": category,
                    "method": name,
                    "budget_frac": frac,
                    "budget_tokens": budget,
                    "effective_budget": effective,
                    "original_tokens": original,
                    "compressed_tokens": compressed,
                    "token_reduction_pct": round(token_reduction_pct(original, compressed), 3),
                    "compression_ratio": "inf" if math.isinf(ratio) else round(ratio, 4),
                    "evidence_recall": round(evidence_recall(messages, gold), 4) if not error else 0.0,
                    "answer_present": answer_present(messages, gold) if not error else 0.0,
                    "within_budget": "" if name == "full_context" else int(compressed <= effective),
                    "latency_ms": round(latency_ms, 2),
                    "error": error,
                })
        if idx % 10 == 0 or idx == total:
            print(f"  processed {idx}/{total} cases", flush=True)
    return rows


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["budget_frac"], r["category"])].append(r)
        groups[(r["method"], r["budget_frac"], "ALL")].append(r)

    summary = []
    for (method, frac, category), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        ratios = [r["compression_ratio"] for r in items if r["compression_ratio"] != "inf"]
        checked = [r["within_budget"] for r in items if r["within_budget"] != ""]
        summary.append({
            "method": method,
            "budget_frac": frac,
            "category": category,
            "n_cases": len(items),
            "mean_token_reduction_pct": round(_mean([r["token_reduction_pct"] for r in items]), 2),
            "mean_compression_ratio": round(_mean(ratios), 3) if ratios else "",
            "mean_evidence_recall": round(_mean([r["evidence_recall"] for r in items]), 4),
            "answer_present_rate": round(_mean([r["answer_present"] for r in items]), 4),
            "within_budget_rate": round(_mean(checked), 3) if checked else "",
            "mean_latency_ms": round(_mean([r["latency_ms"] for r in items]), 2),
            "n_errors": sum(1 for r in items if r["error"]),
        })
    return summary


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_overall(summary: list[dict]) -> None:
    print("\nOVERALL (all categories)")
    header = f"{'method':<14}{'budget':>7}{'cases':>6}{'reduction%':>11}{'recall':>8}{'answer':>8}{'in-budget':>10}{'ms':>9}{'err':>5}"
    print(header)
    print("-" * len(header))
    for s in summary:
        if s["category"] != "ALL":
            continue
        wb = s["within_budget_rate"]
        print(
            f"{s['method']:<14}{s['budget_frac']:>7}{s['n_cases']:>6}"
            f"{s['mean_token_reduction_pct']:>11}{s['mean_evidence_recall']:>8}"
            f"{s['answer_present_rate']:>8}{(wb if wb != '' else '-'):>10}"
            f"{s['mean_latency_ms']:>9}{s['n_errors']:>5}"
        )


def print_by_category(summary: list[dict]) -> None:
    cats = sorted({s["category"] for s in summary if s["category"] != "ALL"})
    fracs = sorted({s["budget_frac"] for s in summary})
    for frac in fracs:
        print(f"\nanswer_present_rate by category at budget {frac}")
        print(f"{'method':<14}" + "".join(f"{c[:11]:>13}" for c in cats))
        for method in ALL_METHODS:
            cells = {s["category"]: s for s in summary if s["method"] == method and s["budget_frac"] == frac}
            if not cells:
                continue
            print(f"{method:<14}" + "".join(f"{cells[c]['answer_present_rate'] if c in cells else '-':>13}" for c in cats))


def plot_rate_distortion(summary: list[dict], out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    styles = {
        "full_context": ("s", "-"),
        "last_n": ("o", "--"),
        "head_tail": ("^", ":"),
        "promptvault": ("D", "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, key, title in (
        (axes[0], "mean_evidence_recall", "Evidence recall"),
        (axes[1], "answer_present_rate", "Answer fully present"),
    ):
        for method in ALL_METHODS:
            pts = sorted(
                (s["mean_token_reduction_pct"], s[key])
                for s in summary
                if s["method"] == method and s["category"] == "ALL"
            )
            if not pts:
                continue
            marker, line = styles[method]
            ax.plot(*zip(*pts), marker=marker, linestyle=line, label=method)
        ax.set_xlabel("Token reduction (%)  →  more compression")
        ax.set_ylabel(title)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(title + " vs. compression")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle("Rate-distortion: retained evidence as the budget tightens")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--methods", default=",".join(ALL_METHODS), help=f"comma list from {ALL_METHODS}")
    parser.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_BUDGETS),
                        help="comma list of budget fractions of each case's original tokens")
    parser.add_argument("--safety-margin", type=float, default=0.9,
                        help="effective budget = budget * margin, for ALL methods (matches PromptVault)")
    parser.add_argument("--counter", choices=["tiktoken", "words"], default="tiktoken")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N cases")
    parser.add_argument("--by-category", action="store_true", help="also print per-category tables")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    try:
        args.budgets = [float(b) for b in args.budgets.split(",") if b.strip()]
    except ValueError:
        raise SystemExit("--budgets must be comma-separated numbers, e.g. 0.1,0.3,0.5")
    if not all(0 < b <= 1 for b in args.budgets):
        raise SystemExit("each budget fraction must be in (0, 1]")
    if not 0 < args.safety_margin <= 1:
        raise SystemExit("--safety-margin must be in (0, 1]")
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {args.dataset}")
    rows = run(args)
    summary = summarize(rows)

    write_csv(args.out_dir / "results.csv", RESULT_FIELDS, rows)
    write_csv(args.out_dir / "summary.csv", list(summary[0].keys()), summary)
    print_overall(summary)
    if args.by_category:
        print_by_category(summary)

    print(f"\nWrote {args.out_dir / 'results.csv'}")
    print(f"Wrote {args.out_dir / 'summary.csv'}")
    if not args.no_plot:
        if plot_rate_distortion(summary, args.out_dir / "rate_distortion.png"):
            print(f"Wrote {args.out_dir / 'rate_distortion.png'}")
        else:
            print("matplotlib not installed; skipped the plot (pip install matplotlib).")

    errors = sum(1 for r in rows if r["error"])
    if errors:
        print(f"\n{errors} run(s) raised errors; see the 'error' column in results.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())