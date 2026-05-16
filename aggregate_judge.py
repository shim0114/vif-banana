"""Aggregate every `judge_<generator>_<judge>.txt` under
`Main/IF-Banana_Task_v1_text/<task>/` into comparison tables.

Outputs under `judge_results/`:
  - overall_summary.{csv,json}    (generator × judge) mean per metric + average
  - by_main_count.{csv,json}      (generator × judge × n_main) mean per metric
  - per_task.{csv,json}           full per-task scores (one row per task)

Also prints all three tables to stdout for quick inspection.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from judge import (
    DEFAULT_OUTPUT_DIR,
    GENERATOR_MODELS,
    JUDGE_MODELS,
    METRIC_NAMES,
    METRIC_SLUGS,
    TASK_ROOT,
    extract_scores,
)

N_MAIN_RE = re.compile(r"^n(\d+)_")
# Legacy v1 tasks predate the n<N>_<id> naming. Fall back to inspecting the
# task's layout.json so they still aggregate into the right N bucket.
LEGACY_N_BY_LAYOUT = True


def _infer_nmain_from_layout(task_name):
    if not LEGACY_N_BY_LAYOUT:
        return None
    layout = TASK_ROOT / task_name / "layout.json"
    if not layout.is_file():
        return None
    try:
        return len(json.loads(layout.read_text()).get("boxes", []))
    except Exception:
        return None


def infer_nmain(task_name):
    m = N_MAIN_RE.match(task_name)
    if m:
        return int(m.group(1))
    return _infer_nmain_from_layout(task_name)


def collect_entries(base_dir):
    """Return dict {(generator, judge): [{task, nmain, scores: {metric: int}}]}."""
    entries = defaultdict(list)
    for gen in GENERATOR_MODELS:
        for judge in JUDGE_MODELS:
            pattern = f"judge_{gen}_{judge}.txt"
            for fp in sorted(base_dir.rglob(pattern)):
                try:
                    scores = extract_scores(fp.read_text())
                except Exception:
                    scores = None
                if scores is None:
                    continue
                entries[(gen, judge)].append({
                    "task": fp.parent.name,
                    "nmain": infer_nmain(fp.parent.name),
                    "scores": scores,
                })
    return entries


def _mean(vs):
    return sum(vs) / len(vs) if vs else 0.0


def _metric_means(entries):
    return {m: _mean([e["scores"][m] for e in entries]) for m in METRIC_NAMES}


def overall_summary(data):
    rows = []
    for (gen, judge), es in sorted(data.items()):
        if not es:
            continue
        pm = _metric_means(es)
        row = {
            "generator": gen,
            "judge": judge,
            "n_samples": len(es),
            **{METRIC_SLUGS[m]: pm[m] for m in METRIC_NAMES},
            "average": _mean(list(pm.values())),
        }
        rows.append(row)
    return rows


def by_nmain_summary(data):
    rows = []
    for (gen, judge), es in sorted(data.items()):
        bucket = defaultdict(list)
        for e in es:
            if e["nmain"] is not None:
                bucket[e["nmain"]].append(e)
        for nmain in sorted(bucket.keys()):
            sub = bucket[nmain]
            pm = _metric_means(sub)
            rows.append({
                "generator": gen,
                "judge": judge,
                "n_main": nmain,
                "n_samples": len(sub),
                **{METRIC_SLUGS[m]: pm[m] for m in METRIC_NAMES},
                "average": _mean(list(pm.values())),
            })
    return rows


def per_task_summary(data):
    rows = []
    for (gen, judge), es in sorted(data.items()):
        for e in es:
            scores = e["scores"]
            rows.append({
                "generator": gen,
                "judge": judge,
                "task": e["task"],
                "n_main": e["nmain"] if e["nmain"] is not None else "",
                **{METRIC_SLUGS[m]: scores[m] for m in METRIC_NAMES},
                "average": _mean(list(scores.values())),
            })
    return rows


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r.get(c, "")) for c in columns) + "\n")
    print(f"wrote {path}")


def write_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"wrote {path}")


def print_table(title, rows, columns):
    if not rows:
        return
    print(f"\n=== {title} ===")
    widths = {
        c: max(len(c), max(len(_fmt(r.get(c, ""))) for r in rows))
        for c in columns
    }
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    print(header)
    print(sep)
    for r in rows:
        line = []
        for c in columns:
            v = r.get(c, "")
            cell = _fmt(v)
            line.append(cell.rjust(widths[c]) if isinstance(v, float) else cell.ljust(widths[c]))
        print(" | ".join(line))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=str(TASK_ROOT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--quiet", action="store_true", help="do not print tables")
    args = parser.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.output_dir)

    data = collect_entries(base)
    if not data:
        print(f"No judge_*.txt files found under {base}/.")
        return

    metric_cols = [METRIC_SLUGS[m] for m in METRIC_NAMES]

    # Overall
    overall = overall_summary(data)
    overall_cols = ["generator", "judge", "n_samples"] + metric_cols + ["average"]
    write_csv(out_dir / "overall_summary.csv", overall, overall_cols)
    write_json(out_dir / "overall_summary.json", overall)

    # By main count
    by_n = by_nmain_summary(data)
    by_n_cols = ["generator", "judge", "n_main", "n_samples"] + metric_cols + ["average"]
    if by_n:
        write_csv(out_dir / "by_main_count.csv", by_n, by_n_cols)
        write_json(out_dir / "by_main_count.json", by_n)

    # Per task
    per_task = per_task_summary(data)
    per_task_cols = ["generator", "judge", "task", "n_main"] + metric_cols + ["average"]
    if per_task:
        write_csv(out_dir / "per_task.csv", per_task, per_task_cols)
        write_json(out_dir / "per_task.json", per_task)

    if not args.quiet:
        print_table("OVERALL (generator × judge)", overall, overall_cols)
        if by_n:
            print_table("BY MAIN COUNT (generator × judge × n_main)", by_n, by_n_cols)


if __name__ == "__main__":
    main()
