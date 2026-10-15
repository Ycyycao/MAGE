#!/usr/bin/env python3
"""Write one JSON line per task of a run: the first attempt's scores (@1) next to the final ones.

The evaluator's retries are decided against the gold answers (each dataset's _run_judge compares the answers with
them before choosing what to replay), so only @1 can be set beside methods that get no such retry. parse_results.py
prints the averages at every attempt; this gives the per-task rows, e.g. for paired comparisons.

usage: python export_results.py <run dir with results.csv> [out.jsonl]   (default: <run dir>/tasks.jsonl)
Row: task_id, sub_scores / ps / sr / tokens at @1, attempts, and sub_scores_final / sr_final / tokens_final.
"""

import argparse
import json
from pathlib import Path

from parse_results import (RETRY_RE, get_attempt_failures, load_token_usage_at_k, parse_results_csv)


def export(log_dir):
    directory = Path(log_dir)
    tasks = parse_results_csv(directory / "results.csv")
    attempts = dict.fromkeys(tasks, 1)
    for path in directory.iterdir():
        match = RETRY_RE.fullmatch(path.name)
        if match and int(match[1]) in tasks:
            attempts[int(match[1])] = max(attempts[int(match[1])], int(match[2]) + 1)
    rows = []
    for task_id, result in sorted(tasks.items()):
        final = result["sub_successes"]
        if attempts[task_id] == 1:
            first, first_success = final, result["task_success"]
        else:  # the first attempt failed the gold check, which is what triggered the retry
            failed = set(get_attempt_failures(directory, task_id, len(final), 1))
            first, first_success = [int(i not in failed) for i in range(1, len(final) + 1)], 0
        rows.append({
            "task_id": task_id,
            "sub_scores": first, "ps": sum(first) / len(first) if first else 0.0, "sr": first_success,
            "tokens": load_token_usage_at_k(directory, task_id, 1),
            "attempts": attempts[task_id],
            "sub_scores_final": final, "sr_final": result["task_success"],
            "tokens_final": load_token_usage_at_k(directory, task_id, attempts[task_id]),
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dir")
    parser.add_argument("out", nargs="?")
    args = parser.parse_args()
    rows = export(args.log_dir)
    out = Path(args.out or Path(args.log_dir) / "tasks.jsonl")
    with out.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    n, subtasks = len(rows) or 1, [s for r in rows for s in r["sub_scores"]]
    print(f"{len(rows)} tasks -> {out}\n"  # PS pooled over subtasks, as parse_results.py prints it
          f"@1: PS {sum(subtasks) / (len(subtasks) or 1):.4f}  SR {sum(r['sr'] for r in rows) / n:.4f}  "
          f"(final SR {sum(r['sr_final'] for r in rows) / n:.4f})")
