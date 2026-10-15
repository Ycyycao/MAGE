"""export_results.py on a fabricated run directory (offline)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
import export_results  # noqa: E402


def test_export_takes_the_first_attempt(tmp_path):
    (tmp_path / "results.csv").write_text(
        '1\t[1, 1, 1]\t1\t[]\n'    # retried once: the first attempt got subtask 2 wrong
        '2\t[1, 0]\t0\t[]\n')      # no retry
    (tmp_path / "task_1_subtask_2.log").write_text("Log: Judge: mismatched products: [2]\n")
    (tmp_path / "task_1_subtask_2_retry1.log").write_text("Log: Judge: mismatched products: []\n")
    for name, total in (("task_1_token_usage.json", 100), ("task_1_token_usage_retry1.json", 50),
                        ("task_2_token_usage.json", 70)):
        (tmp_path / name).write_text(json.dumps({"total_prompt_tokens": total - 10, "total_completion_tokens": 10}))
    rows = {r["task_id"]: r for r in export_results.export(tmp_path)}
    assert rows[1]["sub_scores"] == [1, 0, 1] and rows[1]["sr"] == 0 and rows[1]["tokens"] == 100
    assert rows[1]["attempts"] == 2 and rows[1]["sr_final"] == 1 and rows[1]["tokens_final"] == 150
    assert rows[2]["sub_scores"] == [1, 0] and rows[2]["attempts"] == 1 and rows[2]["tokens"] == 70
