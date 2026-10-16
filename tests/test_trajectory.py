"""agentcore/trajectory.py writes AgentMemory's trajectory format from what the backend and the logger already see:
every LLM call (full messages, raw output, reasoning, tokens, who made it) and every action with its full
observation, placed by subtask (from the subtask log's name) and step."""
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from agentcore import trajectory
from agentcore.commons import AgentAct, TaskPackage
from agentcore.llm.LLMBackend import VLLMLLM
from agentcore.llm.LLMConfig import LLMConfig
from agentcore.logging.terminal_logger import AgentLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
import evaluate  # noqa: E402


def read(path):
    """Records of a finished trajectory with every {"ref"} replaced by its text."""
    texts, out = {}, []
    with gzip.open(path, "rt") as stream:
        for line in stream:
            rec = json.loads(line)
            for m in rec.get("messages", ()):
                c = m["content"]
                if isinstance(c, dict):
                    texts.setdefault(c.get("sha"), c.get("text"))
                    m["content"] = texts[c.get("ref") or c["sha"]]
            out.append(rec)
    return out


def backend():
    llm = VLLMLLM(LLMConfig({"llm_name": "m", "base_url": "http://server/v1"}))

    def create(**kwargs):
        message = SimpleNamespace(content="Action: Finish[done]", reasoning_content="because", reasoning=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=3))
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return llm


def llm_layer(llm, prompt):  # named like BaseAgent.llm_layer, the agent's action step
    return llm.run(prompt)


def verify_llm(llm, prompt):
    return llm(prompt)  # through __call__, as some callers do


def test_calls_and_steps_are_placed_by_subtask_and_step(tmp_path):
    path = tmp_path / "task_7_traj.jsonl"
    llm, long_prompt = backend(), "p" * 1500
    with trajectory.recording(path, {"task_id": "travel:7", "method": "mage_official"}):
        for name in ("task_7_subtask_0.log", "task_7_subtask_1_retry1.log"):
            logger = AgentLogger(str(tmp_path / name), FLAG_PRINT=False, OBS_OFFSET=5)
            logger.receive_task(TaskPackage(instruction="plan"), agent_name="a")
            llm_layer(llm, long_prompt)
            logger.take_action(AgentAct(name="Search", params={"city": "Rome"}), agent_name="a", step_idx=0)
            logger.get_obs("0123456789")
            verify_llm(llm, "check")
            logger.get_obs("restarted")  # an observation without an action (the verify-retry backtrace)
    assert not path.exists()
    recs = read(str(path) + ".gz")

    assert recs[0]["kind"] == "task" and recs[0]["task_id"] == "travel:7"
    body = [(r["kind"], r.get("role") or r.get("tool"), r["subtask"], r["step"], r.get("attempt")) for r in recs[1:]]
    assert body == [("call", "agent", 0, 1, None), ("step", "Search", 0, 1, None), ("call", "verify_llm", 0, 2, None),
                    ("step", "", 0, 2, None),
                    ("call", "agent", 1, 1, 1), ("step", "Search", 1, 1, 1), ("call", "verify_llm", 1, 2, 1),
                    ("step", "", 1, 2, 1)]
    call, step = recs[1], recs[2]
    assert call["messages"] == [{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": long_prompt}]
    assert (call["output"], call["reasoning"], call["prompt_tokens"]) == ("Action: Finish[done]", "because", 1200)
    assert (step["arg"], step["obs"]) == ("{'city': 'Rome'}", "0123456789")    # not cut at OBS_OFFSET
    assert recs[5]["messages"][1]["content"] == long_prompt                     # second copy written as a ref


def test_nothing_is_recorded_outside_a_task(tmp_path):
    llm = backend()
    assert trajectory.current() is None
    assert llm_layer(llm, "hi") == "Action: Finish[done]"
    logger = AgentLogger(str(tmp_path / "task_1_subtask_0.log"), FLAG_PRINT=False)
    logger.receive_task(TaskPackage(instruction="plan"), agent_name="a")
    logger.get_obs("ok")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["task_1_subtask_0.log"]


def test_a_task_that_raises_still_leaves_its_trajectory(tmp_path):
    path = tmp_path / "task_3_traj.jsonl"
    args = SimpleNamespace(no_traj=False, dataset="travelplan", llm="qwen", max_steps=40)

    def boom(*_):
        raise RuntimeError("env down")
    original, evaluate._evaluate_task = evaluate._evaluate_task, boom
    try:
        try:
            evaluate.evaluate_task({"id": 3}, args, tmp_path, 1, {})
        except RuntimeError:
            pass
    finally:
        evaluate._evaluate_task = original
    assert read(str(path) + ".gz")[0]["task_id"] == "travel:3"
    assert trajectory.current() is None
