"""Full trajectory of one task, in AgentMemory's format (hybrid_mem/traj.py there), so its tools read MAGE runs too.

One JSON object per line in <run dir>/task_<id>_traj.jsonl while the task runs, gzipped to .jsonl.gz when it ends:
- {"kind": "task", "schema", ...header}: first line.
- {"kind": "call", "role", "messages", "output", "reasoning", "prompt_tokens", "completion_tokens"}: every LLM call
  made from the task's thread. role is "agent" for the action step (BaseAgent.llm_layer), else the name of the
  function that called the LLM (e.g. verify_llm), so each auxiliary call says what it was for.
- {"kind": "step", "tool", "arg", "obs"}: every action the agent logged, with its full observation. An observation
  logged without an action (the backtrace that restarts a subtask after a failed verification) has tool "".
- {"kind": "result", "sub_scores", "task_success", "answer_pairs", ...}: after the task is judged (each judge round
  on a gold-triggered retry), with the dataset's details (travelplan: `base_person`, the fixed first
  traveler every subtask is given; `questions`; `checks`).
The formats are specified in AgentMemory's docs/conventions/trajectory-format.md (schema SCHEMA).
Records carry `subtask` (0-based) and `step` (1-based, counted within the subtask; an action's LLM call has the same
step as the action); `attempt` is added on the gold-triggered retries (--max_retry_num), whose subtasks run again.
A message text of TEXT_REF_MIN characters or more is written once as {"sha", "text"} and as {"ref": sha} after that.

Recording is passive: it reads what the backend and the logger already see, and changes no prompt or decision.
"""
import gzip
import hashlib
import json
import os
import re
import shutil
import threading
import time
from contextlib import contextmanager

SCHEMA = 1
TEXT_REF_MIN = 1000
_local = threading.local()
_SUBTASK_LOG = re.compile(r"task_\d+_subtask_(\d+)(?:_retry(\d+))?\.log$")


def current():
    """The trajectory of the task running in this thread, or None."""
    return getattr(_local, "traj", None)


@contextmanager
def recording(path, header):
    """Record this thread's task into path (gzipped to path + '.gz' at the end, also when the task raises)."""
    traj = Trajectory(path, header)
    _local.traj = traj
    try:
        yield traj
    finally:
        _local.traj = None
        traj.close()


class Trajectory:
    def __init__(self, path, header):
        self.path, self.lock, self.seen = str(path), threading.Lock(), set()
        self.subtask, self.step, self.attempt, self.pending = None, None, 0, None
        with open(self.path, "w") as stream:
            stream.write("")
        self._append({"kind": "task", "schema": SCHEMA, **header})

    def start_subtask(self, log_file_name):
        """A subtask agent starts; its index (and retry) come from its log file name, task_<id>_subtask_<k>[_retry<r>].log."""
        match = _SUBTASK_LOG.search(str(log_file_name))
        if match:
            self.subtask, self.step, self.pending = int(match[1]), 1, None
            self.attempt = int(match[2] or 0)

    def action(self, name, params):
        self.pending = (name, params)

    def observation(self, obs):
        """Write the pending action with its observation as one step; the next action is the next step."""
        name, params = self.pending or ("", "")
        self.write({"kind": "step", "tool": name, "arg": str(params), "obs": str(obs)})
        self.pending = None
        if self.step is not None:
            self.step += 1

    def result(self, **fields):
        self.write({"kind": "result", **fields})

    def write(self, rec):
        with self.lock:
            self._append(rec)

    def _append(self, rec):
        rec = {"subtask": self.subtask, "step": self.step, **rec, "t": round(time.time(), 3)}
        if self.attempt:
            rec["attempt"] = self.attempt
        with open(self.path, "a") as stream:
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _text(self, text):
        if not isinstance(text, str) or len(text) < TEXT_REF_MIN:
            return text
        digest = hashlib.sha1(text.encode()).hexdigest()[:16]
        if digest in self.seen:
            return {"ref": digest}
        self.seen.add(digest)
        return {"sha": digest, "text": text}

    def call(self, role, messages, output, reasoning="", prompt_tokens=None, completion_tokens=None):
        with self.lock:  # a text's first {"sha", "text"} is written before any {"ref"} to it
            msgs = [{"role": m["role"], "content": self._text(m["content"])} for m in messages]
            self._append({"kind": "call", "role": role, "messages": msgs, "output": output, "reasoning": reasoning,
                          "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})

    def close(self):
        with open(self.path, "rb") as src, gzip.open(self.path + ".gz.tmp", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(self.path + ".gz.tmp", self.path + ".gz")
        os.remove(self.path)
