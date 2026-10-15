"""Offline tests for the compatibility changes: no model server, no network."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

from agentcore.llm.LLMBackend import VLLMLLM
from agentcore.llm.LLMConfig import LLMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
import evaluate  # noqa: E402


CONTEXT = ("This model's maximum context length is 131072 tokens. However, you requested 65536 output tokens and your "
           "prompt contains at least 70000 input tokens, for a total of at least 135536 tokens.")
OUTPUT = ("max_completion_tokens=65536cannot be greater than max_model_len=max_total_tokens=32768. Please request "
          "fewer output tokens.")  # vLLM 0.17 when the output budget alone exceeds the window


def overflow(message=CONTEXT):
    request = httpx.Request("POST", "http://server/v1/chat/completions")
    return BadRequestError(message, response=httpx.Response(400, request=request), body=None)


class FakeCompletions:
    def __init__(self, failures):
        self.failures, self.calls = failures, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        message = SimpleNamespace(content="Finish[done]", reasoning_content=None, reasoning=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))


def backend(failures=(), **config):
    llm = VLLMLLM(LLMConfig({"llm_name": "m", "base_url": "http://server/v1", **config}))
    completions = FakeCompletions(list(failures))
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm, completions


def test_default_request_is_the_official_one():
    llm, completions = backend()
    assert llm.run("hi") == "Finish[done]"
    kwargs = completions.calls[0]
    assert kwargs["max_completion_tokens"] == 65536
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    for name in ("temperature", "top_p", "presence_penalty", "seed", "stop"):
        assert name not in kwargs


def test_sampling_options_are_passed_when_set():
    llm, completions = backend(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)
    llm.run("hi")
    kwargs = completions.calls[0]
    assert (kwargs["temperature"], kwargs["top_p"], kwargs["presence_penalty"]) == (1.0, 0.95, 1.5)
    assert kwargs["extra_body"]["top_k"] == 20


@pytest.mark.parametrize("message", [CONTEXT, OUTPUT])
def test_overflow_retries_with_the_output_room_left(monkeypatch, message):
    llm, completions = backend([overflow(message)])
    monkeypatch.setattr(llm, "_room_for_output", lambda messages: 61072)
    assert llm.run("long prompt") == "Finish[done]"
    assert [c["max_completion_tokens"] for c in completions.calls] == [65536, 61072]


def test_overflow_without_room_still_fails(monkeypatch):
    llm, completions = backend([overflow()])
    monkeypatch.setattr(llm, "_room_for_output", lambda messages: -5)
    with pytest.raises(BadRequestError):
        llm.run("prompt longer than the window")
    assert len(completions.calls) == 1


class Server(BaseHTTPRequestHandler):
    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"data": [{"id": "qwen", "max_model_len": 131072}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send({"count": 100000 if body["model"] == "qwen" else 0})

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_room_for_output_asks_the_server(server):
    llm = VLLMLLM(LLMConfig({"llm_name": "qwen", "base_url": server}))
    assert llm._room_for_output([{"role": "user", "content": "x"}]) == 31072


def test_llm_auto_reads_the_served_model(server):
    assert evaluate.served_model(server) == "qwen"


def test_sampling_config_keeps_only_given_flags():
    args = SimpleNamespace(temperature=None, top_p=None, top_k=20, presence_penalty=1.5)
    assert evaluate.sampling_config(args) == {"top_k": 20, "presence_penalty": 1.5}


def test_errored_tasks_are_retried_then_given_up():
    runs = {1: 0, 2: 0, 3: 0}

    def run_one(task_id):
        runs[task_id] += 1
        if task_id == 2 and runs[2] < 2:  # recovers on its second run
            return task_id
        return task_id if task_id == 3 else None  # task 3 always errors

    assert evaluate.run_with_retries([1, 2, 3], run_one, workers=2, max_errors=4) == [3]
    assert runs == {1: 1, 2: 2, 3: 4}

