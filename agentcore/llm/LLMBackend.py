import json
import re
import time
import urllib.request

from openai import BadRequestError, OpenAI

from agentcore.llm.LLMConfig import LLMConfig


class BaseLLM:
    def __init__(self, llm_config: LLMConfig) -> None:
        self.llm_name = llm_config.llm_name
        self.base_url = llm_config.base_url
        self.temperature: float = llm_config.temperature
        self.top_p: float = llm_config.top_p
        self.top_k: int = llm_config.top_k
        self.presence_penalty: float = llm_config.presence_penalty
        self.seed: int = llm_config.seed
        self.enable_thinking: bool = llm_config.enable_thinking
        self.max_output_tokens: int = llm_config.max_output_tokens
        # LLM call trace, including prompt and completion token counts and other metadata
        self._call_trace: list = []

    def __call__(self, prompt: str, system_prompt: str = None) -> str:
        return self.run(prompt, system_prompt)
    
    def run(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError

    def _build_sampling_kwargs(self) -> dict:
        """Build common sampling parameters dict for API calls."""
        kwargs = {"max_completion_tokens": self.max_output_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if self.top_k is not None:
            kwargs["extra_body"]["top_k"] = self.top_k
        return kwargs

    def reset_trace(self):
        self._call_trace = []

    def get_trace_summary(self) -> dict:
        total_prompt = sum(e["prompt_tokens"] for e in self._call_trace)
        total_completion = sum(e["completion_tokens"] for e in self._call_trace)
        return {
            "llm_name": self.llm_name,
            "num_calls": len(self._call_trace),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "calls": self._call_trace,
        }

    def save_trace(self, path: str):
        with open(path, "w") as f:
            json.dump(self.get_trace_summary(), f, indent=2)


class VLLMLLM(BaseLLM):
    """Backend for local vLLM deployments (e.g. Qwen, Gemma).
    """

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config=llm_config)
        self.client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            max_retries=0,
            timeout=1200,  # 20-min timeout to avoid infinite hangs from vLLM
        )

    def _server_get(self, path: str, body: dict = None) -> dict:
        """GET (or POST ``body`` to) ``path`` on the vLLM server, outside the /v1 prefix."""
        url = re.sub(r"/v1/?$", "", self.base_url) + path
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())

    def _room_for_output(self, messages: list) -> int | None:
        """Output tokens that still fit in the server's context window after the prompt, or None if unknown."""
        try:
            window = next(m["max_model_len"] for m in self._server_get("/v1/models")["data"]
                         if m["id"] == self.llm_name)
            prompt = self._server_get("/tokenize", {
                "model": self.llm_name, "messages": messages, "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            })["count"]
        except Exception:
            return None
        return window - prompt

    @staticmethod
    def _extract_think_block(text: str) -> tuple:
        """Extract <think>...</think> block and return (thinking, stripped_content)."""
        m = re.search(r"<think>(.*?)</think>\s*", text, flags=re.DOTALL)
        if m:
            thinking = m.group(1).strip()
            stripped = text[:m.start()] + text[m.end():]
            return thinking, stripped.strip()
        return None, text

    def _record_trace_with_thinking(self, prompt_tokens: int, completion_tokens: int,
                                    thinking: str = None, prompt: str = None, response: str = None):
        entry = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        if prompt is not None:
            entry["prompt"] = prompt
        if response is not None:
            entry["response"] = response
        if thinking is not None:
            entry["thinking"] = thinking
        self._call_trace.append(entry)

    def run(self, prompt: str, system_prompt: str = None) -> str:
        system_prompt = system_prompt or "You are a helpful assistant."

        max_retries = 3
        retry_delay = 30
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        max_output_tokens = self.max_output_tokens

        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=self.llm_name,
                    messages=messages,
                    **self._build_sampling_kwargs(),
                )
                kwargs["max_completion_tokens"] = max_output_tokens
                # only add stop words when disable thinking, since stop words can also apply to reasoning process
                if not self.enable_thinking:
                    kwargs["stop"] = ["\nObservation: "]
                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                content = (msg.content or "").strip()
                usage = response.usage

                # vLLM with --reasoning-parser puts thinking in reasoning_content
                reasoning_content = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                if reasoning_content:
                    self._record_trace_with_thinking(
                        usage.prompt_tokens, usage.completion_tokens,
                        thinking=reasoning_content,
                        prompt=prompt, response=content,
                    )
                    return content

                # Fallback: extract inline reasoning blocks if the server did not split them.
                thinking, stripped = self._extract_think_block(content)
                self._record_trace_with_thinking(
                    usage.prompt_tokens, usage.completion_tokens,
                    thinking=thinking,
                    prompt=prompt, response=stripped,
                )
                return stripped

            except BadRequestError as e:
                # prompt + max_output_tokens exceeds the window: the request is rejected outright, so shrink the
                # output budget to what fits (once) instead of failing; a prompt that fills the window still fails
                too_long = "context length" in str(e) or "max_model_len" in str(e)  # vLLM words it both ways
                if too_long and max_output_tokens == self.max_output_tokens:
                    room = self._room_for_output(messages)
                    if room is not None and 0 < room < max_output_tokens:
                        print(f"[vLLM] prompt leaves {room} of {max_output_tokens} output tokens; retrying with {room}")
                        max_output_tokens = room
                        continue
                raise
            except Exception as e:
                error_message = str(e).lower()
                is_retriable = any(keyword in error_message for keyword in [
                    'rate limit', 'rate_limit', 'ratelimit',
                    'too many requests', 'quota', 'throttl',
                    'timed out', 'timeout',
                    'out of memory', 'oom',
                    '503', '502', '504',
                    'connection', 'refused',
                ])

                if is_retriable and attempt < max_retries - 1:
                    wait_time = retry_delay
                    print(f"[vLLM] {e} (Attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    if attempt == max_retries - 1:
                        print(f"[vLLM] Max retries reached. Error: {e}")
                    raise e


def get_llm_backend(llm_config: LLMConfig) -> VLLMLLM:
    """Create a vLLM backend for the configured model."""
    return VLLMLLM(llm_config)

if __name__ == "__main__":
    llm_config = LLMConfig({})
    LLM_controller = get_llm_backend(llm_config)

    system_prompt = "Answer the question in Chinese."
    question = "What is the capital of France? "
    response = LLM_controller.run(prompt=question, system_prompt=system_prompt)
    print(f"Response for question \"{question}\": \n{response}")
    print(f"Trace: {LLM_controller.get_trace_summary()}")