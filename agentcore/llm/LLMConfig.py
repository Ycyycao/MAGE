import os

DEFAULT_LLM_NAME = "Qwen/Qwen3.6-27B"
DEFAULT_BASE_URL = "http://localhost:8004/v1"


class LLMConfig:
    """Configuration for a model served through vLLM's OpenAI-compatible API."""

    def __init__(self, config_dict: dict) -> None:
        self.config_dict = config_dict
        
        self.llm_name = config_dict.get("llm_name", DEFAULT_LLM_NAME)
        # Sampling: None leaves the value to the server (vLLM applies the model's generation_config).
        self.temperature = None
        self.top_p = None
        self.top_k = None
        self.presence_penalty = None
        self.seed = None
        self.enable_thinking: bool = True
        self.max_output_tokens: int = 65536
        # Client-side limit per request; a long thinking output from a busy server can need more than 20 min.
        self.request_timeout: float = 1200
        self.api_key: str = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.base_url = config_dict.get("base_url", DEFAULT_BASE_URL)

        self.__dict__.update(config_dict)