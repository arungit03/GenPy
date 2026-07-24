from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import APIConfig
from api.inference import (
    GenPyInferenceService,
    LoadedModelInfo,
    _quantized_checkpoint_matches,
)
from api.schemas import GenerationResponse, HealthResponse, ModelResponse
from genpy_llm.conversation_formatter import ConversationTemplate
from genpy_llm.pretraining_generation import CodeGenerationResult, clean_assistant_response


class FakeService:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.chat_calls = 0

    def health(self) -> HealthResponse:
        return HealthResponse(status="healthy", device="cpu", model_loaded=True)

    def model_info(self) -> ModelResponse:
        return ModelResponse(
            model_name="GenPy GPT",
            parameter_count=123,
            checkpoint_path="/tmp/checkpoint.pt",
            quantization="fp16",
            lora_enabled=False,
            lora_adapter=None,
            device="cpu",
            tokenizer_path="/tmp/tokenizer.json",
            context_length=256,
            vocabulary_size=32000,
            loaded_at="2026-07-23T00:00:00+00:00",
        )

    def generate(self, request) -> GenerationResponse:
        self.generate_calls += 1
        return GenerationResponse(generated_text=f"{request.prompt} -> generated")

    def chat(self, request) -> GenerationResponse:
        self.chat_calls += 1
        return GenerationResponse(generated_text=request.messages[-1].content + " -> reply")


class FakeTokenizer:
    special_tokens = ("<pad>", "<unk>", "<bos>", "<eos>")
    bos_token_id = 2

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id] if text.startswith("<bos>") else [99]

    def decode(self, ids, *, skip_special_tokens: bool = True) -> str:
        values = list(ids)
        if skip_special_tokens:
            values = [item for item in values if item != self.bos_token_id]
        return " ".join(str(item) for item in values)


def test_health_endpoint() -> None:
    client = _client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "device": "cpu",
        "model_loaded": True,
    }


def test_model_endpoint() -> None:
    client = _client()

    response = client.get("/model")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "GenPy GPT"
    assert body["parameter_count"] == 123
    assert body["quantization"] == "fp16"
    assert body["lora_enabled"] is False


def test_generate_endpoint() -> None:
    service = FakeService()
    client = _client(service)

    response = client.post(
        "/generate",
        json={
            "prompt": "Write hello world.",
            "max_new_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"generated_text": "Write hello world. -> generated"}
    assert service.generate_calls == 1


def test_chat_endpoint() -> None:
    service = FakeService()
    client = _client(service)

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Write binary search."}]},
    )

    assert response.status_code == 200
    assert response.json() == {"generated_text": "Write binary search. -> reply"}
    assert service.chat_calls == 1


def test_cleanup_strips_conversation_markers_and_next_user_turn() -> None:
    output = clean_assistant_response(
        "<|assistant|>\n"
        "def bubble_sort(items):\n"
        "    return items\n"
        "<|user|>\n"
        "Write insertion sort."
    )

    assert output == "def bubble_sort(items):\n    return items"
    assert "<|system|>" not in output
    assert "<|user|>" not in output
    assert "<|assistant|>" not in output


@pytest.mark.parametrize("path", ["/generate", "/chat"])
def test_generation_endpoints_return_clean_assistant_reply_only(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    seen_prompts: list[str] = []

    def fake_generate_code_sample(**kwargs) -> CodeGenerationResult:
        prompt = kwargs["prompt"]
        seen_prompts.append(prompt)
        return CodeGenerationResult(
            prompt=prompt,
            text=(
                f"{prompt}"
                "<|assistant|>\n"
                "def bubble_sort(items):\n"
                "    return items\n"
                "<|user|>\n"
                "Write insertion sort."
            ),
            generated_token_ids=(11, 12, 13, 14),
            stopped=True,
        )

    monkeypatch.setattr("api.inference.generate_code_sample", fake_generate_code_sample)
    client = _client(_inference_service())
    payload = (
        {"prompt": "Write bubble sort."}
        if path == "/generate"
        else {"messages": [{"role": "user", "content": "Write bubble sort."}]}
    )

    response = client.post(path, json=payload)

    assert response.status_code == 200
    assert response.json() == {"generated_text": "def bubble_sort(items):\n    return items"}
    generated_text = response.json()["generated_text"]
    assert "Write bubble sort." not in generated_text
    assert "<|system|>" not in generated_text
    assert "<|user|>" not in generated_text
    assert "<|assistant|>" not in generated_text
    assert seen_prompts == [
        "<bos><|system|>\n"
        "You are GenPy, a Python coding assistant.\n\n"
        "<|user|>\n"
        "Write bubble sort.\n\n"
        "<|assistant|>\n"
    ]


def test_chat_prompt_uses_latest_user_turn_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_prompts: list[str] = []

    def fake_generate_code_sample(**kwargs) -> CodeGenerationResult:
        prompt = kwargs["prompt"]
        seen_prompts.append(prompt)
        return CodeGenerationResult(
            prompt=prompt,
            text="def is_even(number):\n    return number % 2 == 0",
            generated_token_ids=(11, 12, 13),
            stopped=False,
        )

    monkeypatch.setattr("api.inference.generate_code_sample", fake_generate_code_sample)
    client = _client(_inference_service())

    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "odd or even code"},
                {"role": "assistant", "content": "class In(BaseModel):\n    name = None"},
                {
                    "role": "user",
                    "content": "Write a Python program to check whether a number is odd or even.",
                },
            ]
        },
    )

    assert response.status_code == 200
    assert seen_prompts == [
        "<bos><|system|>\n"
        "You are GenPy, a Python coding assistant.\n\n"
        "<|user|>\n"
        "Write a Python program to check whether a number is odd or even.\n\n"
        "<|assistant|>\n"
    ]
    assert "class In(BaseModel)" not in seen_prompts[0]
    assert "odd or even code" not in seen_prompts[0]


def test_quantized_checkpoint_mismatch_is_ignored(tmp_path: Path) -> None:
    primary = tmp_path / "best_checkpoint.pt"
    source = tmp_path / "last_checkpoint.pt"
    quantized = tmp_path / "last_checkpoint_fp16.pt"
    primary.write_bytes(b"primary")
    source.write_bytes(b"source")
    torch.save({"quantization": {"source_checkpoint": str(source)}}, quantized)

    assert _quantized_checkpoint_matches(quantized, primary) is False


def test_quantized_checkpoint_matching_source_is_accepted(tmp_path: Path) -> None:
    primary = tmp_path / "best_checkpoint.pt"
    quantized = tmp_path / "best_checkpoint_fp16.pt"
    primary.write_bytes(b"primary")
    torch.save({"quantization": {"source_checkpoint": str(primary)}}, quantized)

    assert _quantized_checkpoint_matches(quantized, primary) is True


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/generate", {"prompt": "", "temperature": 0.7}),
        ("/generate", {"prompt": "ok", "temperature": 0}),
        ("/generate", {"prompt": "ok", "top_p": 1.5}),
        ("/generate", {"prompt": "ok", "max_new_tokens": 0}),
        ("/chat", {"messages": [{"role": "assistant", "content": "hello"}]}),
    ],
)
def test_invalid_requests_return_422(path: str, payload: dict[str, object]) -> None:
    client = _client()

    response = client.post(path, json=payload)

    assert response.status_code == 422


def test_docs_and_redoc_are_enabled() -> None:
    client = _client()

    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200


def test_startup_loads_service_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_from_config(_config: APIConfig) -> FakeService:
        nonlocal calls
        calls += 1
        return FakeService()

    monkeypatch.setattr("api.app.GenPyInferenceService.from_config", fake_from_config)
    app = create_app(config=APIConfig(checkpoint=Path("checkpoints/fine_tuned/best_checkpoint.pt")))

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/model").status_code == 200

    assert calls == 1


def _client(service: object | None = None) -> TestClient:
    app = create_app(
        config=APIConfig(checkpoint=Path("checkpoints/fine_tuned/best_checkpoint.pt")),
        service=service or FakeService(),
    )
    return TestClient(app)


def _inference_service() -> GenPyInferenceService:
    return GenPyInferenceService(
        model=torch.nn.Identity(),
        tokenizer=FakeTokenizer(),
        phase7_config=SimpleNamespace(template=ConversationTemplate()),
        api_config=APIConfig(checkpoint=Path("checkpoints/fine_tuned/best_checkpoint.pt")),
        info=LoadedModelInfo(
            model_name="GenPy GPT",
            checkpoint_path=Path("/tmp/checkpoint.pt"),
            quantization=None,
            lora_adapter=None,
            device=torch.device("cpu"),
            tokenizer_path=Path("/tmp/tokenizer.json"),
            context_length=256,
            vocabulary_size=32000,
            parameter_count=0,
            loaded_at="2026-07-23T00:00:00+00:00",
        ),
    )
