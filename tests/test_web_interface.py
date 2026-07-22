from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from genpy_llm.config import ConfigError, TokenizationConfig, VocabularyConfig, load_config
from genpy_llm.generation import TextGenerator
from genpy_llm.tokenization import TextTokenizer
from genpy_llm.vocabulary import Vocabulary
from genpy_llm.web_interface import GenPyWebInterface, WebInterfaceState


class ScriptedWebModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = 8
        self.context_length = 4
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_id = 6 if self.calls == 0 else 3
        self.calls += 1
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
            -20.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, -1, token_id] = 20.0 + self.anchor * 0
        return logits


def test_web_interface_config_is_validated(tmp_path: Path) -> None:
    config_data = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    config_data["web_interface"]["port"] = 70000
    config_path = tmp_path / "bad_web.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="web_interface.port"):
        load_config(config_path)


def test_web_interface_builds_gradio_blocks_without_launch() -> None:
    pytest.importorskip("gradio")
    web = _web_interface()

    app = web.build()

    assert app.__class__.__name__ == "Blocks"


def test_web_generation_callback_returns_metrics() -> None:
    web = _web_interface()

    output, status, elapsed, token_count, tokens_per_second, eos_status = web.generate(
        "Hello",
        2,
        1.0,
        0,
        1.0,
        1.0,
        True,
    )

    assert output == "Hello world"
    assert status == "Generation complete"
    assert elapsed >= 0
    assert token_count == 2
    assert tokens_per_second >= 0
    assert eos_status == "Yes"


def test_web_generation_hides_errors_from_ui_users() -> None:
    web = _web_interface()

    output, status, elapsed, token_count, tokens_per_second, eos_status = web.generate(
        " ",
        2,
        1.0,
        0,
        1.0,
        1.0,
        True,
    )

    assert output == ""
    assert status.startswith("Generation failed:")
    assert "Traceback" not in status
    assert elapsed == 0
    assert token_count == 0
    assert tokens_per_second == 0
    assert eos_status == "No"


def test_web_clear_resets_fields() -> None:
    web = _web_interface()

    assert web.clear() == ("", "", "Ready", 0.0, 0, 0.0, "No")


def _web_interface() -> GenPyWebInterface:
    return GenPyWebInterface(
        generator=_generator(),
        config=load_config(),
        state=WebInterfaceState(
            checkpoint_path=Path("checkpoints/genpy_best.pt"),
            device="cpu",
            quantization="none",
            torch_compile=False,
            compile_mode="default",
        ),
    )


def _generator() -> TextGenerator:
    return TextGenerator(
        model=ScriptedWebModel(),
        tokenizer=TextTokenizer(_tokenization_config()),
        vocabulary=_vocabulary(),
        device=torch.device("cpu"),
        context_length=4,
    )


def _tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(
        method="word",
        preserve_case=True,
        preserve_punctuation=True,
        preserve_newlines=True,
        split_contractions=False,
        normalize_quotes=True,
        normalize_dashes=True,
        add_bos_token=False,
        add_eos_token=True,
        add_newline_token=True,
        bos_token="<BOS>",
        eos_token="<EOS>",
        newline_token="<NL>",
        unknown_token="<UNK>",
    )


def _vocabulary_config() -> VocabularyConfig:
    return VocabularyConfig(
        min_frequency=1,
        max_size=None,
        include_special_tokens=True,
        save_frequencies=True,
        strict_special_token_validation=True,
        pad_token="<PAD>",
        unknown_token="<UNK>",
        bos_token="<BOS>",
        eos_token="<EOS>",
        newline_token="<NL>",
        special_token_order=("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NL>"),
    )


def _vocabulary() -> Vocabulary:
    return Vocabulary(
        token_to_id={
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<NL>": 4,
            "Hello": 5,
            "world": 6,
            "!": 7,
        },
        frequencies=None,
        config=_vocabulary_config(),
    )
