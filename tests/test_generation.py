from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from genpy_llm.checkpointing import CheckpointError, save_checkpoint
from genpy_llm.config import (
    ConfigError,
    TokenizationConfig,
    VocabularyConfig,
    load_config,
)
from genpy_llm.generation import (
    GenerationError,
    GenerationResult,
    TextGenerator,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    create_generator_from_checkpoint,
)
from genpy_llm.tokenization import TextTokenizer
from genpy_llm.vocabulary import Vocabulary


class ScriptedLogitModel(nn.Module):
    def __init__(self, vocab_size: int, next_ids: list[int]) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = 4
        self.next_ids = list(next_ids)
        self.calls = 0
        self.context_lengths: list[int] = []
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del padding_mask
        self.context_lengths.append(int(input_ids.shape[1]))
        token_id = self.next_ids[min(self.calls, len(self.next_ids) - 1)]
        self.calls += 1
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
            -20.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, -1, token_id] = 20.0 + self.anchor * 0
        return logits


class FlatLogitModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = 4
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del padding_mask
        logits = torch.arange(self.vocab_size, dtype=torch.float32, device=input_ids.device)
        return logits.repeat(input_ids.shape[0], input_ids.shape[1], 1) + self.anchor * 0


def test_greedy_generation_returns_expected_result() -> None:
    generator = _generator(ScriptedLogitModel(9, [6, 7, 3]))

    result = generator.generate("Hello", max_new_tokens=5, do_sample=False)

    assert isinstance(result, GenerationResult)
    assert result.prompt == "Hello"
    assert result.prompt_tokens == ("Hello",)
    assert result.generated_token_ids == (6, 7, 3)
    assert result.generated_tokens == ("world", "!", "<EOS>")
    assert result.generated_text == "Hello world!"
    assert result.stopped_on_eos is True
    assert result.total_tokens == 4


def test_generation_honors_max_new_token_limit() -> None:
    generator = _generator(ScriptedLogitModel(9, [6, 7, 6]))

    result = generator.generate("Hello", max_new_tokens=2, do_sample=False, stop_on_eos=True)

    assert result.generated_token_ids == (6, 7)
    assert result.stopped_on_eos is False


def test_eos_can_be_generated_without_stopping_when_disabled() -> None:
    generator = _generator(ScriptedLogitModel(9, [3, 6]))

    result = generator.generate("Hello", max_new_tokens=2, do_sample=False, stop_on_eos=False)

    assert result.generated_token_ids == (3, 6)
    assert result.stopped_on_eos is False


def test_context_window_is_truncated() -> None:
    model = ScriptedLogitModel(9, [6, 6, 6, 6])
    generator = _generator(model, context_length=2)

    generator.generate("Hello world", max_new_tokens=4, do_sample=False, stop_on_eos=False)

    assert model.context_lengths == [2, 2, 2, 2]


def test_prompt_tokenization_encoding_and_unknown_prompt_tokens() -> None:
    generator = _generator(ScriptedLogitModel(9, [6]))

    result = generator.generate("Mystery", max_new_tokens=1, do_sample=False)

    assert result.prompt_tokens == ("Mystery",)
    assert result.prompt_token_ids == (generator.vocabulary.unknown_id,)


def test_temperature_validation_and_scaling() -> None:
    logits = torch.tensor([1.0, 2.0])

    assert torch.equal(apply_temperature(logits, 2.0), torch.tensor([0.5, 1.0]))
    with pytest.raises(GenerationError, match="temperature"):
        apply_temperature(logits, 0)


def test_top_k_filtering_keeps_only_configured_count() -> None:
    logits = torch.tensor([1.0, 5.0, 3.0])

    filtered = apply_top_k(logits, 2)

    assert filtered[0] == torch.finfo(logits.dtype).min
    assert filtered[1] == 5.0
    assert filtered[2] == 3.0


def test_top_p_filtering_keeps_at_least_one_token() -> None:
    logits = torch.tensor([10.0, 1.0, 0.5])

    filtered = apply_top_p(logits, 0.5)

    assert filtered[0] == 10.0
    assert filtered[1] == torch.finfo(logits.dtype).min
    assert filtered[2] == torch.finfo(logits.dtype).min
    with pytest.raises(GenerationError, match="top_p"):
        apply_top_p(logits, 0)


def test_repetition_penalty_adjusts_seen_token_logits() -> None:
    logits = torch.tensor([2.0, -2.0, 4.0])
    generated_ids = torch.tensor([0, 1])

    penalized = apply_repetition_penalty(logits, generated_ids, 2.0)

    assert penalized[0] == 1.0
    assert penalized[1] == -4.0
    assert penalized[2] == 4.0


def test_sampling_generation_is_deterministic_with_fixed_seed() -> None:
    generator = _generator(FlatLogitModel(9))
    torch.manual_seed(7)
    first = generator.generate("Hello", max_new_tokens=3, top_k=3, do_sample=True)
    torch.manual_seed(7)
    second = generator.generate("Hello", max_new_tokens=3, top_k=3, do_sample=True)

    assert first.generated_token_ids == second.generated_token_ids


def test_greedy_generation_is_deterministic() -> None:
    generator = _generator(FlatLogitModel(9))

    first = generator.generate("Hello", max_new_tokens=3, do_sample=False)
    second = generator.generate("Hello", max_new_tokens=3, do_sample=False)

    assert first.generated_token_ids == second.generated_token_ids


def test_model_eval_mode_no_gradients_and_no_parameter_updates() -> None:
    model = ScriptedLogitModel(9, [6, 7])
    model.train()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    generator = _generator(model)

    with torch.enable_grad():
        generator.generate("Hello", max_new_tokens=2, do_sample=False)

    assert model.training is False
    for name, parameter in model.named_parameters():
        assert parameter.grad is None
        assert torch.equal(parameter.detach(), before[name])


def test_tamil_prompt_support() -> None:
    generator = _generator(ScriptedLogitModel(9, [3]))

    result = generator.generate("தமிழ்", max_new_tokens=1, do_sample=False)

    assert result.prompt_tokens == ("தமிழ்",)
    assert result.prompt_token_ids == (8,)


def test_invalid_prompt_is_rejected() -> None:
    generator = _generator(ScriptedLogitModel(9, [6]))

    with pytest.raises(GenerationError, match="prompt"):
        generator.generate("   ")


def test_cpu_behavior() -> None:
    generator = _generator(ScriptedLogitModel(9, [6]), device=torch.device("cpu"))

    result = generator.generate("Hello", max_new_tokens=1, do_sample=False)

    assert result.generated_token_ids == (6,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_optional_cuda_behavior() -> None:
    generator = _generator(ScriptedLogitModel(9, [6]), device=torch.device("cuda"))

    result = generator.generate("Hello", max_new_tokens=1, do_sample=False)

    assert result.generated_token_ids == (6,)


def test_missing_checkpoint_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        create_generator_from_checkpoint(
            tmp_path / "missing.pt",
            None,
            None,
            torch.device("cpu"),
        )


def test_corrupted_checkpoint_is_reported(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "bad.pt"
    checkpoint_path.write_text("not a torch checkpoint", encoding="utf-8")

    with pytest.raises((CheckpointError, GenerationError)):
        create_generator_from_checkpoint(
            checkpoint_path,
            None,
            None,
            torch.device("cpu"),
        )


def test_incompatible_vocabulary_size_is_reported(tmp_path: Path) -> None:
    config_path = _tiny_config_path(tmp_path)
    vocabulary_path = tmp_path / "vocab.json"
    _vocabulary().save(vocabulary_path)
    model = ScriptedLogitModel(4, [3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint_path = tmp_path / "mismatch.pt"
    save_checkpoint(checkpoint_path, model, optimizer, epoch=1, global_step=1)

    with pytest.raises(GenerationError, match="vocabulary size"):
        create_generator_from_checkpoint(
            checkpoint_path,
            config_path,
            vocabulary_path,
            torch.device("cpu"),
        )


def test_generation_config_validation(tmp_path: Path) -> None:
    config_data = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    config_data["generation"]["top_p"] = 2.0
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ConfigError, match="generation.top_p"):
        load_config(config_path)


def test_steps_1_to_16_import_surfaces_remain_available() -> None:
    from genpy_llm.checkpointing import load_checkpoint
    from genpy_llm.gpt import GPTModel
    from genpy_llm.losses import GPTCrossEntropyLoss
    from genpy_llm.optimizers import create_optimizer
    from genpy_llm.training import GPTTrainer

    assert GPTModel is not None
    assert GPTCrossEntropyLoss is not None
    assert create_optimizer is not None
    assert GPTTrainer is not None
    assert load_checkpoint is not None


def _generator(
    model: nn.Module,
    *,
    context_length: int = 4,
    device: torch.device | None = None,
) -> TextGenerator:
    device = device or torch.device("cpu")
    return TextGenerator(
        model=model,
        tokenizer=TextTokenizer(_tokenization_config()),
        vocabulary=_vocabulary(),
        device=device,
        context_length=context_length,
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
            "தமிழ்": 8,
        },
        frequencies=None,
        config=_vocabulary_config(),
    )


def _tiny_config_path(tmp_path: Path) -> Path:
    config_data = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    config_data["model"].update(
        {
            "context_length": 4,
            "embedding_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
        }
    )
    config_data["positional_encoding"]["max_sequence_length"] = 4
    config_data["dataset"]["context_length"] = 4
    config_data["dataset"]["stride"] = 4
    config_data["feed_forward"]["hidden_multiplier"] = 2
    config_data["feed_forward"]["dropout"] = 0.0
    config_data["attention"]["dropout"] = 0.0
    config_data["residual"]["dropout"] = 0.0
    config_data["transformer_block"].update(
        {
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
            "feed_forward_dropout": 0.0,
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return config_path
