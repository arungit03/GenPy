#!/usr/bin/env python3
"""Run a non-invasive GenPy model quality diagnostic.

This script reads datasets, reports, metrics, checkpoints, and tokenizer/model
artifacts. It does not train, save checkpoints, or modify the transformer
architecture.
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import ast
import builtins
import gzip
import json
import math
import re
import statistics
import sys
import textwrap
import time
import tokenize
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from api.config import detect_api_device, load_api_config
from genpy_llm.checkpointing import load_checkpoint
from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.fine_tuning import load_phase7_config
from genpy_llm.instruction_dataset import load_instruction_records
from genpy_llm.instruction_generation import format_generation_prompt
from genpy_llm.pretraining import create_phase6_model
from genpy_llm.pretraining_generation import CodeGenerationSettings, generate_code_sample
from genpy_llm.quantization import load_quantized_checkpoint

UTC = timezone.utc


BENCHMARK_PROMPTS = (
    "Hello",
    "1 + 1 =",
    "Write bubble sort",
    "Reverse a linked list",
    "Explain Python decorators",
    "Fibonacci",
    "Prime numbers",
)
CONVERSATION_MARKERS = ("<|system|>", "<|user|>", "<|assistant|>")
REPORT_DIR = PROJECT_ROOT / "reports" / "model_quality"


def main() -> int:
    args = _parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or REPORT_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    phase7_config = load_phase7_config(args.phase7_config)
    tokenizer = CodeTokenizer.from_file(phase7_config.data.tokenizer)
    api_config = load_api_config(args.api_config)
    effective_checkpoint = _effective_checkpoint(api_config)

    print(f"diagnostic_output={output_dir}")
    print("analyzing_pretraining_dataset")
    pretraining = analyze_pretraining_dataset(tokenizer)
    print("analyzing_instruction_dataset")
    instruction = analyze_instruction_dataset(phase7_config, tokenizer)
    print(f"analyzing_checkpoint checkpoint={effective_checkpoint}")
    checkpoint = analyze_checkpoint(effective_checkpoint)

    print("loading_model_for_benchmarks")
    device = detect_api_device(args.device or api_config.device)
    benchmark = run_benchmarks(
        checkpoint_path=effective_checkpoint,
        plain_checkpoint_path=api_config.resolve_path(api_config.checkpoint),
        phase7_config=phase7_config,
        api_config=api_config,
        tokenizer=tokenizer,
        device=device,
        responses_dir=responses_dir,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        force_greedy=args.greedy,
    )

    diagnosis = diagnose_issue(
        pretraining=pretraining,
        instruction=instruction,
        checkpoint=checkpoint,
        benchmark=benchmark,
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "run_id": run_id,
        "api_config": str((PROJECT_ROOT / args.api_config).resolve())
        if not Path(args.api_config).is_absolute()
        else str(Path(args.api_config)),
        "phase7_config": str((PROJECT_ROOT / args.phase7_config).resolve())
        if not Path(args.phase7_config).is_absolute()
        else str(Path(args.phase7_config)),
        "pretraining_dataset": pretraining,
        "instruction_dataset": instruction,
        "checkpoint": checkpoint,
        "benchmarks": benchmark,
        "diagnosis": diagnosis,
    }
    json_path = output_dir / "diagnostic_report.json"
    markdown_path = output_dir / "diagnostic_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={markdown_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-config", default="configs/api.yaml")
    parser.add_argument("--phase7-config", default="configs/finetuning.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Override API sampling and run deterministic greedy decoding.",
    )
    return parser.parse_args()


def _effective_checkpoint(api_config: Any) -> Path:
    quantized = api_config.resolve_path(api_config.quantized_checkpoint)
    if quantized is not None:
        return quantized
    checkpoint = api_config.resolve_path(api_config.checkpoint)
    if checkpoint is None:
        raise ValueError("api checkpoint path is not configured")
    return checkpoint


def analyze_pretraining_dataset(tokenizer: CodeTokenizer) -> dict[str, Any]:
    stats = _read_json(PROJECT_ROOT / "data/pretraining/statistics.json")
    index = _read_json(PROJECT_ROOT / "data/pretraining/index.json")
    token_stats = _read_json(PROJECT_ROOT / "reports/pretraining/token_statistics.json")
    duplicate_report = _read_json(PROJECT_ROOT / "reports/pretraining/duplicate_report.json")
    source_report = _read_json(PROJECT_ROOT / "reports/pretraining/source_report.json")
    corpus_report = _read_json(PROJECT_ROOT / "reports/pretraining/corpus_report.json")
    manifest_path = PROJECT_ROOT / "data/pretraining/corpus_manifest.jsonl"
    manifest_summary = summarize_pretraining_manifest(manifest_path)
    source_mix = estimate_pretraining_source_mix(manifest_path)
    accepted = int(
        duplicate_report.get("accepted_count") or corpus_report.get("accepted_files") or 0
    )
    duplicate_count = int(
        duplicate_report.get("duplicate_count") or corpus_report.get("duplicates_removed") or 0
    )
    considered = accepted + duplicate_count
    duplicate_rate = duplicate_count / considered if considered else None
    used_vocab = int(
        token_stats.get("used_token_ids")
        or token_stats.get("observed_vocabulary_tokens")
        or 0
    )
    vocab_size = int(token_stats.get("vocab_size") or tokenizer.vocab_size)
    return {
        "paths": {
            "statistics": str(PROJECT_ROOT / "data/pretraining/statistics.json"),
            "index": str(PROJECT_ROOT / "data/pretraining/index.json"),
            "manifest": str(manifest_path),
        },
        "total_tokens": int(stats.get("total_tokens") or index.get("token_count") or 0),
        "packed_token_count": int(index.get("token_count") or 0),
        "vocab_size": vocab_size,
        "used_token_ids": used_vocab,
        "vocabulary_coverage": used_vocab / vocab_size if vocab_size else None,
        "duplicate_count": duplicate_count,
        "accepted_count": accepted,
        "duplicate_rate": duplicate_rate,
        "corpus_files": int(stats.get("corpus_files") or corpus_report.get("accepted_files") or 0),
        "byte_count": int(stats.get("byte_count") or index.get("byte_count") or 0),
        "language_token_counts": manifest_summary["language_token_counts"],
        "python_language_token_percentage": manifest_summary["python_language_token_percentage"],
        "python_code_percentage": source_mix["code_percentage"],
        "natural_language_percentage": source_mix["natural_language_percentage"],
        "source_mix_method": source_mix["method"],
        "source_report": source_report,
    }


def summarize_pretraining_manifest(path: Path) -> dict[str, Any]:
    language_tokens: Counter[str] = Counter()
    total_tokens = 0
    if not path.is_file():
        return {"language_token_counts": {}, "python_language_token_percentage": None}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            language = str(record.get("language") or "unknown")
            token_count = int(record.get("token_count") or 0)
            language_tokens[language] += token_count
            total_tokens += token_count
    return {
        "language_token_counts": dict(sorted(language_tokens.items())),
        "python_language_token_percentage": (
            language_tokens.get("Python", 0) / total_tokens if total_tokens else None
        ),
    }


def estimate_pretraining_source_mix(path: Path) -> dict[str, Any]:
    code_chars = 0
    natural_chars = 0
    unknown_files = 0
    sampled_files = 0
    if not path.is_file():
        return {
            "code_percentage": None,
            "natural_language_percentage": None,
            "method": "not computed; manifest missing",
        }
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            stored_path = record.get("stored_path") or record.get("relative_path")
            if not isinstance(stored_path, str):
                unknown_files += 1
                continue
            source_path = PROJECT_ROOT / "data/raw" / stored_path
            if not source_path.is_file():
                unknown_files += 1
                continue
            text = source_path.read_text(encoding="utf-8", errors="ignore")
            code, natural = _python_code_nl_character_counts(text)
            code_chars += code
            natural_chars += natural
            sampled_files += 1
    total = code_chars + natural_chars
    return {
        "code_percentage": code_chars / total if total else None,
        "natural_language_percentage": natural_chars / total if total else None,
        "code_characters": code_chars,
        "natural_language_characters": natural_chars,
        "sampled_files": sampled_files,
        "missing_source_files": unknown_files,
        "method": (
            "Character-weighted over pretraining source files: comments and AST docstrings "
            "count as natural language; remaining non-whitespace Python source counts as code."
        ),
    }


def _python_code_nl_character_counts(text: str) -> tuple[int, int]:
    natural_spans: list[tuple[int, int]] = []
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def offset(position: tuple[int, int]) -> int:
        line, col = position
        if line <= 0:
            return 0
        if line > len(offsets):
            return len(text)
        return min(len(text), offsets[line - 1] + col)

    try:
        tokens = tokenize.tokenize(BytesIO(text.encode("utf-8")).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                natural_spans.append((offset(token.start), offset(token.end)))
    except tokenize.TokenError:
        pass
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)
                    and hasattr(body[0], "lineno")
                    and hasattr(body[0], "end_lineno")
                ):
                    natural_spans.append(
                        (
                            offset((body[0].lineno, body[0].col_offset)),
                            offset(
                                (
                                    body[0].end_lineno or body[0].lineno,
                                    body[0].end_col_offset or 0,
                                )
                            ),
                        )
                    )
    except SyntaxError:
        pass
    mask = bytearray(len(text))
    for start, end in natural_spans:
        for index in range(max(0, start), min(len(mask), end)):
            mask[index] = 1
    natural = 0
    code = 0
    for index, char in enumerate(text):
        if char.isspace():
            continue
        if mask[index]:
            natural += 1
        else:
            code += 1
    return code, natural


def analyze_instruction_dataset(phase7_config: Any, tokenizer: CodeTokenizer) -> dict[str, Any]:
    paths = [
        phase7_config.data.train_path,
        phase7_config.data.validation_path,
        PROJECT_ROOT / "data/fine_tuning/test.jsonl",
    ]
    split_summaries = {}
    all_prompt_lengths = []
    all_response_lengths = []
    all_prompt_chars = []
    all_response_chars = []
    duplicate_hashes = Counter()
    marker_repetition = Counter()
    total_examples = 0
    for path in paths:
        if path is None or not Path(path).is_file():
            continue
        records = load_instruction_records(path)
        split = Path(path).stem
        total_examples += len(records)
        prompt_lengths = []
        response_lengths = []
        for record in records:
            prompt = phase7_config.template.format_prompt(record.instruction, record.input)
            response = record.output
            prompt_len = len(tokenizer.encode(prompt))
            response_len = len(tokenizer.encode(response))
            prompt_lengths.append(prompt_len)
            response_lengths.append(response_len)
            all_prompt_lengths.append(prompt_len)
            all_response_lengths.append(response_len)
            all_prompt_chars.append(len(prompt))
            all_response_chars.append(len(response))
            duplicate_hashes[record.output] += 1
            full = phase7_config.template.format_conversation(
                record.instruction,
                record.input,
                response,
            )
            for marker in CONVERSATION_MARKERS:
                if full.count(marker) > 1:
                    marker_repetition[marker] += 1
        split_summaries[split] = {
            "path": str(Path(path)),
            "examples": len(records),
            "average_prompt_tokens": _mean(prompt_lengths),
            "average_response_tokens": _mean(response_lengths),
        }
    duplicate_outputs = sum(count - 1 for count in duplicate_hashes.values() if count > 1)
    return {
        "paths": [str(path) for path in paths if path is not None and Path(path).is_file()],
        "examples": total_examples,
        "splits": split_summaries,
        "average_prompt_tokens": _mean(all_prompt_lengths),
        "average_response_tokens": _mean(all_response_lengths),
        "average_prompt_characters": _mean(all_prompt_chars),
        "average_response_characters": _mean(all_response_chars),
        "duplicate_output_rate": duplicate_outputs / total_examples if total_examples else None,
        "conversation_template_format": {
            "system_prefix": phase7_config.template.system_prefix,
            "user_prefix": phase7_config.template.user_prefix,
            "assistant_prefix": phase7_config.template.assistant_prefix,
            "system_prompt": phase7_config.template.system_prompt,
            "format": (
                "<|system|>\\n{system_prompt}\\n\\n"
                "<|user|>\\n{instruction}[\\n\\n{input}]\\n\\n"
                "<|assistant|>\\n{output}"
            ),
        },
        "records_with_repeated_markers": dict(marker_repetition),
        "mask_prompt_tokens": phase7_config.data.mask_prompt_tokens,
        "dataset_context_length": phase7_config.data.context_length,
    }


def analyze_checkpoint(path: Path) -> dict[str, Any]:
    payload = _torch_load(path)
    if "quantization" in payload:
        source = payload["quantization"].get("source_checkpoint")
        source_payload = _torch_load(Path(source)) if source and Path(source).is_file() else {}
        metadata = source_payload.get("metadata") if isinstance(source_payload, Mapping) else None
        optimizer_state = source_payload.get("optimizer_state_dict", {})
        source_parameter_count = (
            metadata.get("model_parameter_count") if isinstance(metadata, Mapping) else None
        )
        quantized_parameter_count = payload.get("quantization", {}).get("parameter_count")
        return {
            "path": str(path.resolve()),
            "kind": "quantized",
            "quantization": payload.get("quantization"),
            "source_checkpoint": source,
            "source_metadata": metadata,
            "parameter_count": source_parameter_count
            or quantized_parameter_count
            or _parameter_count_from_state(payload.get("model_state_dict", {})),
            "training_steps": None
            if not isinstance(metadata, Mapping)
            else metadata.get("global_step"),
            "final_validation_loss": None
            if not isinstance(metadata, Mapping)
            else metadata.get("validation_loss"),
            "perplexity": _perplexity(
                None if not isinstance(metadata, Mapping) else metadata.get("validation_loss")
            ),
            "optimizer_state": summarize_optimizer_state(optimizer_state),
        }
    metadata = payload.get("metadata")
    optimizer_state = payload.get("optimizer_state_dict", {})
    validation_loss = metadata.get("validation_loss") if isinstance(metadata, Mapping) else None
    return {
        "path": str(path.resolve()),
        "kind": "standard",
        "metadata": metadata,
        "model_config": payload.get("model_config"),
        "parameter_count": _parameter_count_from_state(payload.get("model_state_dict", {}))
        or (metadata or {}).get("model_parameter_count"),
        "training_steps": None
        if not isinstance(metadata, Mapping)
        else metadata.get("global_step"),
        "final_validation_loss": validation_loss,
        "perplexity": _perplexity(validation_loss),
        "optimizer_state": summarize_optimizer_state(optimizer_state),
    }


def summarize_optimizer_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, Mapping) or not state:
        return {"available": False}
    param_groups = state.get("param_groups", [])
    states = state.get("state", {})
    tensor_count = 0
    tensor_bytes = 0
    if isinstance(states, Mapping):
        for item in states.values():
            tensor_count += _count_tensors(item)
            tensor_bytes += _tensor_bytes(item)
    return {
        "available": True,
        "param_groups": len(param_groups) if isinstance(param_groups, list) else None,
        "state_entries": len(states) if isinstance(states, Mapping) else None,
        "tensor_count": tensor_count,
        "tensor_bytes": tensor_bytes,
    }


def run_benchmarks(
    *,
    checkpoint_path: Path,
    plain_checkpoint_path: Path | None,
    phase7_config: Any,
    api_config: Any,
    tokenizer: CodeTokenizer,
    device: torch.device,
    responses_dir: Path,
    seed: int,
    max_new_tokens: int | None,
    force_greedy: bool,
) -> dict[str, Any]:
    model = create_phase6_model(phase7_config.model, tokenizer)
    payload = _torch_load(checkpoint_path)
    quantization = None
    if "quantization" in payload:
        loaded = load_quantized_checkpoint(checkpoint_path, model, map_location="cpu")
        model = loaded.model
        quantization = loaded.method
    else:
        load_checkpoint(
            checkpoint_path,
            model,
            optimizer=None,
            map_location="cpu",
            restore_rng=False,
        )

    serving_device = torch.device("cpu") if quantization == "dynamic_int8" else device
    if serving_device.type != "cpu":
        model.to(serving_device)
    model.eval()

    generation = api_config.generation
    settings = CodeGenerationSettings(
        prompts=(),
        max_new_tokens=max_new_tokens or generation.max_new_tokens,
        temperature=generation.temperature,
        top_k=None,
        top_p=generation.top_p,
        do_sample=False if force_greedy else generation.do_sample,
        repetition_penalty=generation.repetition_penalty,
        stop_tokens=generation.stop_tokens,
    )
    results = []
    for index, prompt in enumerate(BENCHMARK_PROMPTS, start=1):
        formatted_prompt = format_generation_prompt(prompt, template=phase7_config.template)
        torch.manual_seed(seed + index)
        started = time.perf_counter()
        generated = generate_code_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            device=serving_device,
            context_length=phase7_config.model.context_length,
            settings=settings,
        )
        elapsed = time.perf_counter() - started
        raw_text = tokenizer.decode(generated.raw_generated_token_ids, skip_special_tokens=True)
        diagnostics = detect_response_issues(
            prompt=prompt,
            formatted_prompt=formatted_prompt,
            response=generated.text,
            raw_generated_text=raw_text,
            generated_token_ids=generated.raw_generated_token_ids,
        )
        result = {
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "response": generated.text,
            "raw_generated_text": raw_text,
            "generated_token_count": len(generated.raw_generated_token_ids),
            "emitted_token_count": len(generated.generated_token_ids),
            "stopped": generated.stopped,
            "elapsed_seconds": elapsed,
            "issues": diagnostics,
            "generator_diagnostic_report": generated.diagnostic_report,
        }
        response_path = responses_dir / f"{index:02d}_{_slug(prompt)}.json"
        response_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        result["response_path"] = str(response_path)
        results.append(result)
        print(f"benchmarked prompt={prompt!r} tokens={result['generated_token_count']}")
    summary_counts = Counter()
    for result in results:
        for name, value in result["issues"].items():
            if isinstance(value, bool) and value:
                summary_counts[name] += 1
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "plain_checkpoint_path": None
        if plain_checkpoint_path is None
        else str(plain_checkpoint_path.resolve()),
        "quantization": quantization,
        "device": str(serving_device),
        "settings": asdict(settings),
        "prompts": list(BENCHMARK_PROMPTS),
        "results": results,
        "issue_counts": dict(summary_counts),
    }


def detect_response_issues(
    *,
    prompt: str,
    formatted_prompt: str,
    response: str,
    raw_generated_text: str,
    generated_token_ids: Iterable[int],
) -> dict[str, Any]:
    token_ids = list(generated_token_ids)
    code_blocks = extract_code_blocks(response)
    code_like = looks_like_python(response)
    syntax_targets = code_blocks or ([response] if code_like else [])
    syntax_errors = []
    undefined_names = Counter()
    unfinished_functions = False
    for snippet in syntax_targets:
        parsed, error = parse_python(snippet)
        if error is not None:
            syntax_errors.append(error)
            if "expected an indented block" in error or snippet.rstrip().endswith(":"):
                unfinished_functions = True
            continue
        if parsed is not None:
            undefined_names.update(possible_undefined_names(parsed))
            unfinished_functions = unfinished_functions or has_unfinished_function(parsed, snippet)
    marker_counts = {marker: raw_generated_text.count(marker) for marker in CONVERSATION_MARKERS}
    return {
        "prompt_echoing": response.strip().startswith(prompt.strip())
        or raw_generated_text.strip().startswith(formatted_prompt.strip())
        or formatted_prompt.strip() in raw_generated_text,
        "repeated_conversation_markers": any(count > 0 for count in marker_counts.values()),
        "conversation_marker_counts": marker_counts,
        "repetition_loop": has_repetition_loop(token_ids, raw_generated_text),
        "max_consecutive_token_repeat": max_consecutive_repeat(token_ids),
        "repeated_ngram": repeated_ngram(token_ids),
        "invalid_python_syntax": bool(syntax_errors),
        "syntax_errors": syntax_errors,
        "possible_undefined_identifiers": sorted(undefined_names),
        "hallucinated_identifiers": sorted(undefined_names),
        "unfinished_functions": unfinished_functions
        or response.rstrip().endswith(":")
        or "TODO" in response
        or "..." in response,
    }


def diagnose_issue(
    *,
    pretraining: Mapping[str, Any],
    instruction: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    issue_counts = Counter(benchmark.get("issue_counts", {}))
    final_loss = checkpoint.get("final_validation_loss")
    pretraining_tokens = pretraining.get("total_tokens") or 0
    instruction_markers = sum(instruction.get("records_with_repeated_markers", {}).values())
    labels = {
        "A": "pretraining",
        "B": "instruction tuning",
        "C": "generation",
        "D": "stopping criteria",
        "E": "model capacity",
    }
    evidence = []
    scores = Counter()
    if pretraining_tokens and pretraining_tokens < 50_000_000:
        scores["A"] += 2
        evidence.append(f"Pretraining corpus is small for an LLM: {pretraining_tokens:,} tokens.")
    if final_loss is not None and final_loss > 3.0:
        scores["A"] += 1
        scores["B"] += 1
        evidence.append(
            f"Checkpoint validation loss is high: {final_loss:.4f} "
            f"(perplexity {_perplexity(final_loss):.2f})."
        )
    if instruction.get("duplicate_output_rate", 0) > 0.05:
        scores["B"] += 1
        evidence.append(
            f"Instruction duplicate-output rate is {instruction['duplicate_output_rate']:.2%}."
        )
    if instruction_markers:
        scores["B"] += 2
        evidence.append(
            f"Instruction data has repeated conversation markers in {instruction_markers} records."
        )
    else:
        evidence.append(
            "Instruction template inserts exactly one system/user/assistant marker per record."
        )
    if issue_counts.get("repetition_loop", 0):
        scores["C"] += 2
        evidence.append(
            f"Benchmark repetition loops: {issue_counts['repetition_loop']} / 7 prompts."
        )
    if issue_counts.get("repeated_conversation_markers", 0):
        scores["D"] += 2
        evidence.append(
            "Generated text includes conversation markers in "
            f"{issue_counts['repeated_conversation_markers']} / 7 prompts."
        )
    if issue_counts.get("prompt_echoing", 0):
        scores["C"] += 1
        evidence.append(f"Prompt echoing detected in {issue_counts['prompt_echoing']} / 7 prompts.")
    if checkpoint.get("parameter_count") and checkpoint["parameter_count"] < 100_000_000:
        scores["E"] += 1
        evidence.append(f"Model is small: {checkpoint['parameter_count']:,} parameters.")
    primary = max(scores, key=lambda key: (scores[key], -ord(key))) if scores else "C"
    return {
        "primary_issue": primary,
        "primary_issue_label": labels[primary],
        "scores": dict(scores),
        "evidence": evidence,
        "conclusion": _diagnosis_sentence(primary, labels[primary], evidence),
    }


def _diagnosis_sentence(primary: str, label: str, evidence: list[str]) -> str:
    return (
        f"The strongest supported diagnosis is {primary}. {label}. "
        "This is based only on collected dataset, checkpoint, and benchmark evidence."
    )


def extract_code_blocks(text: str) -> list[str]:
    blocks = []
    for match in re.finditer(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.I):
        block = textwrap.dedent(match.group(1)).strip()
        if block:
            blocks.append(block)
    return blocks


def looks_like_python(text: str) -> bool:
    return bool(
        re.search(r"(?m)^\s*(def|class|import|from|for|while|if|try|with)\b", text)
        or re.search(r"\b(return|yield|lambda|print)\b", text)
    )


def parse_python(text: str) -> tuple[ast.AST | None, str | None]:
    try:
        tree = ast.parse(text)
        compile(tree, "<genpy-diagnostic>", "exec")
        return tree, None
    except SyntaxError as exc:
        return None, f"{exc.msg} at line {exc.lineno}, column {exc.offset}"


def possible_undefined_names(tree: ast.AST) -> Counter[str]:
    defined = set(dir(builtins)) | {"self", "cls", "True", "False", "None"}
    loaded = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param)):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.alias):
            defined.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined:
            loaded[node.id] += 1
    return loaded


def has_unfinished_function(tree: ast.AST, text: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [item for item in node.body if not _is_docstring_expr(item)]
            if not body:
                return True
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                return True
            if len(body) == 1 and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and value.value is Ellipsis:
                    return True
    return "TODO" in text


def _is_docstring_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(getattr(node, "value", None), ast.Constant)
        and isinstance(node.value.value, str)
    )


def has_repetition_loop(token_ids: list[int], text: str) -> bool:
    if max_consecutive_repeat(token_ids) >= 8:
        return True
    ngram = repeated_ngram(token_ids)
    if ngram and ngram["count"] >= 4 and ngram["length"] >= 2:
        return True
    words = re.findall(r"\b\w+\b", text)
    if len(words) >= 20:
        most_common = Counter(words).most_common(1)[0]
        if most_common[1] / len(words) >= 0.35 and most_common[1] >= 8:
            return True
    return False


def max_consecutive_repeat(items: list[int]) -> int:
    best = 0
    current = 0
    previous = object()
    for item in items:
        if item == previous:
            current += 1
        else:
            current = 1
            previous = item
        best = max(best, current)
    return best


def repeated_ngram(token_ids: list[int]) -> dict[str, Any] | None:
    for length in range(12, 1, -1):
        counts = Counter(
            tuple(token_ids[index : index + length])
            for index in range(len(token_ids) - length + 1)
        )
        repeated = [(ngram, count) for ngram, count in counts.items() if count >= 3]
        if repeated:
            ngram, count = max(repeated, key=lambda item: item[1])
            return {"length": length, "count": count, "token_ids": list(ngram)}
    return None


def render_markdown(report: Mapping[str, Any]) -> str:
    pre = report["pretraining_dataset"]
    inst = report["instruction_dataset"]
    ckpt = report["checkpoint"]
    bench = report["benchmarks"]
    diag = report["diagnosis"]
    lines = [
        "# GenPy Model Quality Diagnostic",
        "",
        f"Created: {report['created_at']}",
        "",
        "## Conclusion",
        "",
        f"Primary issue: **{diag['primary_issue']}. {diag['primary_issue_label']}**",
        "",
        diag["conclusion"],
        "",
        "Evidence:",
        *[f"- {item}" for item in diag["evidence"]],
        "",
        "## Pretraining Dataset",
        "",
        f"- Total tokens: {_fmt_int(pre['total_tokens'])}",
        f"- Vocabulary coverage: {_fmt_pct(pre['vocabulary_coverage'])} "
        f"({_fmt_int(pre['used_token_ids'])} / {_fmt_int(pre['vocab_size'])} token IDs used)",
        f"- Duplicate rate: {_fmt_pct(pre['duplicate_rate'])} "
        f"({_fmt_int(pre['duplicate_count'])} duplicates / "
        f"{_fmt_int(pre['accepted_count'] + pre['duplicate_count'])} considered)",
        f"- Python language token percentage: {_fmt_pct(pre['python_language_token_percentage'])}",
        f"- Python code percentage: {_fmt_pct(pre['python_code_percentage'])}",
        f"- Natural-language percentage: {_fmt_pct(pre['natural_language_percentage'])}",
        f"- Source mix method: {pre['source_mix_method']}",
        "",
        "## Instruction Dataset",
        "",
        f"- Examples: {_fmt_int(inst['examples'])}",
        f"- Average prompt length: {inst['average_prompt_tokens']:.2f} tokens",
        f"- Average response length: {inst['average_response_tokens']:.2f} tokens",
        f"- Conversation template: `{inst['conversation_template_format']['format']}`",
        f"- Repeated marker records: {inst['records_with_repeated_markers'] or {}}",
        f"- Duplicate output rate: {_fmt_pct(inst['duplicate_output_rate'])}",
        "",
        "## Checkpoint",
        "",
        f"- Path: `{ckpt['path']}`",
        f"- Kind: {ckpt['kind']}",
        f"- Parameter count: {_fmt_int(ckpt['parameter_count'])}",
        f"- Training steps: {_fmt_int(ckpt['training_steps'])}",
        f"- Final validation loss: {_fmt_float(ckpt['final_validation_loss'])}",
        f"- Perplexity: {_fmt_float(ckpt['perplexity'])}",
        f"- Optimizer state: {ckpt['optimizer_state']}",
        "",
        "## Benchmarks",
        "",
        f"- Device: {bench['device']}",
        f"- Quantization: {bench['quantization']}",
        f"- Settings: `{bench['settings']}`",
        f"- Issue counts: {bench['issue_counts']}",
        "",
    ]
    for result in bench["results"]:
        issues = result["issues"]
        lines.extend(
            [
                f"### {result['prompt']}",
                "",
                f"- Response file: `{result['response_path']}`",
                f"- Tokens generated: {result['generated_token_count']}",
                f"- Stopped: {result['stopped']}",
                f"- Prompt echoing: {issues['prompt_echoing']}",
                f"- Repeated markers: {issues['repeated_conversation_markers']} "
                f"{issues['conversation_marker_counts']}",
                f"- Repetition loop: {issues['repetition_loop']}",
                f"- Invalid Python syntax: {issues['invalid_python_syntax']}",
                f"- Possible undefined identifiers: {issues['possible_undefined_identifiers']}",
                f"- Unfinished functions: {issues['unfinished_functions']}",
                "",
                "Response:",
                "",
                "```text",
                result["response"][:2000],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _torch_load(path: Path | str) -> Mapping[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Expected checkpoint mapping: {path}")
    return loaded


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            return json.load(file)
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[int]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _perplexity(loss: Any) -> float | None:
    if loss is None:
        return None
    return math.exp(min(20.0, float(loss)))


def _parameter_count_from_state(state: Any) -> int | None:
    if not isinstance(state, Mapping):
        return None
    total = 0
    for value in state.values():
        if isinstance(value, torch.Tensor):
            total += value.numel()
    return total or None


def _count_tensors(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, Mapping):
        return sum(_count_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_tensors(item) for item in value)
    return 0


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "prompt"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
