"""Generate Python code with a GenPy Code LLM checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.code_generation import generate_code_text, load_code_model_for_generation
from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.code_training import load_code_config, select_device
from genpy_llm.utils import set_seed


def main() -> int:
    _configure_console()
    args = _parse_args()
    try:
        config = load_code_config(_resolve(args.config))
        set_seed(args.seed if args.seed is not None else config.seed)
        tokenizer = CodeTokenizer.from_file(config.tokenizer.path)
        device = select_device(args.device or config.training.device)
        model = load_code_model_for_generation(
            config=config,
            tokenizer=tokenizer,
            checkpoint_path=_resolve(args.checkpoint),
            device=device,
        )
        instruction_mode = (
            "instruct" in args.checkpoint.name or "instruction" in args.checkpoint.name
        )
        result = generate_code_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens or config.generation.max_new_tokens,
            temperature=args.temperature or config.generation.temperature,
            top_k=args.top_k if args.top_k is not None else config.generation.top_k,
            top_p=args.top_p if args.top_p is not None else config.generation.top_p,
            repetition_penalty=args.repetition_penalty or config.generation.repetition_penalty,
            do_sample=not args.greedy if args.greedy else config.generation.do_sample,
            stop_on_eos=config.generation.stop_on_eos,
            instruction_mode=instruction_mode,
            code_only=args.code_only,
            context_length=config.model.context_length,
        )
    except Exception as exc:  # noqa: BLE001
        _report_error(exc, debug=args.debug)
        return 1
    print("GenPy Code Generation")
    print("=====================")
    print(f"Checkpoint: {_resolve(args.checkpoint)}")
    print(f"Device: {device}")
    print(f"Instruction mode: {instruction_mode}")
    print(f"Generated tokens: {len(result.generated_token_ids)}")
    print(f"Tokens per second: {result.tokens_per_second:.2f}")
    print(f"Stopped on EOS: {result.stopped_on_eos}")
    print()
    print(result.text)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Python code from a code checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/code_small.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _report_error(exc: Exception, *, debug: bool) -> None:
    if debug:
        logging.exception("Code generation failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
