"""Inspect the trained GenPy Code LLM tokenizer."""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.code_tokenizer import DEFAULT_TOKENIZER_PATH, CodeTokenizer

SAMPLES = (
    "def reverse_text(text):\n    return text[::-1]\n",
    '# தமிழ் கருத்து\nprint("வணக்கம்")\n',
)


def main() -> int:
    _configure_console()
    args = _parse_args()
    try:
        tokenizer = CodeTokenizer.from_file(_resolve(args.tokenizer))
    except Exception as exc:  # noqa: BLE001
        _report_error(exc, debug=args.debug)
        return 1
    print("Code tokenizer inspection")
    print("=========================")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print("Special token IDs:")
    for token in tokenizer.special_tokens:
        print(f"  {token}: {tokenizer.token_to_id(token)}")
    for sample in SAMPLES:
        ids = tokenizer.encode(sample)
        decoded = tokenizer.decode(ids)
        print()
        print("Sample:")
        print(sample)
        print(f"Token IDs: {ids[:40]}")
        print(f"Token strings: {[tokenizer.id_to_token(token_id) for token_id in ids[:20]]}")
        print("Decoded:")
        print(decoded)
        expected = unicodedata.normalize("NFC", sample) if tokenizer.is_phase5 else sample
        print(f"Exact normalized round trip: {decoded == expected}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a trained code tokenizer.")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _report_error(exc: Exception, *, debug: bool) -> None:
    if debug:
        logging.exception("Tokenizer inspection failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
