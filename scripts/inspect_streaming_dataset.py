"""Inspect the streaming Python code dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.code_tokenizer import CodeTokenizer
from genpy_llm.code_training import load_code_config
from genpy_llm.device import select_device
from genpy_llm.streaming_dataset import StreamingGPTDataset


def main() -> int:
    args = _parse_args()
    try:
        config = load_code_config(_resolve(args.config))
        tokenizer = CodeTokenizer.from_file(config.tokenizer.path)
        pattern = (
            config.streaming_dataset.train_pattern
            if args.split == "train"
            else config.streaming_dataset.validation_pattern
        )
        dataset = StreamingGPTDataset(
            config.project_root / pattern,
            tokenizer,
            text_field=config.streaming_dataset.text_field,
            context_length=config.streaming_dataset.context_length,
            stride=config.streaming_dataset.stride,
            append_eos=config.streaming_dataset.append_eos,
            pack_across_files=config.streaming_dataset.pack_across_files,
            shuffle_shards=False,
            incomplete_window_policy=config.streaming_dataset.incomplete_window_policy,
        )
        loader = DataLoader(dataset, batch_size=1, num_workers=config.streaming_dataset.num_workers)
        device = select_device(args.device)
        print("Streaming dataset inspection")
        print("============================")
        print(f"Shard count: {len(dataset.shard_paths)}")
        print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
        print(f"Context length: {config.streaming_dataset.context_length}")
        print(f"DataLoader worker count: {config.streaming_dataset.num_workers}")
        for index, batch in enumerate(loader):
            if index >= args.max_samples:
                break
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            print()
            print(f"Sample {index}")
            print(f"Input shape: {tuple(input_ids.shape)}")
            print(f"Target shape: {tuple(target_ids.shape)}")
            print(f"Attention-mask shape: {tuple(attention_mask.shape)}")
            print(f"First token IDs: {input_ids[0, :20].tolist()}")
            print(f"Decoded input: {tokenizer.decode(input_ids[0].cpu().tolist())[:300]}")
            print(f"Decoded target: {tokenizer.decode(target_ids[0].cpu().tolist())[:300]}")
            print(f"Padding tokens: {int((~attention_mask[0].bool()).sum().item())}")
    except Exception as exc:  # noqa: BLE001
        _report_error(exc, debug=args.debug)
        return 1
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect streaming code dataset samples.")
    parser.add_argument("--config", type=Path, default=Path("configs/code_small.yaml"))
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _report_error(exc: Exception, *, debug: bool) -> None:
    if debug:
        logging.exception("Streaming dataset inspection failed.")
    else:
        print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
