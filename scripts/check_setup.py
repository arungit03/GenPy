"""Verify the local GenPy LLM development setup."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import gradio as gr
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genpy_llm.config import load_config
from genpy_llm.device import select_device
from genpy_llm.utils import ensure_directories


def main() -> int:
    """Run setup checks and print a compact report."""

    checks: list[tuple[str, bool, str]] = []

    config = None
    try:
        checks.append(("Python 3.11+", sys.version_info >= (3, 11), platform.python_version()))
        checks.append(("PyTorch import", True, torch.__version__))
        checks.append(("Gradio import", True, gr.__version__))
        checks.append(("CUDA availability", True, str(torch.cuda.is_available())))
        checks.append(("MPS availability", True, str(_mps_available())))

        config = load_config()
        checks.append(("Configuration loading", True, str(PROJECT_ROOT / "configs" / "base.yaml")))

        device = select_device(config.training.device)
        checks.append(("Selected device", True, str(device)))

        required_dirs = [
            config.paths.data_dir,
            config.paths.raw_data_dir,
            config.paths.processed_data_dir,
            config.paths.tokenized_data_dir,
            config.data.vocabulary_dir,
            config.data.dataset_dir,
            config.paths.checkpoints_dir,
            config.paths.logs_dir,
            config.paths.notebooks_dir,
        ]
        ensure_directories(required_dirs)
        missing_dirs = [path for path in required_dirs if not path.exists()]
        checks.append(
            ("Required directories", not missing_dirs, _format_paths(missing_dirs) or "ok")
        )

        tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        result = tensor.mean().item()
        checks.append(("Tensor operation", abs(result - 2.0) < 1e-6, f"mean={result:.1f}"))
    except Exception as exc:  # noqa: BLE001 - this script should report setup failures clearly.
        checks.append(("Setup verification", False, f"{type(exc).__name__}: {exc}"))

    _print_report(checks)
    return 0 if all(passed for _, passed, _ in checks) and config is not None else 1


def _mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def _format_paths(paths: list[Path]) -> str:
    return ", ".join(str(path) for path in paths)


def _print_report(checks: list[tuple[str, bool, str]]) -> None:
    print("GenPy LLM setup check")
    print("=" * 24)
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")

    if all(passed for _, passed, _ in checks):
        print("\nSuccess: GenPy LLM Step 1 setup is ready.")
    else:
        print("\nFailure: Fix the failed check above and run this script again.")


if __name__ == "__main__":
    raise SystemExit(main())
