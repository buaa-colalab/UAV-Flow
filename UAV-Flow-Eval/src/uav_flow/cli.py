# SPDX-License-Identifier: Apache-2.0
"""Public entry point; importing it does not initialize a simulator."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] == "train-env":
            from tools.rl_envs_linux import main as train_main

            return train_main(argv[1:]) or 0
        if argv and argv[0] == "evaluate":
            from evaluation.run_single_model import cli as evaluate_main

            return evaluate_main(argv[1:])
        parser = argparse.ArgumentParser(
            description="Training environments and one-model closed-loop evaluation."
        )
        parser.add_argument(
            "command",
            choices=("train-env", "evaluate"),
            help="Use COMMAND --help for workflow options",
        )
        parser.parse_args(argv)
    except Exception as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
