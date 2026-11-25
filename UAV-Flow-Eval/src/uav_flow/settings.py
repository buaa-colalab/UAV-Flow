# SPDX-License-Identifier: Apache-2.0
"""Typed TOML defaults shared by the two workflow argument parsers.

CLI values override TOML; TOML paths are relative to the settings file. No
settings file is discovered implicitly from the checkout or current directory.
"""

import argparse
import getpass
import math
import os
import re
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path


def default_ue_user() -> str | None:
    return None if os.geteuid() == 0 else getpass.getuser()


def absolute_path(value: str | Path, directory: Path | None = None) -> Path:
    text = os.path.expandvars(str(value))
    if re.search(r"\$[A-Za-z_]|\$\{", text):
        raise ValueError("Path contains an unset environment variable")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() if directory is None else directory) / path
    # Resolve relative components without following a venv interpreter symlink.
    return Path(os.path.abspath(path))


def adapter_indices(value: object) -> list[int]:
    if isinstance(value, str):
        try:
            value = [int(item.strip()) for item in value.split(",")]
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "Adapters must be comma-separated nonnegative integers"
            ) from error
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise argparse.ArgumentTypeError("Adapters must be a nonempty list of nonnegative integers")
    return value


def parse_settings(
    parser: argparse.ArgumentParser, section: str, argv: Sequence[str] | None = None
) -> argparse.Namespace:
    """Apply only declared options; action and one-shot mode flags stay on CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    default = os.environ.get("UAV_FLOW_CONFIG") or None
    parser.add_argument(
        "--settings",
        type=Path,
        default=default,
        help="Workflow TOML (or UAV_FLOW_CONFIG); CLI overrides this file",
    )
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--settings", default=default)
    selected, _ = probe.parse_known_args(argv)
    if selected.settings is not None:
        try:
            filename = absolute_path(selected.settings)
            with filename.open("rb") as stream:
                document = tomllib.load(stream)
            if set(document) - {"training", "evaluation"}:
                raise ValueError("Settings may contain only [training] and [evaluation] tables")
            if any(not isinstance(value, dict) for value in document.values()):
                raise ValueError("Workflow settings must be TOML tables")
            values = document.get(section, {})
            excluded = {"help", "settings", "action", "dry_run", "smoke", "score_only"}
            actions = {
                action.dest: action
                for action in parser._actions
                if action.option_strings and action.dest not in excluded
            }
            unknown = set(values) - set(actions)
            if unknown:
                raise ValueError("Unknown " + section + " settings: " + ", ".join(sorted(unknown)))
            defaults = {}
            converted: Path | str | bool | int | float | list[int]
            for key, value in values.items():
                action = actions[key]
                if action.type is Path:
                    if not isinstance(value, str) or not value:
                        raise ValueError(key + " must be a nonempty path string")
                    converted = absolute_path(value, filename.parent)
                elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                    if type(value) is not bool:
                        raise ValueError(key + " must be a boolean")
                    converted = value
                elif action.type is adapter_indices:
                    converted = adapter_indices(value)
                elif action.type is int:
                    if type(value) is not int:
                        raise ValueError(key + " must be an integer")
                    converted = value
                elif action.type is float:
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError(key + " must be a finite number")
                    converted = float(value)
                else:
                    if not isinstance(value, str):
                        raise ValueError(key + " must be a string")
                    converted = value
                if action.choices is not None and converted not in action.choices:
                    raise ValueError(key + " is not an allowed choice")
                defaults[key] = converted
            explicit = {token.partition("=")[0] for token in argv if token.startswith("--")}
            for group in parser._mutually_exclusive_groups:
                selected_actions = [
                    action
                    for action in group._group_actions
                    if explicit.intersection(action.option_strings)
                ]
                if selected_actions:
                    for action in group._group_actions:
                        if action not in selected_actions:
                            defaults.pop(action.dest, None)
            parser.set_defaults(**defaults, settings=filename)
        except (OSError, ValueError, argparse.ArgumentTypeError) as error:
            parser.error(f"Invalid workflow settings: {error}")
    args = parser.parse_args(argv)
    for action in parser._actions:
        if action.type is Path and getattr(args, action.dest, None) is not None:
            try:
                setattr(args, action.dest, absolute_path(getattr(args, action.dest)))
            except ValueError as error:
                parser.error(f"{action.dest}: {error}")
    return args
