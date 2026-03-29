#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openpi_thor.host_integration import plan_host_pyproject_patch
from openpi_thor.host_integration import write_host_pyproject_patch


def _print_summary(plan, *, wrote: bool) -> None:
    print(f"Host root: {plan.host_root}")
    print(f"pyproject.toml: {plan.pyproject_path}")
    if wrote:
        print("Mode: write")
    else:
        print("Mode: preview")

    if plan.changed:
        print("\nChanges needed:" if not wrote else "\nApplied changes:")
        for item in plan.changed:
            print(f"- {item}")
    if plan.already_correct:
        print("\nAlready correct:")
        for item in plan.already_correct:
            print(f"- {item}")
    if plan.could_not_patch:
        print("\nCould not patch automatically:")
        for item in plan.could_not_patch:
            print(f"- {item}")
    if plan.errors:
        print("\nErrors:")
        for item in plan.errors:
            print(f"- {item}")
    elif not wrote:
        print("\nPreview only. Re-run with --write to update the host pyproject.toml.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview or apply the openpi-thor host pyproject patch.")
    parser.add_argument("--host-root", type=Path, required=True, help="Path to the host openpi repo root.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply the patch to the host root pyproject.toml. Without this flag the script only previews changes.",
    )
    args = parser.parse_args(argv)

    plan = write_host_pyproject_patch(args.host_root) if args.write else plan_host_pyproject_patch(args.host_root)
    _print_summary(plan, wrote=args.write)
    return 1 if plan.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
