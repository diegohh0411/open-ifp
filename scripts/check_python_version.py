#!/usr/bin/env python3
"""Reject setup interpreters that do not match the released protocol."""

from __future__ import annotations

import argparse
import platform
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--actual", default=platform.python_version())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.actual != args.expected:
        print(
            f"Need Python {args.expected}; found {args.actual}.",
            file=sys.stderr,
        )
        return 1
    print(f"Python {args.actual} matches protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
