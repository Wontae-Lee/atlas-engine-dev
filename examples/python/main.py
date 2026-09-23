"""Select and run one maintained Atlas Python example case."""

import sys

from cases import cylinder


def main() -> None:
    """Parse the case name and step count from the command line."""
    case_name = sys.argv[1] if len(sys.argv) > 1 else "cylinder"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    if case_name == "cylinder":
        cylinder.run(steps)
        return

    raise SystemExit(f"unknown example case: {case_name}")


if __name__ == "__main__":
    main()
