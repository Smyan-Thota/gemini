"""Compatibility entrypoint for the supported multi-step sequence host command."""
from .sequence_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
