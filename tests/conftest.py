"""Ensure the simple source tree is importable from pytest's console entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
