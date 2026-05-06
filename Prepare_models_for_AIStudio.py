#!/usr/bin/env python3
"""Compatibility wrapper for Prepare_models_for_AINavigator.py.
The orchestrator calls AIStudio, while the implementation is named AINavigator.
"""
from pathlib import Path
import runpy
import sys

script = Path(__file__).resolve().with_name("Prepare_models_for_AINavigator.py")
if not script.is_file():
    print(f"ERROR: missing {script}", file=sys.stderr)
    sys.exit(2)
runpy.run_path(str(script), run_name="__main__")
