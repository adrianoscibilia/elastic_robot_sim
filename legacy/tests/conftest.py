"""Archived tests (R5_10 T-6): not collected by the live suite (pyproject testpaths).

Run them explicitly with ``pytest legacy/tests``.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
