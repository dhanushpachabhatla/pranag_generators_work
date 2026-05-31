"""Verify all registered domains can produce valid configs and optionally instantiate.

This test compiles generated code for every `EQUATION_PATTERNS` entry and, when
PyTorch is available, attempts to import and instantiate each generated class to
catch runtime issues early.
"""

import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Model/models")))

from Model.models.simulation_generator import check_registry_integrity


def test_registry_integrity_and_instantiation():
    """Run integrity checks and attempt instantiation when possible."""
    try:
        import torch
        can_instantiate = True
    except Exception:
        can_instantiate = False

    results = check_registry_integrity(instantiate_models=can_instantiate)
    errors = {k: v for k, v in results.items() if not v.startswith('ok')}
    if errors:
        # Provide a clear assertion message with the failing domains
        msgs = '\n'.join(f"{k}: {v}" for k, v in errors.items())
        pytest.fail(f"Registry integrity failures:\n{msgs}")
