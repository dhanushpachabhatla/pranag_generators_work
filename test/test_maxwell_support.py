"""Tests for basic Maxwell support in the simulation generator.

Ensures the generator can produce a class for the `maxwell` pattern and that
the generated code is syntactically valid (i.e., templates compile).
"""

import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Model/models")))

from Model.models.simulation_generator import SimulationGenerator, EQUATION_PATTERNS, _validate_pattern_dict


def test_maxwell_pattern_validation_and_generation():
    gen = SimulationGenerator()
    # Ensure the pattern exists and validates
    assert 'maxwell' in EQUATION_PATTERNS
    pat = EQUATION_PATTERNS['maxwell']
    _validate_pattern_dict('maxwell', pat)

    # Generate a config and class code; this will run syntax validation
    cfg = gen.from_domain('maxwell')
    code = gen.generate_class(cfg)
    assert isinstance(code, str) and 'class' in code
