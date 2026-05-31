"""
test_equation_pattern_validation.py - Validation tests for EQUATION_PATTERNS and LOSS_TEMPLATES.
=====================================================================================

This file verifies that the domain metadata registry is validated and rejects malformed
entries before any generation logic runs.
"""

import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Model/models")))

from Model.models.simulation_generator import (
    EQUATION_PATTERNS,
    LOSS_TEMPLATES,
    _validate_equation_patterns,
    _validate_pattern_dict,
    _validate_loss_templates,
)


def test_validate_equation_patterns_with_valid_registry():
    """Valid registry should pass the import-time validation checks."""
    _validate_equation_patterns()


def test_validate_pattern_missing_required_field_raises():
    """Missing required fields in a pattern should raise a clear error."""
    bad_pattern = {
        "keywords": ["toy"],
        "eq_hint": "du/dt = a * d2u/dx2",
        "domain_class": "physics",
        "independent": ["t", "x"],
        "dependent": ["u"],
        # "params" omitted on purpose
        "loss_template": "heat_1d",
        "description": "Missing params field",
    }

    with pytest.raises(ValueError, match=r"missing required fields"):
        _validate_pattern_dict("broken_heat", bad_pattern)


def test_validate_pattern_with_invalid_loss_template_raises():
    """Patterns that reference unknown loss templates should fail fast."""
    bad_pattern = {
        "keywords": ["test"],
        "eq_hint": "du/dt = alpha * d2u/dx2",
        "domain_class": "physics",
        "independent": ["t", "x"],
        "dependent": ["u"],
        "params": {"alpha": 0.01},
        "loss_template": "nonexistent_template",
        "description": "Invalid loss template",
    }

    with pytest.raises(ValueError, match=r"unknown template"):
        _validate_pattern_dict("broken_heat", bad_pattern)


def test_validate_pattern_with_invalid_parameter_name_raises():
    """Invalid parameter identifiers should be rejected during validation."""
    bad_pattern = {
        "keywords": ["test"],
        "eq_hint": "du/dt = alpha * d2u/dx2",
        "domain_class": "physics",
        "independent": ["t", "x"],
        "dependent": ["u"],
        "params": {"1alpha": 0.01},
        "loss_template": "heat_1d",
        "description": "Invalid parameter name",
    }

    with pytest.raises(ValueError, match=r"invalid parameter name"):
        _validate_pattern_dict("broken_heat", bad_pattern)


def test_validate_loss_templates_syntax_error_raises():
    """Invalid loss template code should be caught before generation."""
    bad_templates = dict(LOSS_TEMPLATES)
    bad_templates["corrupt_template"] = "def physics_loss(self, x):\n    return x+"

    with pytest.raises(ValueError, match=r"invalid Python syntax"):
        _validate_loss_templates(bad_templates)


def test_validate_equation_patterns_rejects_invalid_registry_entry():
    """The full registry validator should reject malformed entries from EQUATION_PATTERNS."""
    patched_patterns = dict(EQUATION_PATTERNS)
    patched_patterns["broken_heat"] = {
        "keywords": ["test"],
        "eq_hint": "du/dt = alpha * d2u/dx2",
        "domain_class": "physics",
        "independent": ["t", "x"],
        "dependent": ["u"],
        "params": {"alpha": 0.01},
        "loss_template": "missing_template",
        "description": "Invalid loss template",
    }

    with pytest.raises(ValueError, match=r"unknown template"):
        _validate_pattern_dict("broken_heat", patched_patterns["broken_heat"])


def test_validate_pattern_independent_dependent_overlap_raises():
    """Having the same name in independent and dependent should be rejected."""
    bad_pattern = {
        "keywords": ["test"],
        "eq_hint": "du/dt = alpha * d2u/dx2",
        "domain_class": "physics",
        "independent": ["t", "u"],
        "dependent": ["u"],
        "params": {"alpha": 0.01},
        "loss_template": "heat_1d",
        "description": "Overlap in vars",
    }

    with pytest.raises(ValueError):
        _validate_pattern_dict("overlap", bad_pattern)


def test_validate_pattern_missing_template_param_raises():
    """If a loss template references `self.<name>` that isn't in params, fail fast."""
    # Use the existing beam template which references `self.q` in its code.
    bad_pattern = {
        "keywords": ["beam"],
        "eq_hint": "EI*d4w/dx4 = q(x)",
        "domain_class": "materials",
        "independent": ["x"],
        "dependent": ["w"],
        "params": {"EI": 1.0},  # deliberately omit `q`
        "loss_template": "euler_bernoulli_beam",
        "description": "Missing q param",
    }

    with pytest.raises(ValueError, match=r"missing parameters referenced by loss template"):
        _validate_pattern_dict("beam_broken", bad_pattern)
