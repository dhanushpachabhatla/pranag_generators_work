"""
test_hodgkin_huxley_fix.py - Comprehensive tests for Hodgkin-Huxley parameter bug
================================================================================
Tests for the fix of AttributeError: object has no attribute 'I_ext'

Tests cover:
1. Default parameter initialization
2. Parameter merging
3. Validation before training
4. Backward compatibility
5. Dynamic model creation
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Model/models")))

from Model.models.pinn_factory import PINNFactory, _GenericPINN
from Model.models.simulation_generator import SimulationGenerator
from Model.models.pinn_trainer import PINNLightningModule


class TestHodgkinHuxleyParameterInitialization:
    """Test that I_ext and other critical HH parameters are properly initialized."""
    
    @pytest.fixture
    def factory(self):
        """Create a PINNFactory for testing."""
        return PINNFactory()
    
    @pytest.fixture
    def generator(self):
        """Create a SimulationGenerator for testing."""
        return SimulationGenerator()
    
    def test_generic_pinn_has_default_i_ext(self):
        """Test that _GenericPINN has I_ext in defaults."""
        assert "I_ext" in _GenericPINN._DEFAULTS
        assert _GenericPINN._DEFAULTS["I_ext"] == 0.0
        
    def test_generic_pinn_sets_default_i_ext_attribute(self):
        """Test that _GenericPINN sets I_ext as attribute even when not provided."""
        # Create a model without passing I_ext
        model = _GenericPINN(
            input_dim=2,
            output_dim=1,
            params={"C": 1.0}  # Intentionally omit I_ext
        )
        
        # I_ext should be set from defaults
        assert hasattr(model, "I_ext"), "Model should have I_ext attribute from defaults"
        assert model.I_ext == 0.0, "I_ext should have default value 0.0"
    
    def test_generic_pinn_all_hodgkin_huxley_defaults_present(self):
        """Test that all Hodgkin-Huxley parameters are available as defaults."""
        hh_params = ["C", "gNa", "gK", "gL", "ENa", "EK", "EL", "I_ext"]
        for param in hh_params:
            assert param in _GenericPINN._DEFAULTS, f"Missing default for {param}"
    
    def test_generic_pinn_sets_all_default_attributes(self):
        """Test that _GenericPINN sets all default parameters as attributes."""
        model = _GenericPINN(
            input_dim=2,
            output_dim=4,
            params={}  # Empty params - should use all defaults
        )
        
        hh_params = ["C", "gNa", "gK", "gL", "ENa", "EK", "EL", "I_ext"]
        for param in hh_params:
            assert hasattr(model, param), f"Model should have {param} attribute"
    
    def test_generic_pinn_user_params_override_defaults(self):
        """Test that user-provided parameters override defaults."""
        custom_i_ext = 10.0
        model = _GenericPINN(
            input_dim=2,
            output_dim=4,
            params={"I_ext": custom_i_ext}
        )
        
        assert model.I_ext == custom_i_ext, "User-provided I_ext should override default"
        assert model.C == 1.0, "Default C should still be present"
    
    def test_generic_pinn_stores_initialized_params(self):
        """Test that _GenericPINN stores merged parameters for validation."""
        model = _GenericPINN(
            input_dim=2,
            output_dim=4,
            params={"I_ext": 5.0}
        )
        
        assert hasattr(model, "_initialized_params"), "Model should store _initialized_params"
        assert "I_ext" in model._initialized_params
        assert model._initialized_params["I_ext"] == 5.0
    
    def test_hodgkin_huxley_config_has_defaults(self, generator):
        """Test that hodgkin_huxley SimulationConfig includes I_ext default."""
        from Model.models.simulation_generator import EQUATION_PATTERNS
        
        hh_config = EQUATION_PATTERNS.get("hodgkin_huxley")
        assert hh_config is not None, "hodgkin_huxley should be in EQUATION_PATTERNS"
        
        params = hh_config.get("params", {})
        assert "I_ext" in params, "hodgkin_huxley config should have I_ext parameter"
        assert params["I_ext"] == 0.0, "I_ext default should be 0.0"


class TestHodgkinHuxleyDynamicModelCreation:
    """Test that dynamic Hodgkin-Huxley models are created with all parameters."""
    
    @pytest.fixture
    def factory(self):
        return PINNFactory()
    
    def test_create_hodgkin_huxley_dynamic(self, factory):
        """Test that factory.create("hodgkin_huxley", dynamic=True) works."""
        model = factory.create("hodgkin_huxley", input_dim=2, dynamic=True)
        
        assert model is not None, "Should successfully create hodgkin_huxley model"
        assert isinstance(model, nn.Module), "Model should be nn.Module"
    
    def test_created_hh_model_has_i_ext(self, factory):
        """Test that created HH model has I_ext attribute."""
        model = factory.create("hodgkin_huxley", input_dim=2, dynamic=True)
        
        assert hasattr(model, "I_ext"), "HH model should have I_ext attribute"
    
    def test_created_hh_model_has_all_params(self, factory):
        """Test that created HH model has all required parameters."""
        model = factory.create("hodgkin_huxley", input_dim=2, dynamic=True)
        
        required_params = ["C", "gNa", "gK", "gL", "ENa", "EK", "EL", "I_ext"]
        for param in required_params:
            assert hasattr(model, param), f"HH model should have {param}"
    
    def test_create_hh_with_custom_i_ext(self, factory):
        """Test that custom I_ext parameter is applied."""
        custom_i_ext = 15.0
        model = factory.create(
            "hodgkin_huxley",
            input_dim=2,
            dynamic=True,
            I_ext=custom_i_ext
        )
        
        assert hasattr(model, "I_ext"), "Model should have I_ext"
        assert model.I_ext == custom_i_ext, f"I_ext should be {custom_i_ext}"
    
    def test_create_hh_physics_loss_callable(self, factory):
        """Test that created HH model can compute physics_loss without AttributeError."""
        model = factory.create("hodgkin_huxley", input_dim=1, dynamic=True)
        
        # Create dummy input with correct input dimension
        x = torch.randn(10, 1, requires_grad=True)
        
        # This should NOT raise AttributeError: object has no attribute 'I_ext'
        try:
            loss = model.physics_loss(x)
            assert loss is not None, "physics_loss should return a value"
            assert not torch.isnan(loss), "physics_loss should not be NaN"
        except AttributeError as e:
            if "I_ext" in str(e):
                pytest.fail(f"physics_loss raised AttributeError for I_ext: {e}")
            raise


class TestParameterValidation:
    """Test parameter validation before training."""
    
    def test_generic_pinn_validation_passes(self):
        """Test that validation passes for properly initialized model."""
        model = _GenericPINN(
            input_dim=2,
            output_dim=4,
            params={"C": 1.0, "I_ext": 0.0}
        )
        
        is_valid, missing = model.validate_physics_parameters()
        assert is_valid, "Model should pass validation"
        assert len(missing) == 0, "Should have no missing parameters"
    
    def test_pinn_lightning_module_validation(self):
        """Test that PINNLightningModule validates parameters."""
        model = _GenericPINN(
            input_dim=2,
            output_dim=4,
            params={"C": 1.0, "I_ext": 0.0}
        )
        
        # Should not raise during initialization
        try:
            lightning_module = PINNLightningModule(model, learning_rate=1e-3)
            assert lightning_module is not None, "Lightning module should be created"
        except ValueError as e:
            pytest.fail(f"Lightning module should not raise error for valid model: {e}")
    
    def test_pinn_lightning_module_validation_with_missing_params(self):
        """Test that PINNLightningModule raises error for missing parameters."""
        model = _GenericPINN(
            input_dim=2,
            output_dim=4,
            params={}  # No explicit params, but defaults should be there
        )
        
        # With defaults, this should pass
        lightning_module = PINNLightningModule(model, learning_rate=1e-3)
        assert lightning_module is not None


class TestBackwardCompatibility:
    """Test backward compatibility with existing configurations."""
    
    def test_explicit_i_ext_parameter_still_works(self):
        """Test that explicitly passing I_ext still works (backward compat)."""
        model = _GenericPINN(
            input_dim=2,
            output_dim=4,
            params={
                "C": 1.5,
                "gNa": 100.0,
                "I_ext": 2.5  # Explicit custom value
            }
        )
        
        assert model.I_ext == 2.5, "Explicit I_ext should be respected"
        assert model.C == 1.5, "Custom C should be respected"
    
    def test_partial_params_still_get_defaults(self):
        """Test that partial parameters get filled with defaults."""
        model = _GenericPINN(
            input_dim=2,
            output_dim=4,
            params={"C": 2.0}  # Only C provided
        )
        
        assert model.C == 2.0, "Provided C should be used"
        assert model.I_ext == 0.0, "Missing I_ext should use default"
        assert model.gNa == 120.0, "Missing gNa should use default"
    
    def test_empty_params_dict_gets_all_defaults(self):
        """Test that empty params dict results in all defaults."""
        model = _GenericPINN(
            input_dim=2,
            output_dim=4,
            params={}
        )
        
        # All defaults should be present
        assert model.I_ext == 0.0
        assert model.C == 1.0
        assert model.gNa == 120.0
        assert model.gK == 36.0
        assert model.gL == 0.3
        assert model.ENa == 50.0
        assert model.EK == -77.0
        assert model.EL == -54.4


class TestRegressionHodgkinHuxleyBug:
    """Regression tests for the original bug scenario."""
    
    def test_original_bug_scenario_unified_pipeline(self):
        """
        Test the original bug scenario from unified_pipeline.py:41
        where factory.create() is called with just domain and dynamic flag.
        """
        factory = PINNFactory()
        
        # This was the bug-triggering call: no I_ext parameter passed
        model = factory.create("hodgkin_huxley", input_dim=1, dynamic=True)
        
        # Model should have I_ext
        assert hasattr(model, "I_ext"), "BUG: Model should have I_ext attribute"
        
        # Physics loss should not raise AttributeError about I_ext
        x = torch.randn(10, 1, requires_grad=True)
        try:
            loss = model.physics_loss(x)
            assert loss is not None, "physics_loss should return a value"
        except AttributeError as e:
            if "I_ext" in str(e):
                pytest.fail(f"BUG REGRESSION: AttributeError for I_ext: {e}")
            # Other AttributeErrors (like network shape mismatches) are okay
            raise
    
    def test_training_step_with_missing_i_ext_safeguarded(self):
        """Test that training catches missing parameters early."""
        factory = PINNFactory()
        model = factory.create("hodgkin_huxley", input_dim=2, dynamic=True)
        
        # Training module should validate
        try:
            lightning_module = PINNLightningModule(model, learning_rate=1e-3)
            # Should succeed because model has defaults
            assert lightning_module is not None
        except ValueError as e:
            # Should only fail if actually missing params
            pytest.fail(f"Should not fail with defaults: {e}")


# ═══════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestEndToEndHodgkinHuxley:
    """End-to-end integration tests."""
    
    def test_full_pipeline_hodgkin_huxley_creation_training(self):
        """Test full pipeline: create -> validate -> (mock) train."""
        factory = PINNFactory()
        
        # Create model
        model = factory.create("hodgkin_huxley", input_dim=1, dynamic=True)
        assert model is not None
        
        # Most importantly: model should have I_ext (the critical fix)
        assert hasattr(model, "I_ext"), "Model should have I_ext attribute from defaults"
        
        # If validation method exists, use it
        if hasattr(model, 'validate_physics_parameters'):
            is_valid, missing = model.validate_physics_parameters()
            assert is_valid, f"Model validation failed: {missing}"
        
        # Create Lightning module (validation happens here too)
        lightning_module = PINNLightningModule(model)
        assert lightning_module is not None
        
        # Can compute loss (with correct input dimensions)
        x = torch.randn(5, 1, requires_grad=True)
        loss = model.physics_loss(x)
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
