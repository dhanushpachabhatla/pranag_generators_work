import importlib.util
import os
import sys
import tempfile

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.models.simulation_generator import SimulationGenerator, _dynamic_import_and_instantiate


@pytest.fixture(scope="module")
def generator():
    return SimulationGenerator()


DETECTION_CASES = [
    ("2D heat diffusion", "heat_2d"),
    ("heat diffusion", "heat"),
    ("2D incompressible Navier-Stokes", "navier_stokes"),
    ("Burgers equation", "burgers"),
    ("Schrodinger quantum wavefunction", "schrodinger"),
    ("Black-Scholes option pricing PDE", "black_scholes"),
    ("Lotka Volterra predator prey", "lotka_volterra"),
]


@pytest.mark.parametrize("hint,expected", DETECTION_CASES)
def test_equation_detection(generator, hint, expected):
    eq_type, _ = generator._detect_equation_type(hint)
    assert eq_type == expected


NONLINEAR_CASES = [
    "u*u_x + v*u_y",
    "v*u_y + u*u_x",
    "u^2 - d2u/dx2",
    "x**2 + y",
    "dx/dt = u*v",
    "u * du/dx",
    "u*u",
]


@pytest.mark.parametrize("equation", NONLINEAR_CASES)
def test_nonlinear_detection(generator, equation):
    info = generator._parse_equation_string(equation)
    assert info.is_nonlinear, f"Expected nonlinear detection for: {equation}"


def test_normalized_class_names(generator):
    cfg = generator.from_domain("black_scholes")
    assert cfg.class_name == "BlackScholesPINN"
    assert cfg.name.endswith("PINN")
    assert cfg.equation_info.equation_type == "black_scholes"

    cfg2 = generator.from_hint("2D incompressible Navier-Stokes")
    assert cfg2.class_name == "NavierStokesPINN"
    assert cfg2.equation_info.equation_type == "navier_stokes"


@pytest.mark.parametrize("hint,_expected", DETECTION_CASES)
def test_generated_code_compiles(generator, hint, _expected):
    cfg = generator.from_hint(hint)
    code = generator.generate_class(cfg)
    assert code.startswith("\"\"\"")
    compile(code, cfg.class_name + ".py", "exec")


@pytest.mark.parametrize("hint,_expected", DETECTION_CASES)
def test_generated_code_imports_and_instantiates(generator, hint, _expected):
    cfg = generator.from_hint(hint)
    code = generator.generate_class(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"{cfg.class_name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        instance = _dynamic_import_and_instantiate(path, cfg.class_name)
        import torch

        sample = torch.rand(1, cfg.input_dim)
        output = instance(sample)
        assert output.shape == (1, cfg.output_dim)

        if hasattr(instance, "physics_loss"):
            phys = instance.physics_loss(torch.rand(2, cfg.input_dim))
            assert phys.shape == () or phys.dim() == 0
