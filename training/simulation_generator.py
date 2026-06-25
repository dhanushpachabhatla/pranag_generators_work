"""
simulation_generator.py  —  Autonomous Simulation Generator
=============================================================
Creates PINN configurations and complete training-ready code from:
  • Plain equation strings  ("du/dt = D * d2u/dx2 + r*u*(1-u)")
  • Physics hints            ("heat diffusion in 2D plate")
  • Domain + variable names  (domain="fluid", vars=["u","v","p"])

Supported equation families (auto-detected):
  Heat / Diffusion      Navier-Stokes / Euler        Wave
  Schrodinger           Maxwell / Laplace / Poisson   Burgers
  Reaction-Diffusion    Advection-Diffusion           Elasticity
  Black-Scholes         SIR/SEIR Epidemiology         Lotka-Volterra
  Logistic / Growth     Hodgkin-Huxley                Gray-Scott
  Allen-Cahn            Cahn-Hilliard                 Korteweg-de Vries (KdV)
  Klein-Gordon          Euler-Bernoulli Beam          Darcy Flow
  Lid-Driven Cavity     ... + any custom equation

Usage:
    gen = SimulationGenerator()

    # From equation string
    cfg = gen.from_equation("du/dt = alpha * d2u/dx2",
                            variables=["u"], coords=["t", "x"],
                            params={"alpha": 0.01})
    code = gen.generate_class(cfg)

    # From physics hint
    cfg = gen.from_hint("2D incompressible Navier-Stokes with Re=100")
    print(cfg)

    # From domain name
    cfg = gen.from_domain("heat")

    # Full pipeline: hint -> code file
    gen.generate_to_file("2D wave equation",
                         output_path="wave_pinn.py")
"""

import ast
import importlib.util
import os
import re
import textwrap
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import json

try:
    import sympy as sp
    from sympy import symbols, Function, diff, Eq, latex
    _SYMPY = True
except ImportError:
    _SYMPY = False


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class EquationInfo:
    """Parsed information about a PDE/ODE."""
    raw:            str
    equation_type:  str          # "heat", "wave", "navier_stokes", "custom", …
    domain_class:   str          # physics, biology, chemistry, economics, …
    lhs_var:        str          # dependent variable being differentiated
    independent:    List[str]    # coordinates (t, x, y, z, …)
    dependent:      List[str]    # solution variables (u, v, p, …)
    parameters:     Dict[str, float]
    order:          int          # highest derivative order
    is_pde:         bool
    is_nonlinear:   bool
    description:    str = ""


@dataclass
class SimulationConfig:
    """Complete configuration for a generated PINN."""
    name:              str
    class_name:        str
    equation_info:     EquationInfo
    input_dim:         int
    output_dim:        int
    hidden_dim:        int        = 64
    num_layers:        int        = 4
    activation:        str        = "tanh"
    physics_loss_code: str        = ""
    boundary_code:     str        = ""
    domain_range:      Dict[str, Tuple[float, float]] = field(default_factory=dict)
    training_config:   Dict[str, Any] = field(default_factory=dict)
    dependencies:      List[str]  = field(default_factory=list)
    description:       str        = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["equation_info"] = asdict(self.equation_info)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ═══════════════════════════════════════════════════════════════
# Equation Pattern Database
# ═══════════════════════════════════════════════════════════════

EQUATION_PATTERNS = {
    # ── Heat / Diffusion ──────────────────────────────────────
    "heat": {
        "keywords": ["heat", "diffusion", "thermal", "conduction", "temperature",
                     "alpha.*d2u", "du/dt.*d2u"],
        "eq_hint": "du/dt = alpha * d2u/dx2",
        "domain_class": "physics",
        "independent": ["t", "x"],
        "dependent": ["u"],
        "params": {"alpha": 0.01},
        "loss_template": "heat_1d",
        "description": "1D heat / diffusion equation: du/dt = alpha * d2u/dx2",
    },
    "heat_2d": {
        "keywords": ["heat 2d", "2d heat", "2d diffusion", "thermal 2d"],
        "eq_hint": "du/dt = alpha*(d2u/dx2 + d2u/dy2)",
        "domain_class": "physics",
        "independent": ["t", "x", "y"],
        "dependent": ["u"],
        "params": {"alpha": 0.01},
        "loss_template": "heat_2d",
        "description": "2D heat equation",
    },
    # ── Wave ──────────────────────────────────────────────────
    "wave": {
        "keywords": ["wave", "vibration", "acoustic", "d2u/dt2", "c2.*d2u"],
        "eq_hint": "d2u/dt2 = c**2 * d2u/dx2",
        "domain_class": "physics",
        "independent": ["t", "x"],
        "dependent": ["u"],
        "params": {"c": 1.0},
        "loss_template": "wave_1d",
        "description": "1D wave equation: d2u/dt2 = c^2 * d2u/dx2",
    },
    # ── Navier-Stokes ─────────────────────────────────────────
    "navier_stokes": {
        "keywords": ["navier", "stokes", "navier-stokes", "incompressible",
                     "fluid", "reynolds", "viscous flow"],
        "eq_hint": "rho*(du/dt + u*du/dx + v*du/dy) = -dp/dx + mu*(d2u/dx2+d2u/dy2)",
        "domain_class": "physics",
        "independent": ["t", "x", "y"],
        "dependent": ["u", "v", "p"],
        "params": {"Re": 100.0, "mu": 0.01, "rho": 1.0},
        "loss_template": "navier_stokes_2d",
        "description": "2D incompressible Navier-Stokes",
    },
    # ── Burgers ───────────────────────────────────────────────
    "burgers": {
        "keywords": ["burgers", "viscous burgers"],
        "eq_hint": "du/dt + u*du/dx = nu * d2u/dx2",
        "domain_class": "physics",
        "independent": ["t", "x"],
        "dependent": ["u"],
        "params": {"nu": 0.01},
        "loss_template": "burgers_1d",
        "description": "Viscous Burgers equation",
    },
    # ── Poisson / Laplace ─────────────────────────────────────
    "poisson": {
        "keywords": ["poisson", "laplace", "electrostatic", "potential",
                     "d2u/dx2 + d2u/dy2"],
        "eq_hint": "d2u/dx2 + d2u/dy2 = f",
        "domain_class": "physics",
        "independent": ["x", "y"],
        "dependent": ["u"],
        "params": {"f": 1.0},
        "loss_template": "poisson_2d",
        "description": "2D Poisson equation (f=0 gives Laplace)",
    },
    # ── Schrodinger ───────────────────────────────────────────
    "schrodinger": {
        "keywords": ["schrodinger", "quantum", "wavefunction", "psi", "hbar",
                     "quantum mechanics"],
        "eq_hint": "i*hbar*dpsi/dt = -hbar^2/(2m)*d2psi/dx2 + V*psi",
        "domain_class": "quantum",
        "independent": ["t", "x"],
        "dependent": ["psi_r", "psi_i"],
        "params": {"hbar": 1.0, "m": 1.0},
        "loss_template": "schrodinger_1d",
        "description": "1D time-dependent Schrodinger equation (real/imag split)",
    },
    # ── Maxwell ───────────────────────────────────────────────
    "maxwell": {
        "keywords": ["maxwell", "electromagnetic", "electric field", "magnetic",
                     "em waves", "div E", "curl B"],
        "eq_hint": "dE/dt = curl(B)/mu0 - J/eps0; dB/dt = -curl(E)",
        "domain_class": "physics",
        "independent": ["t", "x", "y", "z"],
        "dependent": ["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
        "params": {"c": 3e8, "mu0": 1.257e-6, "eps0": 8.854e-12},
        "loss_template": "maxwell_3d",
        "description": "Experimental Vector Field Template (Vacuum EM)",
    },
    # ── Reaction-Diffusion ────────────────────────────────────
    "reaction_diffusion": {
        "keywords": ["reaction diffusion", "reaction_diffusion", "activator inhibitor", "turing",
                     "gray scott", "fitzhugh", "pattern formation"],
        "eq_hint": "du/dt = Du*laplacian(u) + f(u,v); dv/dt = Dv*laplacian(v) + g(u,v)",
        "domain_class": "biology",
        "independent": ["t", "x", "y"],
        "dependent": ["u", "v"],
        "params": {"Du": 0.16, "Dv": 0.08, "F": 0.035, "k": 0.065},
        "loss_template": "gray_scott_2d",
        "description": "Gray-Scott reaction-diffusion (Turing patterns)",
    },
    # ── Allen-Cahn ────────────────────────────────────────────
    "allen_cahn": {
        "keywords": ["allen cahn", "phase field", "interface", "crystal growth"],
        "eq_hint": "du/dt = epsilon^2*d2u/dx2 + u - u^3",
        "domain_class": "materials",
        "independent": ["t", "x"],
        "dependent": ["u"],
        "params": {"epsilon": 0.1},
        "loss_template": "allen_cahn_1d",
        "description": "Allen-Cahn phase field equation",
    },
    # ── Cahn-Hilliard ─────────────────────────────────────────
    "cahn_hilliard": {
        "keywords": ["cahn hilliard", "spinodal", "decomposition", "binary alloy"],
        "eq_hint": "du/dt = -laplacian(epsilon^2*laplacian(u) - f'(u))",
        "domain_class": "materials",
        "independent": ["t", "x", "y"],
        "dependent": ["u", "mu"],
        "params": {"epsilon": 0.01, "M": 1.0},
        "loss_template": "cahn_hilliard_2d",
        "description": "Cahn-Hilliard spinodal decomposition",
    },
    # ── KdV ───────────────────────────────────────────────────
    "kdv": {
        "keywords": ["kdv", "korteweg de vries", "soliton", "shallow water wave"],
        "eq_hint": "du/dt + 6*u*du/dx + d3u/dx3 = 0",
        "domain_class": "physics",
        "independent": ["t", "x"],
        "dependent": ["u"],
        "params": {},
        "loss_template": "kdv_1d",
        "description": "Korteweg-de Vries soliton equation",
    },
    # ── Klein-Gordon ──────────────────────────────────────────
    "klein_gordon": {
        "keywords": ["klein gordon", "relativistic", "scalar field", "mass field"],
        "eq_hint": "d2u/dt2 - c^2*d2u/dx2 + m^2*u = 0",
        "domain_class": "quantum",
        "independent": ["t", "x"],
        "dependent": ["u"],
        "params": {"c": 1.0, "m": 1.0},
        "loss_template": "klein_gordon_1d",
        "description": "Klein-Gordon equation (relativistic wave)",
    },
    # ── Elasticity / Euler-Bernoulli ──────────────────────────
    "elasticity": {
        "keywords": ["elasticity", "stress", "strain", "hooke", "solid mechanics",
                     "deformation", "young modulus"],
        "eq_hint": "d2u/dx2 + nu/(1-2nu)*(d2u/dx2+d2v/dxdy) = -(1+nu)/E * fx",
        "domain_class": "materials",
        "independent": ["x", "y"],
        "dependent": ["u", "v"],
        "params": {"E": 200e9, "nu": 0.3},
        "loss_template": "elasticity_2d",
        "description": "2D linear elasticity",
    },
    "beam": {
        "keywords": ["beam", "euler bernoulli", "bending", "deflection", "cantilever"],
        "eq_hint": "EI*d4w/dx4 = q(x)",
        "domain_class": "materials",
        "independent": ["x"],
        "dependent": ["w"],
        "params": {"EI": 1.0, "q": 0.0},
        "loss_template": "euler_bernoulli_beam",
        "description": "Euler-Bernoulli beam bending",
    },
    # ── Darcy Flow ────────────────────────────────────────────
    "darcy": {
        "keywords": ["darcy", "porous media", "groundwater", "permeability"],
        "eq_hint": "div(K*grad(p)) = f",
        "domain_class": "physics",
        "independent": ["x", "y"],
        "dependent": ["p"],
        "params": {"K": 1.0, "f_src": 0.0},
        "loss_template": "darcy_2d",
        "description": "Darcy flow through porous media",
    },
    # ── Advection-Diffusion ───────────────────────────────────
    "advection_diffusion": {
        "keywords": ["advection", "convection diffusion", "transport", "pollutant",
                     "tracer", "convection"],
        "eq_hint": "du/dt + v*du/dx = D*d2u/dx2",
        "domain_class": "physics",
        "independent": ["t", "x"],
        "dependent": ["u"],
        "params": {"v": 1.0, "D": 0.01},
        "loss_template": "advection_diffusion_1d",
        "description": "1D advection-diffusion (convection-diffusion)",
    },
    # ── Biology / Ecology ─────────────────────────────────────
    "sir": {
        "keywords": ["sir", "seir", "epidemic", "pandemic", "infection",
                     "susceptible", "infectious", "recovered"],
        "eq_hint": "dS/dt=-beta*S*I/N; dI/dt=beta*S*I/N-gamma*I; dR/dt=gamma*I",
        "domain_class": "biology",
        "independent": ["t"],
        "dependent": ["S", "I", "R"],
        "params": {"beta": 0.3, "gamma": 0.1, "N": 1000.0},
        "loss_template": "sir_ode",
        "description": "SIR epidemic ODE system",
    },
    "seir": {
        "keywords": ["seir", "latent period", "exposed", "incubation"],
        "eq_hint": "dS/dt=-b*S*I; dE/dt=b*S*I-sigma*E; dI/dt=sigma*E-gamma*I; dR/dt=gamma*I",
        "domain_class": "biology",
        "independent": ["t"],
        "dependent": ["S", "E", "I", "R"],
        "params": {"b": 0.3, "sigma": 0.2, "gamma": 0.1},
        "loss_template": "seir_ode",
        "description": "SEIR epidemic ODE with latent class",
    },
    "lotka_volterra": {
        "keywords": ["lotka volterra", "predator prey", "population dynamics",
                     "prey", "predator", "ecological"],
        "eq_hint": "dx/dt = alpha*x - beta*x*y; dy/dt = delta*x*y - gamma*y",
        "domain_class": "biology",
        "independent": ["t"],
        "dependent": ["x", "y"],
        "params": {"alpha": 1.0, "beta": 0.1, "delta": 0.075, "gamma": 1.5},
        "loss_template": "lotka_volterra_ode",
        "description": "Lotka-Volterra predator-prey",
    },
    "logistic": {
        "keywords": ["logistic", "verhulst", "population growth", "carrying capacity"],
        "eq_hint": "dN/dt = r*N*(1 - N/K)",
        "domain_class": "biology",
        "independent": ["t"],
        "dependent": ["N"],
        "params": {"r": 0.3, "K": 1000.0},
        "loss_template": "logistic_ode",
        "description": "Logistic population growth",
    },
    "cardinal_temperature": {
        "keywords": ["cardinal", "thermal performance", "temperature stress"],
        "eq_hint": "dT/dt = -k*(T - Topt)^2",
        "domain_class": "biology",
        "independent": ["t"],
        "dependent": ["T_perf"],
        "params": {"k": 0.01, "Topt": 25.0},
        "loss_template": "cardinal_ode",
        "description": "Cardinal temperature performance curve",
    },
    "stress": {
        "keywords": ["stress", "environmental stress", "tolerance"],
        "eq_hint": "dS/dt = alpha*S*(1 - S/S_max)",
        "domain_class": "biology",
        "independent": ["t"],
        "dependent": ["S"],
        "params": {"alpha": 0.1, "S_max": 1.0},
        "loss_template": "stress_ode",
        "description": "Environmental stress accumulation",
    },
    "biology": {
        "keywords": ["biology", "adaptive trait", "phenotype"],
        "eq_hint": "dA/dt = r*A - m*A^2",
        "domain_class": "biology",
        "independent": ["t"],
        "dependent": ["A"],
        "params": {"r": 0.5, "m": 0.1},
        "loss_template": "biology_ode",
        "description": "Generic biological adaptive trait expression",
    },
    # ── Neuroscience ──────────────────────────────────────────
    "hodgkin_huxley": {
        "keywords": ["hodgkin huxley", "neuron", "action potential", "membrane",
                     "sodium potassium", "ion channel"],
        "eq_hint": "C*dV/dt = I - gNa*m^3*h*(V-ENa) - gK*n^4*(V-EK) - gL*(V-EL)",
        "domain_class": "neuroscience",
        "independent": ["t"],
        "dependent": ["V", "m", "h", "n"],
        "params": {"C": 1.0, "gNa": 120.0, "gK": 36.0, "gL": 0.3,
                   "ENa": 50.0, "EK": -77.0, "EL": -54.4, "I_ext": 0.0},
        "loss_template": "hodgkin_huxley_ode",
        "description": "Hodgkin-Huxley neuron model",
    },
    "fitzhugh_nagumo": {
        "keywords": ["fitzhugh", "nagumo", "excitable", "nerve impulse"],
        "eq_hint": "dv/dt = v - v^3/3 - w + I; dw/dt = (v + a - b*w)/tau",
        "domain_class": "neuroscience",
        "independent": ["t"],
        "dependent": ["v", "w"],
        "params": {"a": 0.7, "b": 0.8, "tau": 12.5, "I": 0.5},
        "loss_template": "fitzhugh_nagumo_ode",
        "description": "FitzHugh-Nagumo excitable neuron (simplified HH)",
    },
    # ── Chemistry ─────────────────────────────────────────────
    "arrhenius": {
        "keywords": ["arrhenius", "reaction rate", "activation energy",
                     "chemical kinetics"],
        "eq_hint": "k = A * exp(-Ea/(R*T))",
        "domain_class": "chemistry",
        "independent": ["T"],
        "dependent": ["k"],
        "params": {"A": 1e13, "Ea": 50000.0, "R": 8.314},
        "loss_template": "arrhenius_ode",
        "description": "Arrhenius reaction rate",
    },
    "van_der_pol": {
        "keywords": ["van der pol", "oscillator", "nonlinear oscillator", "limit cycle"],
        "eq_hint": "d2x/dt2 - mu*(1-x^2)*dx/dt + x = 0",
        "domain_class": "physics",
        "independent": ["t"],
        "dependent": ["x"],
        "params": {"mu": 1.0},
        "loss_template": "van_der_pol_ode",
        "description": "Van der Pol nonlinear oscillator",
    },
    # ── Finance ───────────────────────────────────────────────
    "black_scholes": {
        "keywords": ["black scholes", "option pricing", "derivative", "finance",
                     "volatility"],
        "eq_hint": "dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0",
        "domain_class": "economics",
        "independent": ["t", "S"],
        "dependent": ["V"],
        "params": {"r": 0.05, "sigma": 0.2},
        "loss_template": "black_scholes_pde",
        "description": "Black-Scholes option pricing PDE",
    },
    # ── Fluid Dynamics Extra ──────────────────────────────────
    "euler_fluid": {
        "keywords": ["euler equation", "euler fluid", "inviscid", "compressible euler"],
        "eq_hint": "drho/dt+div(rho*u)=0; rho*(du/dt+u.grad(u))+grad(p)=0",
        "domain_class": "physics",
        "independent": ["t", "x", "y"],
        "dependent": ["rho", "u", "v", "p"],
        "params": {"gamma": 1.4},
        "loss_template": "euler_fluid_2d",
        "description": "2D compressible Euler equations",
    },
    "magnetohydrodynamics": {
        "keywords": ["mhd", "magnetohydrodynamics", "plasma", "magnetic fluid"],
        "eq_hint": "rho*(du/dt+u.grad(u)) = -grad(p) + J x B + nu*laplacian(u)",
        "domain_class": "physics",
        "independent": ["t", "x", "y", "z"],
        "dependent": ["u", "v", "w", "Bx", "By", "Bz", "p"],
        "params": {"mu0": 1.0, "nu": 0.01},
        "loss_template": "mhd_3d",
        "description": "3D MHD (plasma physics)",
    },
    # ── Miscellaneous (Added to match PINN Factory) ───────────
    "orbital": {
        "keywords": ["orbital", "astrodynamics", "gravity", "orbit"],
        "eq_hint": "d2r/dt2 + mu * r / |r|^3 = 0",
        "domain_class": "physics",
        "independent": ["t"],
        "dependent": ["rx", "ry"],
        "params": {"mu": 1.0},
        "loss_template": "orbital_ode",
        "description": "Orbital mechanics 2-body problem",
    },
    "solid_mechanics": {
        "keywords": ["solid mechanics", "elasticity", "structures", "alloys"],
        "eq_hint": "d2u/dx2 + nu/(1-2nu)*(d2u/dx2+d2v/dxdy) = 0",
        "domain_class": "materials",
        "independent": ["x", "y"],
        "dependent": ["u", "v"],
        "params": {"E": 200e9, "nu": 0.3},
        "loss_template": "solid_mechanics_2d",
        "description": "2D linear elasticity for solid mechanics",
    },
    "phase_change": {
        "keywords": ["phase change", "melting", "freezing", "stefan"],
        "eq_hint": "C_eff(T) dT/dt = k d2T/dx2",
        "domain_class": "physics",
        "independent": ["t", "x"],
        "dependent": ["T"],
        "params": {"L": 334.0, "Tm": 273.15},
        "loss_template": "phase_change_1d",
        "description": "1D Phase Change (Stefan Problem) Enthalpy formulation",
    },
    "radiation": {
        "keywords": ["radiation", "decay", "transport"],
        "eq_hint": "dI/dt + dI/dx + alpha * I = 0",
        "domain_class": "physics",
        "independent": ["t", "x"],
        "dependent": ["I"],
        "params": {"alpha": 1.0},
        "loss_template": "radiation_pde",
        "description": "Radiation transport/decay",
    },
    "economics": {
        "keywords": ["cost", "economics", "supply", "decay"],
        "eq_hint": "dC/dt + decay * C = 0",
        "domain_class": "economics",
        "independent": ["t"],
        "dependent": ["C"],
        "params": {"decay": 0.1},
        "loss_template": "economics_ode",
        "description": "Cost decay dynamics",
    },
}

# Keyword scoring: domain name -> list of keywords
DOMAIN_KEYWORDS = {
    "physics":      ["heat", "wave", "fluid", "stress", "strain", "vibration",
                     "diffusion", "potential", "temperature", "pressure"],
    "chemistry":    ["reaction", "arrhenius", "kinetics", "catalyst", "concentration",
                     "mole", "activation energy"],
    "biology":      ["growth", "population", "cell", "bacteria", "organism",
                     "mutation", "evolution", "ecology"],
    "materials":    ["material", "alloy", "crystal", "phase", "elastic",
                     "plastic", "fracture", "composite"],
    "quantum":      ["quantum", "schrodinger", "wavefunction", "electron",
                     "photon", "spin", "orbital"],
    "neuroscience": ["neuron", "action potential", "synapse", "neural",
                     "brain", "cortex", "membrane"],
    "economics":    ["price", "option", "market", "volatility", "interest",
                     "finance", "asset", "portfolio"],
}


# ═══════════════════════════════════════════════════════════════
# Physics Loss Code Templates
# ═══════════════════════════════════════════════════════════════

LOSS_TEMPLATES = {

"heat_1d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of du/dt = alpha * d2u/dx2"""
        x = x.clone().requires_grad_(True)
        u = self(x)
        grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                    create_graph=True)[0]
        u_t  = grads[:, 0:1]
        u_x  = grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0][:, 1:2]
        alpha = x[:, 2:3]
        residual = u_t - alpha * u_xx
        return (residual ** 2).mean()
''',

"heat_2d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of du/dt = alpha*(d2u/dx2 + d2u/dy2)"""
        x = x.clone().requires_grad_(True)
        u = self(x)
        g = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        u_t, u_x, u_y = g[:,0:1], g[:,1:2], g[:,2:3]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0][:,1:2]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y),
                                   create_graph=True)[0][:,2:3]
        residual = u_t - self.alpha * (u_xx + u_yy)
        return (residual ** 2).mean()
''',

"wave_1d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of d2u/dt2 = c^2 * d2u/dx2"""
        x = x.clone().requires_grad_(True)
        u  = self(x)
        g1 = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_t = g1[:, 0:1]
        u_x = g1[:, 1:2]
        g2t  = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t),
                                    create_graph=True)[0]
        u_tt = g2t[:, 0:1]
        g2x  = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0]
        u_xx = g2x[:, 1:2]
        residual = u_tt - self.c**2 * u_xx
        return (residual ** 2).mean()
''',

"burgers_1d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of du/dt + u*du/dx = nu*d2u/dx2"""
        x = x.clone().requires_grad_(True)
        u  = self(x)
        g  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_t, u_x = g[:,0:1], g[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0][:,1:2]
        residual = u_t + u * u_x - self.nu * u_xx
        return (residual ** 2).mean()
''',

"navier_stokes_2d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """2D Navier-Stokes residual (incompressible)"""
        x = x.clone().requires_grad_(True)
        out = self(x)
        u, v, p = out[:,0:1], out[:,1:2], out[:,2:3]
        def grad1(f):
            return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                                        create_graph=True)[0]
        gu, gv, gp = grad1(u), grad1(v), grad1(p)
        u_t,u_x,u_y = gu[:,0:1],gu[:,1:2],gu[:,2:3]
        v_t,v_x,v_y = gv[:,0:1],gv[:,1:2],gv[:,2:3]
        p_x, p_y    = gp[:,1:2], gp[:,2:3]
        u_xx = grad1(u_x)[:,1:2]; u_yy = grad1(u_y)[:,2:3]
        v_xx = grad1(v_x)[:,1:2]; v_yy = grad1(v_y)[:,2:3]
        re = 1.0 / self.Re
        r1 = u_t + u*u_x + v*u_y + p_x - re*(u_xx + u_yy)  # x-momentum
        r2 = v_t + u*v_x + v*v_y + p_y - re*(v_xx + v_yy)  # y-momentum
        r3 = u_x + v_y                                        # continuity
        return (r1**2 + r2**2 + r3**2).mean()
''',

"maxwell_3d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """3D Maxwell equations (vacuum, source-free):
        ∇·E = 0, ∇·B = 0,
        ∇×E = -∂B/∂t,
        ∇×B = μ0 ε0 ∂E/∂t
        This template computes curl/divergence via autograd on the network outputs.
        """
        x = x.clone().requires_grad_(True)
        out = self(x)
        Ex, Ey, Ez, Bx, By, Bz = (
            out[:,0:1], out[:,1:2], out[:,2:3], out[:,3:4], out[:,4:5], out[:,5:6]
        )

        def grad1(f):
            return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]

        gEx = grad1(Ex); gEy = grad1(Ey); gEz = grad1(Ez)
        gBx = grad1(Bx); gBy = grad1(By); gBz = grad1(Bz)

        # time derivatives (assumes input ordering: t,x,y,z)
        Ex_t = gEx[:,0:1]; Ey_t = gEy[:,0:1]; Ez_t = gEz[:,0:1]
        Bx_t = gBx[:,0:1]; By_t = gBy[:,0:1]; Bz_t = gBz[:,0:1]

        # spatial partials
        Ex_x, Ex_y, Ex_z = gEx[:,1:2], gEx[:,2:3], gEx[:,3:4]
        Ey_x, Ey_y, Ey_z = gEy[:,1:2], gEy[:,2:3], gEy[:,3:4]
        Ez_x, Ez_y, Ez_z = gEz[:,1:2], gEz[:,2:3], gEz[:,3:4]
        Bx_x, Bx_y, Bx_z = gBx[:,1:2], gBx[:,2:3], gBx[:,3:4]
        By_x, By_y, By_z = gBy[:,1:2], gBy[:,2:3], gBy[:,3:4]
        Bz_x, Bz_y, Bz_z = gBz[:,1:2], gBz[:,2:3], gBz[:,3:4]

        # curls
        curlE_x = Ez_y - Ey_z
        curlE_y = Ex_z - Ez_x
        curlE_z = Ey_x - Ex_y

        curlB_x = Bz_y - By_z
        curlB_y = Bx_z - Bz_x
        curlB_z = By_x - Bx_y

        # divergences
        divE = Ex_x + Ey_y + Ez_z
        divB = Bx_x + By_y + Bz_z

        mu0 = getattr(self, 'mu0', 1.0)
        eps0 = getattr(self, 'eps0', 1.0)

        # Residuals: ∇×E + ∂B/∂t = 0  -> curlE + B_t
        r1 = curlE_x + Bx_t
        r2 = curlE_y + By_t
        r3 = curlE_z + Bz_t

        # ∇×B - μ0 ε0 ∂E/∂t = 0
        r4 = curlB_x - mu0 * eps0 * Ex_t
        r5 = curlB_y - mu0 * eps0 * Ey_t
        r6 = curlB_z - mu0 * eps0 * Ez_t

        return (r1**2 + r2**2 + r3**2 + r4**2 + r5**2 + r6**2 + divE**2 + divB**2).mean()
''',

"elasticity_2d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder residual for 2D elasticity equations."""
        x = x.clone().requires_grad_(True)
        out = self(x)
        u, v = out[:,0:1], out[:,1:2]
        gu = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        gv = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        residual = gu[:,0:1] + gv[:,1:2] - self.E
        return (residual ** 2).mean()
''',

"solid_mechanics_2d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder residual for Solid Mechanics."""
        return torch.tensor(0.0, requires_grad=True)
''',

"phase_change_1d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder residual for Phase Change."""
        return torch.tensor(0.0, requires_grad=True)
''',

"euler_fluid_2d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder residual for 2D compressible Euler equations."""
        x = x.clone().requires_grad_(True)
        out = self(x)
        rho, u, v, p = out[:,0:1], out[:,1:2], out[:,2:3], out[:,3:4]
        g_rho = torch.autograd.grad(rho, x, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
        g_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        residual = g_rho[:,0:1] + g_u[:,1:2] + g_u[:,2:3] - self.gamma
        return (residual ** 2).mean()
''',

"cahn_hilliard_2d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder residual for the Cahn-Hilliard equation."""
        x = x.clone().requires_grad_(True)
        out = self(x)
        u, mu = out[:,0:1], out[:,1:2]
        g_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        lap_u = torch.autograd.grad(g_u[:,1:2], x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,1:2]
        residual = mu - (u**3 - u - self.epsilon**2 * lap_u)
        return (residual ** 2).mean()
''',

"arrhenius_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder residual for Arrhenius reaction rate."""
        x = x.clone().requires_grad_(True)
        k = self(x)
        # Ensure T is strictly positive to prevent divide-by-zero
        T = torch.abs(x[:, 0:1]) + 0.1
        A_param = x[:, 1:2]
        # Ensure Activation Energy is strictly positive
        Ea_param = torch.abs(x[:, 2:3]) + 0.1
        
        target = A_param * torch.exp(-Ea_param / (self.R * T))
        residual = k - target
        return (residual ** 2).mean()
''',

"mhd_3d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder residual for 3D MHD equations."""
        x = x.clone().requires_grad_(True)
        out = self(x)
        u, v, w, Bx, By, Bz, p = (
            out[:,0:1], out[:,1:2], out[:,2:3],
            out[:,3:4], out[:,4:5], out[:,5:6], out[:,6:7]
        )
        gu = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        residual = gu[:,1:2] + gu[:,2:3] - self.nu
        return (residual ** 2).mean()
''',

"poisson_2d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of d2u/dx2 + d2u/dy2 = f"""
        x = x.clone().requires_grad_(True)
        u  = self(x)
        g  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_x, u_y = g[:,0:1], g[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0][:,0:1]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y),
                                    create_graph=True)[0][:,1:2]
        residual = u_xx + u_yy - self.f
        return (residual ** 2).mean()
''',

"schrodinger_1d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Schrodinger: i*hbar*dpsi/dt = -hbar^2/(2m)*d2psi/dx2 + V*psi
        (real + imaginary parts split into psi_r, psi_i)"""
        x = x.clone().requires_grad_(True)
        out    = self(x)
        psi_r, psi_i = out[:,0:1], out[:,1:2]
        def grad1(f):
            return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                                        create_graph=True)[0]
        pr_t = grad1(psi_r)[:,0:1]; pr_x = grad1(psi_r)[:,1:2]
        pi_t = grad1(psi_i)[:,0:1]; pi_x = grad1(psi_i)[:,1:2]
        pr_xx = grad1(pr_x)[:,1:2]; pi_xx = grad1(pi_x)[:,1:2]
        coeff = self.hbar / (2 * self.m)
        r_real =  self.hbar * pi_t + coeff * pr_xx  # real part residual
        r_imag = -self.hbar * pr_t + coeff * pi_xx  # imag part residual
        return (r_real**2 + r_imag**2).mean()
''',

"logistic_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of dN/dt = r*N*(1 - N/K)"""
        x = x.clone().requires_grad_(True)
        N  = self(x)
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0][:,0:1]
        r_param = x[:, 1:2]
        K_param = x[:, 2:3]
        residual = dN - r_param * N * (1 - N / K_param)
        return (residual ** 2).mean()
''',

"lotka_volterra_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Lotka-Volterra ODE residuals"""
        x = x.clone().requires_grad_(True)
        out = self(x)
        prey, pred = out[:,0:1], out[:,1:2]
        def grad_t(f):
            return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                                        create_graph=True)[0][:,0:1]
        r1 = grad_t(prey)  - self.alpha*prey + self.beta*prey*pred
        r2 = grad_t(pred)  + self.gamma*pred - self.delta*prey*pred
        return (r1**2 + r2**2).mean()
''',

"sir_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """SIR epidemic ODE residuals"""
        x = x.clone().requires_grad_(True)
        out = self(x)
        S, I, R = out[:,0:1], out[:,1:2], out[:,2:3]
        def grad_t(f):
            return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                                        create_graph=True)[0][:,0:1]
        N = self.N
        r1 = grad_t(S) + self.beta * S * I / N
        r2 = grad_t(I) - self.beta * S * I / N + self.gamma * I
        r3 = grad_t(R) - self.gamma * I
        return (r1**2 + r2**2 + r3**2).mean()
''',

"seir_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """SEIR epidemic ODE residuals"""
        x = x.clone().requires_grad_(True)
        out = self(x)
        S, E, I, R = out[:,0:1], out[:,1:2], out[:,2:3], out[:,3:4]
        def grad_t(f):
            return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                                        create_graph=True)[0][:,0:1]
        r1 = grad_t(S) + self.b * S * I
        r2 = grad_t(E) - self.b * S * I + self.sigma * E
        r3 = grad_t(I) - self.sigma * E + self.gamma * I
        r4 = grad_t(R) - self.gamma * I
        return (r1**2 + r2**2 + r3**2 + r4**2).mean()
''',

"hodgkin_huxley_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Hodgkin-Huxley neuron ODE residuals"""
        x = x.clone().requires_grad_(True)
        out = self(x)
        V, m, h, n = out[:,0:1], out[:,1:2], out[:,2:3], out[:,3:4]
        def grad_t(f):
            return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                                        create_graph=True)[0][:,0:1]
        I_Na = self.gNa * m**3 * h * (V - self.ENa)
        I_K  = self.gK  * n**4      * (V - self.EK)
        I_L  = self.gL               * (V - self.EL)
        # alpha/beta gating functions (simplified)
        alpha_m = 0.1 * (V + 40) / (1 - torch.exp(-(V+40)/10) + 1e-7)
        beta_m  = 4 * torch.exp(-(V+65)/18)
        alpha_h = 0.07 * torch.exp(-(V+65)/20)
        beta_h  = 1 / (1 + torch.exp(-(V+35)/10))
        alpha_n = 0.01 * (V+55) / (1 - torch.exp(-(V+55)/10) + 1e-7)
        beta_n  = 0.125 * torch.exp(-(V+65)/80)
        r1 = self.C * grad_t(V) - (self.I_ext - I_Na - I_K - I_L)
        r2 = grad_t(m) - (alpha_m*(1-m) - beta_m*m)
        r3 = grad_t(h) - (alpha_h*(1-h) - beta_h*h)
        r4 = grad_t(n) - (alpha_n*(1-n) - beta_n*n)
        return (r1**2 + r2**2 + r3**2 + r4**2).mean()
''',

"black_scholes_pde": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of Black-Scholes PDE"""
        x = x.clone().requires_grad_(True)
        V  = self(x)
        g  = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V),
                                  create_graph=True)[0]
        V_t, V_S = g[:,0:1], g[:,1:2]
        V_SS = torch.autograd.grad(V_S, x, grad_outputs=torch.ones_like(V_S),
                                    create_graph=True)[0][:,1:2]
        S = x[:,1:2]
        residual = (V_t + 0.5*self.sigma**2 * S**2 * V_SS
                    + self.r * S * V_S - self.r * V)
        return (residual ** 2).mean()
''',

"orbital_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of orbital mechanics"""
        x = x.clone().requires_grad_(True)
        out = self(x)
        rx, ry = out[:,0:1], out[:,1:2]
        def G(f): return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0][:,0:1]
        vx, vy = G(rx), G(ry)
        ax, ay = G(vx), G(vy)
        r3 = (rx**2 + ry**2)**1.5 + 1e-6
        mu_param = self.mu
        if x.shape[1] > 1:
            mu_param = self.mu * (1.0 + 0.1 * torch.abs(x[:, 1:2]))
        r1 = ax + mu_param * rx / r3
        r2 = ay + mu_param * ry / r3
        return (r1**2 + r2**2).mean()
''',

"radiation_pde": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of radiation transport"""
        x = x.clone().requires_grad_(True)
        I = self(x)
        g = torch.autograd.grad(I, x, grad_outputs=torch.ones_like(I), create_graph=True)[0]
        I_t, I_x = g[:,0:1], g[:,1:2]
        alpha_param = self.alpha
        if x.shape[1] > 2:
            alpha_param = self.alpha * (1.0 + 0.5 * torch.abs(x[:, 2:3]))
        return ((I_t + I_x + alpha_param * I)**2).mean()
''',

"economics_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of economics decay"""
        x = x.clone().requires_grad_(True)
        C = self(x)
        dC = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(C), create_graph=True)[0][:,0:1]
        decay_param = self.decay
        if x.shape[1] > 1:
            decay_param = self.decay * (1.0 + 0.5 * torch.abs(x[:, 1:2]))
        return ((dC + decay_param * C)**2).mean()
''',


"allen_cahn_1d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of du/dt = eps^2*d2u/dx2 + u - u^3"""
        x = x.clone().requires_grad_(True)
        u  = self(x)
        g  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_t, u_x = g[:,0:1], g[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0][:,1:2]
        residual = u_t - self.epsilon**2 * u_xx - u + u**3
        return (residual ** 2).mean()
''',

"kdv_1d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of KdV: du/dt + 6u*du/dx + d3u/dx3 = 0"""
        x = x.clone().requires_grad_(True)
        u   = self(x)
        g1  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        u_t, u_x = g1[:,0:1], g1[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0][:,1:2]
        u_xxx= torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx),
                                    create_graph=True)[0][:,1:2]
        residual = u_t + 6 * u * u_x + u_xxx
        return (residual ** 2).mean()
''',

"advection_diffusion_1d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of du/dt + v*du/dx = D*d2u/dx2"""
        x = x.clone().requires_grad_(True)
        u  = self(x)
        g  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_t, u_x = g[:,0:1], g[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0][:,1:2]
        residual = u_t + self.v * u_x - self.D * u_xx
        return (residual ** 2).mean()
''',

"klein_gordon_1d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of d2u/dt2 - c^2*d2u/dx2 + m^2*u = 0"""
        x = x.clone().requires_grad_(True)
        u  = self(x)
        g1 = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_t, u_x = g1[:,0:1], g1[:,1:2]
        u_tt = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t),
                                    create_graph=True)[0][:,0:1]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0][:,1:2]
        residual = u_tt - self.c**2 * u_xx + self.m**2 * u
        return (residual ** 2).mean()
''',

"euler_bernoulli_beam": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of EI*d4w/dx4 = q"""
        x = x.clone().requires_grad_(True)
        w  = self(x)
        w_x   = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w),
                                     create_graph=True)[0]
        w_xx  = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x),
                                     create_graph=True)[0]
        w_xxx = torch.autograd.grad(w_xx, x, grad_outputs=torch.ones_like(w_xx),
                                     create_graph=True)[0]
        w_xxxx= torch.autograd.grad(w_xxx, x, grad_outputs=torch.ones_like(w_xxx),
                                     create_graph=True)[0]
        residual = self.EI * w_xxxx - self.q
        return (residual ** 2).mean()
''',

"van_der_pol_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of Van der Pol oscillator: d2x/dt2 - mu*(1-x^2)*dx/dt + x = 0"""
        x_in = x.clone().requires_grad_(True)
        out  = self(x_in)
        g1   = torch.autograd.grad(out, x_in, grad_outputs=torch.ones_like(out),
                                    create_graph=True)[0][:,0:1]
        g2   = torch.autograd.grad(g1, x_in, grad_outputs=torch.ones_like(g1),
                                    create_graph=True)[0][:,0:1]
        residual = g2 - self.mu*(1 - out**2)*g1 + out
        return (residual ** 2).mean()
''',

"fitzhugh_nagumo_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """FitzHugh-Nagumo ODE residuals"""
        x = x.clone().requires_grad_(True)
        out = self(x)
        v, w = out[:,0:1], out[:,1:2]
        def grad_t(f):
            return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                                        create_graph=True)[0][:,0:1]
        r1 = grad_t(v) - (v - v**3/3 - w + self.I)
        r2 = grad_t(w) - (v + self.a - self.b*w) / self.tau
        return (r1**2 + r2**2).mean()
''',

"gray_scott_2d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Gray-Scott reaction-diffusion residuals"""
        x = x.clone().requires_grad_(True)
        out = self(x)
        u, v = out[:,0:1], out[:,1:2]
        def laplacian_2d(f, xi):
            g = torch.autograd.grad(f, xi, grad_outputs=torch.ones_like(f),
                                     create_graph=True)[0]
            fx, fy = g[:,1:2], g[:,2:3]
            fxx = torch.autograd.grad(fx, xi, grad_outputs=torch.ones_like(fx),
                                       create_graph=True)[0][:,1:2]
            fyy = torch.autograd.grad(fy, xi, grad_outputs=torch.ones_like(fy),
                                       create_graph=True)[0][:,2:3]
            return fxx + fyy
        def grad_t(f):
            return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                                        create_graph=True)[0][:,0:1]
        F_param = x[:, 3:4]
        k_param = x[:, 4:5]
        r1 = grad_t(u) - self.Du*laplacian_2d(u,x) + u*v**2 - F_param*(1-u)
        r2 = grad_t(v) - self.Dv*laplacian_2d(v,x) - u*v**2 + (F_param+k_param)*v
        return (r1**2 + r2**2).mean()
''',

"darcy_2d": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual of -div(K*grad(p)) = f (Darcy)"""
        x = x.clone().requires_grad_(True)
        p  = self(x)
        g  = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                                  create_graph=True)[0]
        p_x, p_y = g[:,0:1], g[:,1:2]
        p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x),
                                    create_graph=True)[0][:,0:1]
        p_yy = torch.autograd.grad(p_y, x, grad_outputs=torch.ones_like(p_y),
                                    create_graph=True)[0][:,1:2]
        K_param = x[:, 2:3]
        f_param = x[:, 3:4]
        residual = -K_param * (p_xx + p_yy) - f_param
        return (residual ** 2).mean()
''',

"custom": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Custom physics residual — fill in your equation below."""
        x = x.clone().requires_grad_(True)
        u = self(x)
        # TODO: implement residual for your custom equation
        # Example skeleton (modify as needed):
        g = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        u_t = g[:, 0:1]   # first coordinate derivative
        # residual = u_t - ...  (your PDE/ODE rhs)
        residual = u_t * 0.0   # placeholder
        return (residual ** 2).mean()
''',
"cardinal_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        T_perf = self(x)
        dT = torch.autograd.grad(T_perf, x, grad_outputs=torch.ones_like(T_perf),
                                 create_graph=True)[0][:,0:1]
        residual = dT + self.k*(T_perf - self.Topt)**2
        return (residual ** 2).mean()
''',

"stress_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        S = self(x)
        dS = torch.autograd.grad(S, x, grad_outputs=torch.ones_like(S),
                                 create_graph=True)[0][:,0:1]
        alpha_param = x[:, 1:2]
        S_max_param = x[:, 2:3]
        residual = dS - alpha_param*S*(1 - S/S_max_param)
        return (residual ** 2).mean()
''',

"biology_ode": '''
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        A = self(x)
        dA = torch.autograd.grad(A, x, grad_outputs=torch.ones_like(A),
                                 create_graph=True)[0][:,0:1]
        r_param = x[:, 1:2]
        m_param = x[:, 2:3]
        residual = dA - (r_param*A - m_param*A**2)
        return (residual ** 2).mean()
''',

}


# ═══════════════════════════════════════════════════════════════
# Validation Utilities
# ═══════════════════════════════════════════════════════════════

def _is_valid_identifier(name: str) -> bool:
    return isinstance(name, str) and re.match(r'^[A-Za-z_]\w*$', name) is not None


def _validate_loss_templates(templates: Dict[str, str]) -> None:
    if not isinstance(templates, dict):
        raise TypeError("LOSS_TEMPLATES must be a dict of template_name -> code string")
    for key, code in templates.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("LOSS_TEMPLATES keys must be non-empty strings")
        if not isinstance(code, str):
            raise TypeError(f"LOSS_TEMPLATES[{key!r}] must be a string containing Python code")
        try:
            ast.parse(textwrap.dedent(code))
        except SyntaxError as exc:
            raise ValueError(
                f"LOSS_TEMPLATES[{key!r}] contains invalid Python syntax: {exc.msg}"
            ) from exc


class ValidationError(ValueError):
    """Raised for validation problems in registry entries.

    Subclasses `ValueError` so existing `pytest.raises(ValueError)` checks
    continue to work while allowing clearer exception identity.
    """


def _extract_self_attributes_from_code(code: str) -> set:
    """Return set of attribute names referenced as `self.xxx` in a template string."""
    if not isinstance(code, str):
        return set()
    return set(re.findall(r"\bself\.([A-Za-z_]\w*)\b", code))


def _validate_pattern_dict(key: str, pat: dict) -> None:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("EQUATION_PATTERNS keys must be non-empty strings")
    if not isinstance(pat, dict):
        raise TypeError(f"EQUATION_PATTERNS[{key!r}] must be a dict")

    required_fields = [
        "keywords", "eq_hint", "domain_class", "independent",
        "dependent", "params", "loss_template", "description",
    ]
    missing = [field for field in required_fields if field not in pat]
    if missing:
        raise ValueError(
            f"EQUATION_PATTERNS[{key!r}] missing required fields: {missing}"
        )

    keywords = pat["keywords"]
    if not isinstance(keywords, list) or not keywords:
        raise ValueError(f"EQUATION_PATTERNS[{key!r}]['keywords'] must be a non-empty list")
    if not all(isinstance(kw, str) and kw.strip() for kw in keywords):
        raise ValueError(
            f"EQUATION_PATTERNS[{key!r}]['keywords'] must be a list of non-empty strings"
        )

    if not isinstance(pat["eq_hint"], str) or not pat["eq_hint"].strip():
        raise ValueError(f"EQUATION_PATTERNS[{key!r}]['eq_hint'] must be a non-empty string")

    if not isinstance(pat["domain_class"], str) or not pat["domain_class"].strip():
        raise ValueError(f"EQUATION_PATTERNS[{key!r}]['domain_class'] must be a non-empty string")

    independent = pat["independent"]
    if not isinstance(independent, list) or not independent:
        raise ValueError(f"EQUATION_PATTERNS[{key!r}]['independent'] must be a non-empty list")
    for var in independent:
        if not _is_valid_identifier(var):
            raise ValueError(
                f"EQUATION_PATTERNS[{key!r}]['independent'] contains invalid variable name: {var!r}"
            )

    # independent / dependent overlap and duplication checks
    if len(independent) != len(set(independent)):
        raise ValidationError(f"EQUATION_PATTERNS[{key!r}]['independent'] contains duplicate variable names")

    dependent = pat["dependent"]
    if not isinstance(dependent, list) or not dependent:
        raise ValueError(f"EQUATION_PATTERNS[{key!r}]['dependent'] must be a non-empty list")
    for var in dependent:
        if not _is_valid_identifier(var):
            raise ValueError(
                f"EQUATION_PATTERNS[{key!r}]['dependent'] contains invalid variable name: {var!r}"
            )

    if len(dependent) != len(set(dependent)):
        raise ValidationError(f"EQUATION_PATTERNS[{key!r}]['dependent'] contains duplicate variable names")

    overlap = set(independent).intersection(set(dependent))
    if overlap:
        raise ValidationError(
            f"EQUATION_PATTERNS[{key!r}] has overlapping independent/dependent variables: {sorted(list(overlap))}"
        )

    params = pat["params"]
    if not isinstance(params, dict):
        raise TypeError(f"EQUATION_PATTERNS[{key!r}]['params'] must be a dict")
    for pname, pval in params.items():
        if not isinstance(pname, str) or not _is_valid_identifier(pname):
            raise ValueError(
                f"EQUATION_PATTERNS[{key!r}]['params'] contains invalid parameter name: {pname!r}"
            )
        if not isinstance(pval, (int, float)):
            raise ValueError(
                f"EQUATION_PATTERNS[{key!r}]['params'][{pname!r}] must be numeric"
            )

    loss_template = pat["loss_template"]
    if not isinstance(loss_template, str) or not loss_template.strip():
        raise ValueError(
            f"EQUATION_PATTERNS[{key!r}]['loss_template'] must be a non-empty string"
        )
    if loss_template != "custom" and loss_template not in LOSS_TEMPLATES:
        raise ValueError(
            f"EQUATION_PATTERNS[{key!r}]['loss_template'] references unknown template: {loss_template!r}"
        )

    if not isinstance(pat["description"], str):
        raise TypeError(f"EQUATION_PATTERNS[{key!r}]['description'] must be a string")

    # Ensure that the loss template's `self.<attr>` references are satisfied by
    # the pattern's `params` or by variables present in independent/dependent
    # lists or by known config attributes. This helps catch typos like
    # referencing `self.q` in a template while the pattern declares `EI` only.
    if loss_template != "custom":
        tmpl_code = LOSS_TEMPLATES.get(loss_template, "")
        self_attrs = _extract_self_attributes_from_code(tmpl_code)
        # Known attributes that may legitimately appear on the generated class
        simcfg_attrs = {f.name for f in __import__("dataclasses").fields(SimulationConfig)}
        eqinfo_attrs = {f.name for f in __import__("dataclasses").fields(EquationInfo)}
        allowed = set(params.keys()) | set(independent) | set(dependent) | simcfg_attrs | eqinfo_attrs
        missing = sorted([a for a in self_attrs if a not in allowed])
        if missing:
            raise ValidationError(
                f"EQUATION_PATTERNS[{key!r}] missing parameters referenced by loss template '{loss_template}': {missing}"
            )


def _validate_equation_patterns() -> None:
    if not isinstance(EQUATION_PATTERNS, dict):
        raise TypeError("EQUATION_PATTERNS must be a dict")
    for key, pat in EQUATION_PATTERNS.items():
        _validate_pattern_dict(key, pat)


# Validate registry at import time so pattern typos fail fast.
_validate_loss_templates(LOSS_TEMPLATES)
_validate_equation_patterns()


# ═══════════════════════════════════════════════════════════════
# Shared Utilities
# ═══════════════════════════════════════════════════════════════

def _indent(code: str, level: int = 1) -> str:
    if not code:
        return ""
    return textwrap.indent(textwrap.dedent(code).strip('\n'), ' ' * 4 * level)


def _normalize_class_name(name: str) -> str:
    if not name:
        return "GeneratedPINN"
    cleaned = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    if cleaned.lower().endswith('pinn'):
        cleaned = cleaned[:-4]
    parts = [p for p in re.split(r'[_\W]+', cleaned) if p]
    normalized = ''.join(
        p if p.isupper() else p[0].upper() + p[1:].lower()
        for p in parts
    )
    return (normalized or 'Generated') + 'PINN'


def _detect_dimensionality(text: str) -> int:
    if re.search(r'\b3\s*[-]?d\b|\bthree\s+dimensional\b', text):
        return 3
    if re.search(r'\b2\s*[-]?d\b|\btwo\s+dimensional\b', text):
        return 2
    if re.search(r'\b1\s*[-]?d\b|\bone\s+dimensional\b', text):
        return 1
    return 0


def _keyword_is_regex(keyword: str) -> bool:
    return bool(re.search(r'[\.\^\$\*\+\?\{\}\[\]\|\(\)]', keyword))


def _score_equation_pattern(text_lower: str, eq_type: str, pat: dict) -> Tuple[int, int, int, int]:
    score = 0
    matches = 0
    longest = 0
    for kw in pat.get('keywords', []):
        kw_lower = kw.lower()
        found = False
        if not _keyword_is_regex(kw_lower):
            if kw_lower in text_lower:
                score += 4
                found = True
        else:
            try:
                if re.search(kw_lower, text_lower):
                    score += 3
                    found = True
            except re.error:
                if kw_lower in text_lower:
                    score += 2
                    found = True
        if found:
            matches += 1
            longest = max(longest, len(kw_lower))
    if eq_type.replace('_', ' ') in text_lower or eq_type in text_lower:
        score += 2
    dim = _detect_dimensionality(text_lower)
    hints = [c for c in pat.get('independent', []) if c in ('x', 'y', 'z')]
    dim_bonus = 0
    if dim and hints:
        if len(hints) == dim:
            dim_bonus = 7 if dim > 1 else 3
        elif len(hints) >= dim:
            dim_bonus = 4
        else:
            dim_bonus = 1
        score += dim_bonus
    return score, dim_bonus, matches, longest


def _detect_nonlinearity(eq_str: str) -> bool:
    text = eq_str.replace('^', '**')
    if _SYMPY:
        try:
            symbols_found = set(re.findall(r'\b[A-Za-z_]\w*\b', text))
            local_symbols = {name: sp.symbols(name) for name in symbols_found}
            parsed = sp.sympify(text, locals=local_symbols)
            for node in sp.preorder_traversal(parsed):
                if isinstance(node, sp.Pow):
                    exponent = node.args[1]
                    if exponent.is_Number and exponent > 1:
                        return True
                if isinstance(node, sp.Mul):
                    non_number_factors = [arg for arg in node.args if not arg.is_Number]
                    if len(non_number_factors) > 1:
                        return True
            return False
        except Exception:
            pass
    if re.search(r'\b[A-Za-z_]\w*\s*\*\s*[A-Za-z_]\w*\b', eq_str):
        return True
    if re.search(r'\b[A-Za-z_]\w*\s*\^\s*[2-9]\b', eq_str):
        return True
    if re.search(r'\*\*[2-9]', eq_str):
        return True
    if re.search(r'\b[A-Za-z_]\w*_[A-Za-z_]\w*\s*\*\s*[A-Za-z_]\w*\b', eq_str):
        return True
    if re.search(r'\b[A-Za-z_]\w*\s*\*\s*[A-Za-z_]\w*_[A-Za-z_]\w*\b', eq_str):
        return True
    return False


def _validate_generated_code(code: str, filename: str = '<generated>') -> None:
    try:
        ast.parse(code, filename=filename)
        compile(code, filename, 'exec')
    except SyntaxError as exc:
        raise ValueError(f'Generated code has invalid Python syntax: {exc}') from exc


def _dynamic_import_and_instantiate(path: str, class_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(class_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not create import spec for {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f'Class {class_name} not found in generated module')
    return cls()


# ════════════════════════════════════════════════════════════════════

class SimulationGenerator:
    """
    Autonomous PINN code generator.

    Converts equations/hints into complete, runnable PINN classes.
    """

    # ── Equation type detection ──────────────────────────────

    def _detect_equation_type(self, text: str) -> Tuple[str, dict]:
        """
        Return (equation_type, pattern_dict) from free-form text or equation.
        Uses weighted keyword scoring, dimensional hints, and specificity tie-breakers.
        """
        text_lower = text.lower()
        best_type = "custom"
        best_score = (-1, -1, -1, -1, -1)

        for eq_type, pat in EQUATION_PATTERNS.items():
            score, dim_bonus, matches, longest = _score_equation_pattern(
                text_lower, eq_type, pat
            )
            specificity = len(pat.get("keywords", []))
            score_key = (score, dim_bonus, matches, longest, specificity)
            if score_key > best_score:
                best_score = score_key
                best_type = eq_type

        return best_type, EQUATION_PATTERNS.get(best_type, {})

    def _detect_domain_class(self, text: str, eq_type: str) -> str:
        """Infer physics domain class from text + equation type."""
        pat = EQUATION_PATTERNS.get(eq_type, {})
        if "domain_class" in pat:
            return pat["domain_class"]
        text_lower = text.lower()
        for domain, kws in DOMAIN_KEYWORDS.items():
            if any(kw in text_lower for kw in kws):
                return domain
        return "physics"

    def _parse_equation_string(self, eq_str: str) -> EquationInfo:
        """
        Extract variables and structure from equation string.
        Handles forms like:
          "du/dt = alpha * d2u/dx2"
          "d2u/dt2 - c^2 * d2u/dx2 = 0"
          "dS/dt = -beta*S*I/N"
        """
        eq_lower = eq_str.lower()

        # Extract dependent variables (left of d_/dt pattern)
        dep_vars = list(dict.fromkeys(
            re.findall(r'd([a-zA-Z_]\w*)/d', eq_str)
        ))
        if not dep_vars:
            # Try =rhs form: "u_t = ..."
            dep_vars = list(dict.fromkeys(
                re.findall(r'\b([a-zA-Z][a-zA-Z0-9]*)_[tx]\b', eq_str)
            ))
        if not dep_vars:
            dep_vars = ["u"]

        # Extract independent coordinates
        coords = list(dict.fromkeys(
            re.findall(r'/d([a-zA-Z]\w*)', eq_str)
        ))
        if not coords:
            coords = ["t", "x"]

        # Determine if PDE (has spatial derivative) or ODE
        has_space = any(c in coords for c in ["x", "y", "z", "r"])
        is_pde    = has_space and "t" in coords

        # Maximum derivative order
        orders = re.findall(r'd(\d+)[a-zA-Z_]', eq_str)
        order  = max((int(o) for o in orders), default=1)
        if "d2" in eq_lower or "xx" in eq_lower or "yy" in eq_lower:
            order = max(order, 2)

        # Nonlinearity detection uses symbolic awareness when available.
        is_nonlinear = _detect_nonlinearity(eq_str)

        eq_type, pat = self._detect_equation_type(eq_str)
        params = {**pat.get("params", {})}

        return EquationInfo(
            raw          = eq_str,
            equation_type= eq_type,
            domain_class = pat.get("domain_class", "physics"),
            lhs_var      = dep_vars[0] if dep_vars else "u",
            independent  = coords,
            dependent    = dep_vars,
            parameters   = params,
            order        = order,
            is_pde       = is_pde,
            is_nonlinear = is_nonlinear,
            description  = pat.get("description", eq_str[:120]),
        )

    # ── Public API ───────────────────────────────────────────

    def from_equation(
        self,
        equation: str,
        variables:  List[str] = None,
        coords:     List[str] = None,
        params:     Dict[str, float] = None,
        name:       str = None,
        hidden_dim: int = 64,
        num_layers: int = 4,
    ) -> SimulationConfig:
        """
        Build SimulationConfig from an equation string.

        Args:
            equation:   Math expression, e.g. "du/dt = alpha * d2u/dx2"
            variables:  Override detected dependent variables
            coords:     Override detected independent variables
            params:     Physical parameters (merged with detected defaults)
            name:       Class name (auto-generated if None)
            hidden_dim: Hidden layer width
            num_layers: Network depth
        """
        eq_info = self._parse_equation_string(equation)
        if variables:
            eq_info.dependent   = variables
        if coords:
            eq_info.independent = coords
        if params:
            eq_info.parameters.update(params)

        input_dim  = len(eq_info.independent)
        output_dim = len(eq_info.dependent)

        # Build class name
        if name is None:
            slug = re.sub(r'\W+', '_', eq_info.equation_type).strip('_')
            h    = hashlib.md5(equation.encode()).hexdigest()[:4]
            name = f"{slug.title()} PINN {h}"
        class_name = _normalize_class_name(name)

        # Select physics loss template
        pat = EQUATION_PATTERNS.get(eq_info.equation_type, {})
        if pat:
            _validate_pattern_dict(eq_info.equation_type, pat)
        tpl_key = pat.get("loss_template", "custom")
        if tpl_key != "custom" and tpl_key not in LOSS_TEMPLATES:
            raise ValueError(
                f"EQUATION_PATTERNS[{eq_info.equation_type!r}]['loss_template'] references unknown template: {tpl_key!r}"
            )
        phys_code = LOSS_TEMPLATES[tpl_key] if tpl_key in LOSS_TEMPLATES else LOSS_TEMPLATES["custom"]

        # Default domain ranges
        domain_range = {}
        for c in eq_info.independent:
            if c == "t":
                domain_range["t"] = (0.0, 1.0)
            elif c == "S":   # stock price
                domain_range["S"] = (1.0, 200.0)
            else:
                domain_range[c] = (0.0, 1.0)

        training_cfg = {
            "lr":            1e-3,
            "n_epochs":      5000,
            "n_collocation": 2048,
            "n_boundary":    512,
            "scheduler":     "cosine",
            "lambda_data":   1.0,
            "lambda_physics": 1.0,
            "lambda_boundary": 10.0,
        }

        return SimulationConfig(
            name              = name,
            class_name        = class_name,
            equation_info     = eq_info,
            input_dim         = input_dim,
            output_dim        = output_dim,
            hidden_dim        = hidden_dim,
            num_layers        = num_layers,
            physics_loss_code = phys_code,
            domain_range      = domain_range,
            training_config   = training_cfg,
            dependencies      = ["torch", "torch.nn", "torch.autograd"],
            description       = eq_info.description,
        )

    def _config_from_pattern(
        self,
        forced_type: str,
        pat:         dict,
        extra_params: Dict[str, float] = None,
        hidden_dim: int = 64,
        num_layers: int = 4,
        name: str = None,
    ) -> SimulationConfig:
        """
        Build a SimulationConfig directly from a known EQUATION_PATTERNS entry,
        bypassing re-detection so the equation_type is guaranteed correct.
        """
        _validate_pattern_dict(forced_type, pat)
        params = {**pat.get("params", {}), **(extra_params or {})}
        deps   = pat.get("dependent", ["u"])
        coords = pat.get("independent", ["t", "x"])

        input_dim  = len(coords)
        output_dim = len(deps)

        # Build class name
        slug = re.sub(r'\W+', '_', forced_type).strip('_')
        if name is None:
            name = f"{slug.title()} PINN"
        class_name = _normalize_class_name(name)

        tpl_key = pat.get("loss_template", "custom")
        if tpl_key != "custom" and tpl_key not in LOSS_TEMPLATES:
            raise ValueError(
                f"EQUATION_PATTERNS[{forced_type!r}]['loss_template'] references unknown template: {tpl_key!r}"
            )
        phys_code = LOSS_TEMPLATES[tpl_key] if tpl_key in LOSS_TEMPLATES else LOSS_TEMPLATES["custom"]

        domain_range = {}
        for c in coords:
            domain_range[c] = (0.0, 1.0) if c not in ("S",) else (1.0, 200.0)

        eq_info = EquationInfo(
            raw           = pat.get("eq_hint", ""),
            equation_type = forced_type,
            domain_class  = pat.get("domain_class", "physics"),
            lhs_var       = deps[0],
            independent   = coords,
            dependent     = deps,
            parameters    = params,
            order         = 2 if "d2" in pat.get("eq_hint", "") else 1,
            is_pde        = any(c in coords for c in ("x", "y", "z"))
                            and "t" in coords,
            is_nonlinear  = _detect_nonlinearity(pat.get("eq_hint", "")),
            description   = pat.get("description", forced_type),
        )

        training_cfg = {
            "lr": 1e-3, "n_epochs": 5000,
            "n_collocation": 2048, "n_boundary": 512,
        }

        return SimulationConfig(
            name              = name,
            class_name        = class_name,
            equation_info     = eq_info,
            input_dim         = input_dim,
            output_dim        = output_dim,
            hidden_dim        = hidden_dim,
            num_layers        = num_layers,
            physics_loss_code = phys_code,
            domain_range      = domain_range,
            training_config   = training_cfg,
            dependencies      = ["torch", "torch.nn"],
            description       = eq_info.description,
        )

    def from_hint(
        self,
        hint: str,
        hidden_dim: int = 64,
        num_layers: int = 4,
        **extra_params,
    ) -> SimulationConfig:
        """
        Build SimulationConfig from a free-text physics hint.

        Args:
            hint: e.g. "2D incompressible Navier-Stokes with Re=100"
        """
        eq_type, pat = self._detect_equation_type(hint)

        # Extract numeric params from hint ("Re=100", "alpha=0.01")
        found_params: Dict[str, float] = {}
        for m in re.finditer(r'(\w+)\s*=\s*([\d.e+-]+)', hint):
            try:
                found_params[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        found_params.update(extra_params)

        if eq_type != "custom" and pat:
            return self._config_from_pattern(
                forced_type  = eq_type,
                pat          = pat,
                extra_params = found_params,
                hidden_dim   = hidden_dim,
                num_layers   = num_layers,
            )

        # Fallback: parse the hint as an equation
        return self.from_equation(
            equation   = hint,
            params     = found_params,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
        )

    def from_domain(self, domain: str, **kwargs) -> SimulationConfig:
        """
        Build config from a known domain name (e.g. "heat", "navier_stokes").
        """
        key = domain.lower().replace(' ', '_').replace('-', '_')
        if key not in EQUATION_PATTERNS:
            # Partial match
            for k in EQUATION_PATTERNS:
                if key in k or k in key:
                    key = k
                    break
            else:
                key = "custom"

        pat = EQUATION_PATTERNS.get(key, {})
        extra_params = {k: v for k, v in kwargs.items()
                        if isinstance(v, (int, float))}
        non_param_kwargs = {k: v for k, v in kwargs.items()
                            if k not in extra_params}
        return self._config_from_pattern(
            forced_type  = key,
            pat          = pat,
            extra_params = extra_params,
            **{k: v for k, v in non_param_kwargs.items()
               if k in ("hidden_dim", "num_layers", "name")},
        )

    # ── Code generation ──────────────────────────────────────

    def generate_class(self, cfg: SimulationConfig) -> str:
        """
        Generate a complete Python PINN class file as a string.

        The output is import-ready and validated for Python syntax.
        """
        eq = cfg.equation_info
        params = cfg.equation_info.parameters or {}

        init_param_sig = ", ".join(
            f"{k}: float = {v!r}"
            for k, v in params.items()
        )
        if init_param_sig:
            init_param_sig = ", " + init_param_sig

        init_assign_lines = [f"self.{k} = {v!r}" for k, v in params.items()]
        if not init_assign_lines:
            init_assign_lines = ["pass  # no physical parameters"]

        activation_map = {
            "tanh":   "nn.Tanh()",
            "relu":   "nn.ReLU()",
            "silu":   "nn.SiLU()",
            "gelu":   "nn.GELU()",
            "sigmoid": "nn.Sigmoid()",
        }
        act_code = activation_map.get(cfg.activation, "nn.Tanh()")

        network_layers = [
            f"nn.Linear({cfg.input_dim}, {cfg.hidden_dim}),",
            f"{act_code},",
        ]
        for _ in range(cfg.num_layers):
            network_layers.extend([
                f"nn.Linear({cfg.hidden_dim}, {cfg.hidden_dim}),",
                f"{act_code},",
            ])
        network_layers.append(f"nn.Linear({cfg.hidden_dim}, {cfg.output_dim}),")
        network_body = "\n".join(network_layers)

        physics_loss_code = _indent(cfg.physics_loss_code, 1)

        lines = [
            '"""',
            f'{cfg.class_name} — AUTO-GENERATED by SimulationGenerator',
            '========================================================= ',
            f'Equation : {eq.raw}',
            f'Domain   : {eq.domain_class}',
            f'Type     : {eq.equation_type}',
            'Generated: AUTO',
            '"""',
            '',
            'import torch',
            'import torch.nn as nn',
            '',
            '',
            f'class {cfg.class_name}(nn.Module):',
            '    """',
            '    Physics-Informed Neural Network for:',
            f'      {eq.description}',
            '',
            f'    Input  : {eq.independent}  →  dim={cfg.input_dim}',
            f'    Output : {eq.dependent}    →  dim={cfg.output_dim}',
            '    """',
            '',
            f'    def __init__(self{init_param_sig}, **kwargs):',
            '        super().__init__()',
            *_indent('\n'.join(init_assign_lines), 2).splitlines(),
            '        self.network = nn.Sequential(',
            *_indent(network_body, 3).splitlines(),
            '        )',
            '        self._init_weights()',
            '',
            '    def _init_weights(self):',
            '        for m in self.modules():',
            '            if isinstance(m, nn.Linear):',
            '                nn.init.xavier_normal_(m.weight)',
            '                nn.init.zeros_(m.bias)',
            '',
            '    def forward(self, x: torch.Tensor) -> torch.Tensor:',
            '        return self.network(x)',
            '',
            physics_loss_code,
            '',
            '    def boundary_loss(self, x_bc: torch.Tensor,',
            '                      y_bc: torch.Tensor) -> torch.Tensor:',
            '        """L2 loss at boundary / initial conditions."""',
            '        pred = self(x_bc)',
            '        return ((pred - y_bc) ** 2).mean()',
            '',
            '    def total_loss(self, x_data, y_data,',
            '                   x_physics, x_boundary=None, y_boundary=None,',
            '                   lam_data=1.0, lam_phys=1.0, lam_bc=10.0):',
            '        L_data  = ((self(x_data) - y_data) ** 2).mean()',
            '        L_phys  = self.physics_loss(x_physics)',
            '        L_bc    = 0.0',
            '        if x_boundary is not None and y_boundary is not None:',
            '            L_bc = self.boundary_loss(x_boundary, y_boundary)',
            '        return lam_data*L_data + lam_phys*L_phys + lam_bc*L_bc',
            '',
            '',
            'def train_' + cfg.class_name.lower() + '(',
            f'    model: {cfg.class_name},',
            '    x_data:     torch.Tensor,',
            '    y_data:     torch.Tensor,',
            '    x_physics:  torch.Tensor,',
            '    x_bc:       torch.Tensor = None,',
            '    y_bc:       torch.Tensor = None,',
            f'    n_epochs:   int   = {cfg.training_config.get("n_epochs", 5000)},',
            f'    lr:         float = {cfg.training_config.get("lr", 1e-3)},',
            '    verbose:    bool  = True,',
            '):',
            '    """Standard PINN training loop."""',
            '    opt = torch.optim.Adam(model.parameters(), lr=lr)',
            '    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)',
            '    for epoch in range(1, n_epochs + 1):',
            '        opt.zero_grad()',
            '        loss = model.total_loss(x_data, y_data, x_physics, x_bc, y_bc)',
            '        loss.backward()',
            '        opt.step()',
            '        sched.step()',
            '        if verbose and epoch % (n_epochs // 10) == 0:',
            '            print(f"  Epoch {epoch:5d} / {n_epochs} | loss={loss.item():.6f}")',
            '    return model',
            '',
            'if __name__ == "__main__":',
            '    import torch',
            '    model = ' + cfg.class_name + '()',
            f'    x = torch.rand({cfg.input_dim})',
            '    print(f"Forward  : {model(x.unsqueeze(0)).shape}")',
            f'    xb = torch.rand(32, {cfg.input_dim})',
            '    print(f"Phys loss: {model.physics_loss(xb).item():.6f}")',
            '    print("PINN instantiated successfully.")',
        ]

        class_code = "\n".join(lines).strip() + "\n"
        _validate_generated_code(class_code)
        return class_code

    def generate_to_file(
        self,
        hint_or_equation: str,
        output_path: str,
        **kwargs,
    ) -> SimulationConfig:
        """
        Full pipeline: hint/equation -> PINN class -> saved .py file.

        Returns the SimulationConfig used.
        """
        cfg  = self.from_hint(hint_or_equation, **kwargs)
        code = self.generate_class(cfg)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"[SimulationGenerator] Saved {cfg.class_name} to {output_path}")
        return cfg

    def list_supported_domains(self) -> List[str]:
        """Return all built-in equation types."""
        return sorted(EQUATION_PATTERNS.keys())

    def describe_domain(self, domain: str) -> str:
        """Return description of a known domain."""
        pat = EQUATION_PATTERNS.get(domain, {})
        return pat.get("description", f"Unknown domain: {domain}")


# ═══════════════════════════════════════════════════════════════
# Quick smoke test
# ═══════════════════════════════════════════════════════════════

def _smoke_test():
    gen = SimulationGenerator()

    print("=" * 60)
    print("  SimulationGenerator Smoke Test")
    print("=" * 60)

    tests = [
        ("from_equation", "du/dt = alpha * d2u/dx2",            "heat"),
        ("from_hint",     "2D incompressible Navier-Stokes",     "navier_stokes"),
        ("from_domain",   "wave",                                "wave"),
        ("from_hint",     "predator prey ecology Lotka Volterra","lotka_volterra"),
        ("from_hint",     "Black-Scholes option pricing PDE",    "black_scholes"),
        ("from_domain",   "sir",                                  "sir"),
        ("from_hint",     "Hodgkin Huxley neuron action potential","hodgkin_huxley"),
        ("from_hint",     "Schrodinger quantum wavefunction",    "schrodinger"),
        ("from_domain",   "kdv",                                  "kdv"),
        ("from_hint",     "Allen-Cahn phase field crystal",      "allen_cahn"),
    ]

    all_ok = True
    for method, arg, expected_type in tests:
        try:
            if method == "from_equation":
                cfg = gen.from_equation(arg)
            elif method == "from_domain":
                cfg = gen.from_domain(arg)
            else:
                cfg = gen.from_hint(arg)

            got  = cfg.equation_info.equation_type
            ok   = got == expected_type
            all_ok = all_ok and ok
            print(f"  {'OK  ' if ok else 'FAIL'} {method}({arg[:40]!r}) "
                  f"-> {got} (exp: {expected_type})")
        except Exception as e:
            all_ok = False
            print(f"  FAIL {method}({arg[:40]!r}) -> ERROR: {e}")

    # Code generation test
    print("\n[Code Generation]")
    cfg  = gen.from_domain("heat")
    code = gen.generate_class(cfg)
    print(f"  Generated {cfg.class_name}: {len(code)} chars, "
          f"{code.count(chr(10))} lines")

    # to_json test
    js = cfg.to_json()
    print(f"  Config JSON: {len(js)} chars")

    # list domains
    domains = gen.list_supported_domains()
    print(f"\n  Supported domains ({len(domains)}): "
          + ", ".join(domains[:8]) + " ...")

    print("\n  All tests " + ("passed." if all_ok else "SOME FAILED."))
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()


def check_registry_integrity(instantiate_models: bool = False) -> dict:
    """Run integrity checks across `EQUATION_PATTERNS`.

    Checks performed:
    - Each pattern validates via `_validate_pattern_dict`.
    - Each referenced loss template compiles.
    - Optionally, generates class code and attempts to instantiate the model.

    Returns a mapping domain -> status message ('ok' or error string).
    """
    results = {}
    gen = SimulationGenerator()
    try:
        import torch
        torch_available = True
    except Exception:
        torch_available = False

    for key, pat in EQUATION_PATTERNS.items():
        try:
            _validate_pattern_dict(key, pat)
            tpl = pat.get('loss_template', 'custom')
            if tpl != 'custom' and tpl not in LOSS_TEMPLATES:
                raise ValidationError(f"EQUATION_PATTERNS[{key!r}] references missing loss template: {tpl!r}")

            # Ensure template compiles
            if tpl in LOSS_TEMPLATES:
                _validate_loss_templates({tpl: LOSS_TEMPLATES[tpl]})

            # Generate config and compiled class code
            cfg = gen.from_domain(key)
            code = gen.generate_class(cfg)

            if instantiate_models and torch_available:
                # Write to temporary file and try importing/instantiating
                import tempfile, importlib.util, os
                fd, path = tempfile.mkstemp(prefix=f"gen_{key}_", suffix='.py')
                os.close(fd)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(code)
                try:
                    # Use dynamic import helper to instantiate
                    inst = _dynamic_import_and_instantiate(path, cfg.class_name)
                    # Basic forward pass to detect obvious shape/autograd errors
                    import torch
                    x = torch.randn(1, cfg.input_dim)
                    _ = inst(x)
                finally:
                    try:
                        os.remove(path)
                    except Exception:
                        pass

            results[key] = 'ok'
        except Exception as exc:
            results[key] = f'ERROR: {exc}'

    return results
