"""
pinn_factory.py  —  Universal PINN Factory
==========================================
Creates, manages, and registers Physics-Informed Neural Networks
for ANY scientific domain.

Architecture map:
  Physics     → BasePINN (heat, wave, fluid, poisson, elasticity …)
  Biology     → GrowthPINN variants (20 models)
  Chemistry   → ChemistryPINN (Arrhenius, reaction-diffusion)
  Quantum     → SchrodingerPINN
  Finance     → BlackScholesPINN
  Neuroscience→ HodgkinHuxleyPINN
  Epidemiology→ SIRPINN, SEIRPINN
  Custom      → Auto-generated via SimulationGenerator

Capabilities:
  • create(domain, **params)     → instantiated PINN
  • build_from_config(cfg)       → PINN from SimulationConfig
  • build_from_hint(hint)        → PINN from text hint
  • build_from_equation(eq)      → PINN from equation string
  • register(name, cls)          → add custom PINN class
  • list_domains()      factory = PINNFactory()

    # Create by domain name
    model = factory.create("navier_stokes", Re=20          → all available domains
  • domain_info(name)            → description + input/output dims
  • batch_create(domain_list)    → dict of PINNs
  • from_checkpoint(path, domain)→ load saved model

Usage:
   0)

    # Create from equation
    model = factory.build_from_equation("du/dt = alpha*d2u/dx2")

    # Create from hint
    model = factory.build_from_hint("SIR epidemic model")

    # Use generated code class (dynamic)
    model = factory.build_from_hint("custom oscillator", dynamic=True)

    # All domains at once
    models = factory.batch_create(["heat", "wave", "sir", "burgers"])

    # Save / load
    factory.save(model, "heat", "heat_checkpoint.pt")
    model2 = factory.load("heat_checkpoint.pt", "heat")
"""

import os
import importlib
import inspect
import tempfile
import sys
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from training.simulation_generator import SimulationGenerator, SimulationConfig, EQUATION_PATTERNS


# ═══════════════════════════════════════════════════════════════
# Generic BasePINN (used for dynamically generated models)
# ═══════════════════════════════════════════════════════════════

class _GenericPINN(nn.Module):
    """
    Lightweight PINN base class for dynamically-generated models.
    Physics loss is injected at construction time.
    Ensures all physics parameters are set as attributes with sensible defaults.
    """
    # Default parameters for common equations - ensures critical params always exist
    _DEFAULTS = {
        # Hodgkin-Huxley neuron model
        "C": 1.0, "gNa": 120.0, "gK": 36.0, "gL": 0.3,
        "ENa": 50.0, "EK": -77.0, "EL": -54.4, "I_ext": 0.0,
        # Heat/Burgers
        "alpha": 0.01, "nu": 0.01, "mu": 0.01,
        # Wave
        "c": 1.0,
        # Poisson
        "f": 1.0,
        # Other common
        "Re": 100.0, "rho": 1.0, "hbar": 1.0, "m": 1.0,
    }

    def __init__(
        self,
        input_dim:    int,
        output_dim:   int,
        hidden_dim:   int   = 64,
        num_layers:   int   = 4,
        activation:   str   = "tanh",
        params:       Dict[str, float] = None,
        physics_fn:   Any   = None,   # callable(model, x) -> scalar
        **kwargs
    ):
        super().__init__()
        # Register physics parameters as model attributes
        # First apply defaults, then override with provided params
        merged_params = {**self._DEFAULTS}
        if params:
            merged_params.update(params)
        
        for k, v in merged_params.items():
            setattr(self, k, v)
        
        # Store params for later validation
        self._initialized_params = merged_params

        act_map = {
            "tanh":    nn.Tanh,
            "relu":    nn.ReLU,
            "silu":    nn.SiLU,
            "gelu":    nn.GELU,
            "sigmoid": nn.Sigmoid,
        }
        Act = act_map.get(activation, nn.Tanh)

        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), Act()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), Act()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self._physics_fn = physics_fn
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self._physics_fn is not None:
            return self._physics_fn(self, x)
        return torch.tensor(0.0)

    def boundary_loss(self, x_bc, y_bc) -> torch.Tensor:
        return ((self(x_bc) - y_bc) ** 2).mean()

    def total_loss(self, x_data, y_data, x_phys,
                   x_bc=None, y_bc=None,
                   lam_data=1.0, lam_phys=1.0, lam_bc=10.0) -> torch.Tensor:
        L = lam_data * ((self(x_data) - y_data)**2).mean()
        L = L + lam_phys * self.physics_loss(x_phys)
        if x_bc is not None and y_bc is not None:
            L = L + lam_bc * self.boundary_loss(x_bc, y_bc)
        return L

    def validate_physics_parameters(self) -> Tuple[bool, List[str]]:
        """
        Validate that all critical physics parameters are present.
        Returns (is_valid, list_of_missing_params)
        """
        missing = []
        
        # Check for equation-specific required parameters
        # (based on generated templates that use self.param)
        if hasattr(self, '_physics_fn') and self._physics_fn is not None:
            # For Hodgkin-Huxley
            if hasattr(self, 'I_ext') and hasattr(self, 'C'):
                required_hh = ['C', 'gNa', 'gK', 'gL', 'ENa', 'EK', 'EL', 'I_ext']
                for param in required_hh:
                    if not hasattr(self, param):
                        missing.append(param)
        
        return len(missing) == 0, missing


# ═══════════════════════════════════════════════════════════════
# Built-in PINN classes (static, best-practice implementations)
# ═══════════════════════════════════════════════════════════════

class HeatPINN(_GenericPINN):
    """1D heat equation: du/dt = alpha * d2u/dx2."""
    def __init__(self, alpha: float = 0.05, **kw):
        input_dim = kw.pop("input_dim", 2)
        super().__init__(input_dim=input_dim, output_dim=1, params={"alpha": alpha}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        u = self(x)
        g = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        u_t, u_x = g[:,0:1], g[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0][:,1:2]
        alpha = self.alpha
        if x.shape[1] > 2:
            alpha = self.alpha + 0.5 * torch.abs(x[:, 2:3])
        return ((u_t - alpha * u_xx)**2).mean()


class WavePINN(_GenericPINN):
    """1D wave equation: d2u/dt2 = c^2 * d2u/dx2."""
    def __init__(self, c: float = 1.0, **kw):
        input_dim = kw.pop("input_dim", 2)
        super().__init__(input_dim=input_dim, output_dim=1, params={"c": c}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        u  = self(x)
        g1 = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_t, u_x = g1[:,0:1], g1[:,1:2]
        u_tt = torch.autograd.grad(u_t, x, torch.ones_like(u_t),
                                   create_graph=True)[0][:,0:1]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                   create_graph=True)[0][:,1:2]
        return ((u_tt - self.c**2 * u_xx)**2).mean()


class BurgersPINN(_GenericPINN):
    """Viscous Burgers equation: du/dt + u*du/dx = nu*d2u/dx2."""
    def __init__(self, nu: float = 0.01, **kw):
        input_dim = kw.pop("input_dim", 2)
        super().__init__(input_dim=input_dim, output_dim=1, params={"nu": nu}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        u  = self(x)
        g  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_t, u_x = g[:,0:1], g[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                   create_graph=True)[0][:,1:2]
        return ((u_t + u*u_x - self.nu*u_xx)**2).mean()


class PoissonPINN(_GenericPINN):
    """2D Poisson equation: d2u/dx2 + d2u/dy2 = f."""
    def __init__(self, f: float = 1.0, **kw):
        input_dim = kw.pop("input_dim", 2)
        super().__init__(input_dim=input_dim, output_dim=1, params={"f": f}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        u  = self(x)
        g  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_x, u_y = g[:,0:1], g[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                   create_graph=True)[0][:,0:1]
        u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y),
                                   create_graph=True)[0][:,1:2]
        return ((u_xx + u_yy - self.f)**2).mean()


class NavierStokesPINN(_GenericPINN):
    """2D incompressible Navier-Stokes (u, v, p)."""
    def __init__(self, Re: float = 100.0, **kw):
        input_dim = kw.pop("input_dim", 3)
        super().__init__(input_dim=input_dim, output_dim=3, params={"Re": Re}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        out = self(x)
        u, v, p = out[:,0:1], out[:,1:2], out[:,2:3]
        def G(f):
            return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]
        gu, gv, gp = G(u), G(v), G(p)
        u_t,u_x,u_y = gu[:,0:1],gu[:,1:2],gu[:,2:3]
        v_t,v_x,v_y = gv[:,0:1],gv[:,1:2],gv[:,2:3]
        p_x, p_y    = gp[:,1:2], gp[:,2:3]
        u_xx=G(u_x)[:,1:2]; u_yy=G(u_y)[:,2:3]
        v_xx=G(v_x)[:,1:2]; v_yy=G(v_y)[:,2:3]
        re = 1.0 / self.Re
        r1 = u_t + u*u_x + v*u_y + p_x - re*(u_xx+u_yy)
        r2 = v_t + u*v_x + v*v_y + p_y - re*(v_xx+v_yy)
        r3 = u_x + v_y
        return (r1**2 + r2**2 + r3**2).mean()


class BlackScholesPINN(_GenericPINN):
    """Black-Scholes option pricing PDE."""
    def __init__(self, r: float = 0.05, sigma: float = 0.2, **kw):
        input_dim = kw.pop("input_dim", 2)
        super().__init__(input_dim=input_dim, output_dim=1,
                         params={"r": r, "sigma": sigma}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        V  = self(x)
        g  = torch.autograd.grad(V, x, torch.ones_like(V), create_graph=True)[0]
        V_t, V_S = g[:,0:1], g[:,1:2]
        V_SS = torch.autograd.grad(V_S, x, torch.ones_like(V_S),
                                   create_graph=True)[0][:,1:2]
        S = x[:,1:2]
        return ((V_t + 0.5*self.sigma**2*S**2*V_SS
                 + self.r*S*V_S - self.r*V)**2).mean()


class SIRPINN(_GenericPINN):
    """SIR epidemic ODE system."""
    def __init__(self, beta: float = 0.3, gamma: float = 0.1,
                 N: float = 1000.0, **kw):
        input_dim = kw.pop("input_dim", 1)
        super().__init__(input_dim=input_dim, output_dim=3,
                         params={"beta": beta, "gamma": gamma, "N": N}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        out = self(x)
        S, I, R = out[:,0:1], out[:,1:2], out[:,2:3]
        def dt(f):
            return torch.autograd.grad(f, x, torch.ones_like(f),
                                       create_graph=True)[0][:,0:1]
        r1 = dt(S) + self.beta*S*I/self.N
        r2 = dt(I) - self.beta*S*I/self.N + self.gamma*I
        r3 = dt(R) - self.gamma*I
        return (r1**2 + r2**2 + r3**2).mean()


class SEIRPINN(_GenericPINN):
    """SEIR epidemic ODE with latent compartment."""
    def __init__(self, b: float=0.3, sigma: float=0.2,
                 gamma: float=0.1, **kw):
        input_dim = kw.pop("input_dim", 1)
        super().__init__(input_dim=input_dim, output_dim=4,
                         params={"b": b, "sigma": sigma, "gamma": gamma}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        out = self(x)
        S, E, I, R = out[:,0:1], out[:,1:2], out[:,2:3], out[:,3:4]
        def dt(f):
            return torch.autograd.grad(f, x, torch.ones_like(f),
                                       create_graph=True)[0][:,0:1]
        r1 = dt(S) + self.b*S*I
        r2 = dt(E) - self.b*S*I + self.sigma*E
        r3 = dt(I) - self.sigma*E + self.gamma*I
        r4 = dt(R) - self.gamma*I
        return (r1**2+r2**2+r3**2+r4**2).mean()


class SchrodingerPINN(_GenericPINN):
    """1D time-dependent Schrodinger equation (real+imag split)."""
    def __init__(self, hbar: float = 1.0, m: float = 1.0, **kw):
        input_dim = kw.pop("input_dim", 2)
        super().__init__(input_dim=input_dim, output_dim=2,
                         params={"hbar": hbar, "m": m}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        out = self(x)
        pr, pi = out[:,0:1], out[:,1:2]
        def G(f):
            return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]
        pr_t=G(pr)[:,0:1]; pr_x=G(pr)[:,1:2]
        pi_t=G(pi)[:,0:1]; pi_x=G(pi)[:,1:2]
        pr_xx=G(pr_x)[:,1:2]; pi_xx=G(pi_x)[:,1:2]
        c = self.hbar/(2*self.m)
        r1 = self.hbar*pi_t + c*pr_xx
        r2 = -self.hbar*pr_t + c*pi_xx
        return (r1**2+r2**2).mean()


class AdvDiffPINN(_GenericPINN):
    """1D advection-diffusion: du/dt + v*du/dx = D*d2u/dx2."""
    def __init__(self, v: float = 1.0, D: float = 0.01, **kw):
        input_dim = kw.pop("input_dim", 2)
        super().__init__(input_dim=input_dim, output_dim=1,
                         params={"v": v, "D": D}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        u  = self(x)
        g  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_t, u_x = g[:,0:1], g[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                   create_graph=True)[0][:,1:2]
        return ((u_t + self.v*u_x - self.D*u_xx)**2).mean()


class AllenCahnPINN(_GenericPINN):
    """Allen-Cahn phase-field: du/dt = eps^2*d2u/dx2 + u - u^3."""
    def __init__(self, epsilon: float = 0.1, **kw):
        input_dim = kw.pop("input_dim", 2)
        super().__init__(input_dim=input_dim, output_dim=1,
                         params={"epsilon": epsilon}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        u  = self(x)
        g  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_t, u_x = g[:,0:1], g[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                   create_graph=True)[0][:,1:2]
        return ((u_t - self.epsilon**2*u_xx - u + u**3)**2).mean()


class KdVPINN(_GenericPINN):
    """Korteweg-de Vries soliton: du/dt + 6u*du/dx + d3u/dx3 = 0."""
    def __init__(self, **kw):
        input_dim = kw.pop("input_dim", 2)
        super().__init__(input_dim=input_dim, output_dim=1, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        u   = self(x)
        g1  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_t, u_x = g1[:,0:1], g1[:,1:2]
        u_xx  = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                    create_graph=True)[0][:,1:2]
        u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx),
                                    create_graph=True)[0][:,1:2]
        return ((u_t + 6*u*u_x + u_xxx)**2).mean()


class LotkaVolterraPINN(_GenericPINN):
    """Lotka-Volterra predator-prey ODE."""
    def __init__(self, alpha:float=1.0, beta:float=0.1,
                 delta:float=0.075, gamma:float=1.5, **kw):
        input_dim = kw.pop("input_dim", 1)
        super().__init__(input_dim=input_dim, output_dim=2,
                         params={"alpha":alpha,"beta":beta,
                                 "delta":delta,"gamma":gamma}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        out  = self(x)
        prey, pred = out[:,0:1], out[:,1:2]
        def dt(f):
            return torch.autograd.grad(f, x, torch.ones_like(f),
                                       create_graph=True)[0][:,0:1]
        r1 = dt(prey) - self.alpha*prey + self.beta*prey*pred
        r2 = dt(pred) + self.gamma*pred - self.delta*prey*pred
        return (r1**2+r2**2).mean()


class FitzHughNagumoPINN(_GenericPINN):
    """FitzHugh-Nagumo excitable neuron."""
    def __init__(self, a:float=0.7, b:float=0.8,
                 tau:float=12.5, I:float=0.5, **kw):
        input_dim = kw.pop("input_dim", 1)
        super().__init__(input_dim=input_dim, output_dim=2,
                         params={"a":a,"b":b,"tau":tau,"I":I}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        out = self(x)
        v, w = out[:,0:1], out[:,1:2]
        def dt(f):
            return torch.autograd.grad(f, x, torch.ones_like(f),
                                       create_graph=True)[0][:,0:1]
        r1 = dt(v) - (v - v**3/3 - w + self.I)
        r2 = dt(w) - (v + self.a - self.b*w)/self.tau
        return (r1**2+r2**2).mean()


class EulerBernoulliBeamPINN(_GenericPINN):
    """Euler-Bernoulli beam bending: EI*d4w/dx4 = q."""
    def __init__(self, EI:float=1.0, q:float=1.0, **kw):
        input_dim = kw.pop("input_dim", 1)
        super().__init__(input_dim=input_dim, output_dim=1,
                         params={"EI":EI,"q":q}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        w  = self(x)
        w1 = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0]
        w2 = torch.autograd.grad(w1, x, torch.ones_like(w1), create_graph=True)[0]
        w3 = torch.autograd.grad(w2, x, torch.ones_like(w2), create_graph=True)[0]
        w4 = torch.autograd.grad(w3, x, torch.ones_like(w3), create_graph=True)[0]
        return ((self.EI*w4 - self.q)**2).mean()


class DarcyPINN(_GenericPINN):
    """Darcy flow through porous media: -div(K*grad(p)) = f."""
    def __init__(self, K:float=1.0, f_src:float=1.0, **kw):
        input_dim = kw.pop("input_dim", 2)
        super().__init__(input_dim=input_dim, output_dim=1,
                         params={"K":K,"f_src":f_src}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        p  = self(x)
        g  = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
        p_x, p_y = g[:,0:1], g[:,1:2]
        p_xx = torch.autograd.grad(p_x, x, torch.ones_like(p_x),
                                   create_graph=True)[0][:,0:1]
        p_yy = torch.autograd.grad(p_y, x, torch.ones_like(p_y),
                                   create_graph=True)[0][:,1:2]
        K = self.K
        if x.shape[1] > 2:
            K = self.K + 0.8 * torch.abs(x[:, 2:3])
        return ((-K*(p_xx+p_yy) - self.f_src)**2).mean()


class VanDerPolPINN(_GenericPINN):
    """Van der Pol oscillator: d2x/dt2 - mu*(1-x^2)*dx/dt + x = 0."""
    def __init__(self, mu:float=1.0, **kw):
        input_dim = kw.pop("input_dim", 1)
        super().__init__(input_dim=input_dim, output_dim=1,
                         params={"mu":mu}, **kw)
    def physics_loss(self, x_in):
        x_in = x_in.clone().requires_grad_(True)
        out  = self(x_in)
        g1   = torch.autograd.grad(out, x_in, torch.ones_like(out),
                                   create_graph=True)[0][:,0:1]
        g2   = torch.autograd.grad(g1, x_in, torch.ones_like(g1),
                                   create_graph=True)[0][:,0:1]
        return ((g2 - self.mu*(1-out**2)*g1 + out)**2).mean()


class LogisticPINN(_GenericPINN):
    """Logistic population growth ODE."""
    def __init__(self, r:float=0.3, K:float=1000.0, **kw):
        input_dim = kw.pop("input_dim", 1)
        super().__init__(input_dim=input_dim, output_dim=1,
                         params={"r":r,"K":K}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N  = self(x)
        dN = torch.autograd.grad(N, x, torch.ones_like(N),
                                 create_graph=True)[0][:,0:1]
        r = self.r
        if x.shape[1] > 1:
            r = self.r + 3.0 * torch.abs(x[:, 1:2])
        return ((dN - r*N*(1 - N/self.K))**2).mean()


class CardinalTemperaturePINN(_GenericPINN):
    """Cardinal temperature performance curve ODE."""
    def __init__(self, k:float=0.01, Topt:float=25.0, **kw):
        input_dim = kw.pop("input_dim", 1)
        super().__init__(input_dim=input_dim, output_dim=1,
                         params={"k":k,"Topt":Topt}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        T_perf  = self(x)
        dT = torch.autograd.grad(T_perf, x, torch.ones_like(T_perf),
                                 create_graph=True)[0][:,0:1]
        return ((dT + self.k*(T_perf - self.Topt)**2)**2).mean()


class StressPINN(_GenericPINN):
    """Environmental stress accumulation ODE."""
    def __init__(self, alpha:float=0.1, S_max:float=1.0, **kw):
        input_dim = kw.pop("input_dim", 1)
        super().__init__(input_dim=input_dim, output_dim=1,
                         params={"alpha":alpha,"S_max":S_max}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        S  = self(x)
        dS = torch.autograd.grad(S, x, torch.ones_like(S),
                                 create_graph=True)[0][:,0:1]
        alpha = self.alpha
        if x.shape[1] > 1:
            alpha = self.alpha + 1.5 * torch.abs(x[:, 1:2])
        return ((dS - alpha*S*(1 - S/self.S_max))**2).mean()


class BiologyPINN(_GenericPINN):
    """Generic biological adaptive trait expression ODE."""
    def __init__(self, r:float=0.5, m:float=0.1, **kw):
        input_dim = kw.pop("input_dim", 1)
        super().__init__(input_dim=input_dim, output_dim=1,
                         params={"r":r,"m":m}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        A  = self(x)
        dA = torch.autograd.grad(A, x, torch.ones_like(A),
                                 create_graph=True)[0][:,0:1]
        r = self.r
        if x.shape[1] > 1:
            r = self.r + 3.0 * torch.abs(x[:, 1:2])
        return ((dA - (r*A - self.m*A**2))**2).mean()



class ArrheniusPINN(_GenericPINN):
    """Normalized Arrhenius Equation: k_factor = exp(Ea/R * (1/T_ref - 1/T))"""
    def __init__(self, Ea:float=50000.0, R:float=8.314, T_ref:float=298.15, **kw):
        input_dim = kw.pop("input_dim", 1)
        super().__init__(input_dim=input_dim, output_dim=1,
                         params={"Ea":Ea,"R":R,"T_ref":T_ref}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        k_factor = self(x)  # predicted normalized reaction rate [0 to 1+]
        T = x[:,0:1]
        
        # Map generic input T to realistic absolute temperatures (273K - 373K)
        T_abs = T * 50.0 + 298.15
        
        Ea = self.Ea
        if x.shape[1] > 1:
            Ea = self.Ea * (1.0 - 0.8 * torch.abs(x[:, 1:2]))
            
        # Normalized Arrhenius (1.0 at T_ref)
        # k_factor = exp( (Ea/R) * (1/T_ref - 1/T_abs) )
        exponent = (Ea / self.R) * (1.0/self.T_ref - 1.0/T_abs)
        # Clamp exponent to prevent infinity during bad gradients
        exponent = torch.clamp(exponent, -20.0, 5.0)
        
        k_pred = torch.exp(exponent)
        return ((k_factor - k_pred)**2).mean()


class GrayScottPINN(_GenericPINN):
    """Gray-Scott reaction-diffusion pattern formation."""
    def __init__(self, Du:float=0.16, Dv:float=0.08,
                 F:float=0.035, k:float=0.065, **kw):
        input_dim = kw.pop("input_dim", 3)
        super().__init__(input_dim=input_dim, output_dim=2,
                         params={"Du":Du,"Dv":Dv,"F":F,"k":k}, **kw)
    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        out = self(x)
        u, v = out[:,0:1], out[:,1:2]
        def G(f): return torch.autograd.grad(f, x, torch.ones_like(f),
                                              create_graph=True)[0]
        def lap(f, xi):
            g  = G(f); fx, fy = g[:,1:2], g[:,2:3]
            return (G(fx)[:,1:2] + G(fy)[:,2:3])
        def dt(f): return G(f)[:,0:1]
        
        F_param = self.F
        k_param = self.k
        if x.shape[1] > 3:
            F_param = self.F + 0.05 * torch.abs(x[:, 3:4])
            k_param = self.k + 0.05 * torch.abs(x[:, 3:4])
            
        r1 = dt(u) - self.Du*lap(u,x) + u*v**2 - F_param*(1-u)
        r2 = dt(v) - self.Dv*lap(v,x) - u*v**2 + (F_param+k_param)*v
        return (r1**2+r2**2).mean()


# ═══════════════════════════════════════════════════════════════
# Domain → PINN class + default kwargs registry
# ═══════════════════════════════════════════════════════════════

_BUILTIN_REGISTRY: Dict[str, Tuple[Type[nn.Module], Dict]] = {
    # ── Physics ────────────────────────────────────────────────
    "heat":              (HeatPINN,             {"alpha": 0.01}),
    "wave":              (WavePINN,             {"c": 1.0}),
    "burgers":           (BurgersPINN,          {"nu": 0.01}),
    "poisson":           (PoissonPINN,          {"f": 1.0}),
    "navier_stokes":     (NavierStokesPINN,     {"Re": 100.0}),
    "advection_diffusion":(AdvDiffPINN,         {"v": 1.0, "D": 0.01}),
    "allen_cahn":        (AllenCahnPINN,        {"epsilon": 0.1}),
    "kdv":               (KdVPINN,              {}),
    "darcy":             (DarcyPINN,            {"K": 1.0, "f_src": 1.0}),
    "euler_bernoulli_beam":(EulerBernoulliBeamPINN,{"EI":1.0,"q":1.0}),
    "beam":              (EulerBernoulliBeamPINN,{"EI":1.0,"q":1.0}),
    "van_der_pol":       (VanDerPolPINN,        {"mu": 1.0}),
    # ── Quantum / Electromagnetism ─────────────────────────────
    "schrodinger":       (SchrodingerPINN,      {"hbar": 1.0, "m": 1.0}),
    "klein_gordon":      (WavePINN,             {"c": 1.0}),        # same structure
    # ── Finance ────────────────────────────────────────────────
    "black_scholes":     (BlackScholesPINN,     {"r": 0.05, "sigma": 0.2}),
    # ── Biology / Epidemiology ─────────────────────────────────
    "sir":               (SIRPINN,              {"beta":0.3,"gamma":0.1,"N":1000.0}),
    "seir":              (SEIRPINN,             {"b":0.3,"sigma":0.2,"gamma":0.1}),
    "lotka_volterra":    (LotkaVolterraPINN,    {"alpha":1.0,"beta":0.1,
                                                  "delta":0.075,"gamma":1.5}),
    "logistic":          (LogisticPINN,         {"r":0.3,"K":1000.0}),
    "cardinal_temperature":(CardinalTemperaturePINN,{"k":0.01,"Topt":25.0}),
    "stress":            (StressPINN,           {"alpha":0.1,"S_max":1.0}),
    "biology":           (BiologyPINN,          {"r":0.5,"m":0.1}),
    # ── Chemistry ──────────────────────────────────────────────
    "arrhenius":         (ArrheniusPINN,        {"Ea":50000.0,"R":8.314,"T_ref":298.15}),
    "reaction_diffusion":(GrayScottPINN,        {"Du":0.16,"Dv":0.08,
                                                  "F":0.035,"k":0.065}),
    "gray_scott":        (GrayScottPINN,        {"Du":0.16,"Dv":0.08,
                                                  "F":0.035,"k":0.065}),
    # ── Neuroscience ───────────────────────────────────────────
    "fitzhugh_nagumo":   (FitzHughNagumoPINN,   {"a":0.7,"b":0.8,"tau":12.5,"I":0.5}),
    # ── Growth (delegated to growth_models.py) ─────────────────
    # "monod", "gompertz", etc. are handled via growth_models fallback
}


# ═══════════════════════════════════════════════════════════════
# PINNFactory
# ═══════════════════════════════════════════════════════════════

class PINNFactory:
    """
    Universal Physics-Informed Neural Network factory.

    Supports 30+ physics/biology/chemistry/finance/quantum domains.
    Dynamically generates code for unknown domains via SimulationGenerator.
    """

    def __init__(self):
        # Writable copy of the builtin registry
        self._registry: Dict[str, Tuple[Type[nn.Module], Dict]] = dict(_BUILTIN_REGISTRY)
        self._generator = SimulationGenerator()
        self._dynamic_cache: Dict[str, Type[nn.Module]] = {}

        # Pull growth model PINNs from growth_models.py if available
        try:
            from growth_models import GROWTH_MODEL_REGISTRY
            for name, cls in GROWTH_MODEL_REGISTRY.items():
                if name not in self._registry:
                    self._registry[name] = (cls, {})
        except ImportError:
            pass

        # Pull existing project PINNs from physics_models.py if available
        try:
            import physics_models as pm
            for attr in dir(pm):
                obj = getattr(pm, attr)
                if (inspect.isclass(obj) and issubclass(obj, nn.Module)
                        and obj is not nn.Module):
                    key = attr.lower().replace("pinn", "").strip("_")
                    if key and key not in self._registry:
                        self._registry[key] = (obj, {})
        except ImportError:
            pass

    # ── Core creation ────────────────────────────────────────

    def create(
        self,
        domain: str,
        hidden_dim: int  = 64,
        num_layers: int  = 4,
        dynamic:    bool = False,
        **params,
    ) -> nn.Module:
        """
        Instantiate a PINN for the given domain.

        Args:
            domain:     Physics domain name (e.g. "heat", "navier_stokes")
            hidden_dim: Network width
            num_layers: Network depth
            dynamic:    If True and domain unknown, auto-generate with SimGen
            **params:   Physical parameters (override defaults)
        """
        key = domain.lower().replace(' ', '_').replace('-', '_')
        if key in self._registry:
            cls, defaults = self._registry[key]
            kw = {**defaults, **params}
            # Only pass recognised init parameters
            sig = inspect.signature(cls.__init__).parameters
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
            
            if "hidden_dim" in sig or has_kwargs:
                kw["hidden_dim"] = hidden_dim
            if "num_layers" in sig or has_kwargs:
                kw["num_layers"] = num_layers
                
            if has_kwargs:
                model = cls(**kw)
            else:
                model = cls(**{k: v for k, v in kw.items() if k in sig})
                
            from training.simulation_generator import EQUATION_PATTERNS
            if key in EQUATION_PATTERNS:
                model.base_dim = len(EQUATION_PATTERNS[key].get("independent", ["t", "x"]))
            else:
                model.base_dim = 2
            return model

        if dynamic:
            return self._dynamic_create(key, hidden_dim, num_layers, **params)

        raise ValueError(
            f"Unknown domain: '{domain}'. "
            f"Call create(..., dynamic=True) to auto-generate. "
            f"Known: {sorted(self._registry)[:15]} ..."
        )

    def _dynamic_create(
        self,
        domain: str,
        hidden_dim: int  = 64,
        num_layers: int  = 4,
        **params,
    ) -> nn.Module:
        """Auto-generate a PINN class for an unknown domain."""
        cache_key = f"{domain}_{hidden_dim}_{num_layers}"
        if cache_key in self._dynamic_cache:
            cls = self._dynamic_cache[cache_key]
            return cls()

        cfg = self._generator.from_hint(
            domain, hidden_dim=hidden_dim, num_layers=num_layers, **params
        )

        try:
            # Use the new SymPyLossGenerator to compile the physics loss from the equation
            from training.sympy_loss_generator import SymPyLossGenerator
            sympy_gen = SymPyLossGenerator()
            eq_str = cfg.equation_info.raw
            # Remove whitespace and replace common patterns for deepxde parsing if needed
            input_vars = cfg.equation_info.independent
            output_var = cfg.equation_info.dependent[0] if cfg.equation_info.dependent else "u"
            pde_fn = sympy_gen.compile_pde(eq_str, input_vars=input_vars, output_var=output_var)
            
            all_params = {**cfg.equation_info.parameters, **params}
            
            # Wrap the DeepXDE function (x, y) into a physics_fn (model, x) expected by _GenericPINN
            def wrapped_physics_fn(model, x):
                x_clone = x.clone().requires_grad_(True)
                y = model(x_clone)
                res = pde_fn(x_clone, y)
                return (res**2).mean()
            
            def model_builder(**kwargs):
                runtime_params = {**all_params, **kwargs}
                model = _GenericPINN(
                    input_dim=cfg.input_dim,
                    output_dim=cfg.output_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    activation=cfg.activation,
                    params=runtime_params,
                    physics_fn=wrapped_physics_fn
                )
                model.base_dim = len(cfg.equation_info.independent)
                return model
                
            self._dynamic_cache[cache_key] = model_builder
            self._registry[domain] = (model_builder, all_params)
            print(f"\n[PINNFactory - SUCCESS] Successfully compiled physics for '{domain}' using SymPy/DeepXDE Loss Generator.")
            return model_builder()
            
        except Exception as e:
            print(f"\n[PINNFactory - FALLBACK] SymPy/DeepXDE failed to parse the equation: {e}")
            print(f"[PINNFactory - FALLBACK] Now relying on hardcoded python text strings from SimulationGenerator...")
            code = self._generator.generate_class(cfg)
    
            # Compile the generated code into a module
            tmp = tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", encoding="utf-8",
                delete=False, dir=tempfile.gettempdir()
            )
            tmp.write(code)
            tmp.close()
    
            spec = importlib.util.spec_from_file_location(cfg.class_name, tmp.name)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            os.unlink(tmp.name)
    
            cls = getattr(mod, cfg.class_name)
            # Merge default parameters with any user-provided params
            merged_params = {**cfg.equation_info.parameters, **params}
            self._dynamic_cache[cache_key] = cls
            self._registry[domain] = (cls, merged_params)
            # Instantiate with merged parameters
            try:
                # Try passing params if the generated class accepts them
                instance = cls(params=merged_params)
            except TypeError:
                # Fallback: the generated class doesn't accept params in __init__
                # Instantiate and set attributes directly
                instance = cls()
                for k, v in merged_params.items():
                    setattr(instance, k, v)
            
            instance.base_dim = len(cfg.equation_info.independent)
            return instance

    def build_from_config(
        self,
        cfg: SimulationConfig,
        **params,
    ) -> nn.Module:
        """Build a PINN directly from a SimulationConfig."""
        eq = cfg.equation_info
        all_params = {**eq.parameters, **params}
        
        try:
            from training.sympy_loss_generator import SymPyLossGenerator
            sympy_gen = SymPyLossGenerator()
            eq_str = eq.raw
            input_vars = eq.independent
            output_var = eq.dependent[0] if eq.dependent else "u"
            pde_fn = sympy_gen.compile_pde(eq_str, input_vars=input_vars, output_var=output_var)
            
            def wrapped_physics_fn(model, x):
                x_clone = x.clone().requires_grad_(True)
                y = model(x_clone)
                res = pde_fn(x_clone, y)
                return (res**2).mean()
        except Exception as e:
            print(f"[PINNFactory] Warning: Failed to compile physics from config: {e}")
            wrapped_physics_fn = None

        return _GenericPINN(
            input_dim  = cfg.input_dim,
            output_dim = cfg.output_dim,
            hidden_dim = cfg.hidden_dim,
            num_layers = cfg.num_layers,
            activation = cfg.activation,
            params     = all_params,
            physics_fn = wrapped_physics_fn
        )

    def build_from_hint(
        self,
        hint:       str,
        hidden_dim: int  = 64,
        num_layers: int  = 4,
        dynamic:    bool = True,
        **params,
    ) -> nn.Module:
        """
        Build a PINN from a free-text hint.

        If the hint matches a known domain, use the static implementation.
        Otherwise auto-generate (dynamic=True required).
        """
        cfg = self._generator.from_hint(hint, hidden_dim=hidden_dim,
                                         num_layers=num_layers, **params)
        eq_type = cfg.equation_info.equation_type
        if eq_type in self._registry:
            return self.create(eq_type, hidden_dim=hidden_dim,
                               num_layers=num_layers, **params)
        if dynamic:
            return self._dynamic_create(eq_type, hidden_dim, num_layers, **params)
        return self.build_from_config(cfg, **params)

    def build_from_equation(
        self,
        equation:   str,
        hidden_dim: int = 64,
        num_layers: int = 4,
        **params,
    ) -> nn.Module:
        """Build a PINN from a math equation string."""
        cfg = self._generator.from_equation(
            equation, params=params, hidden_dim=hidden_dim, num_layers=num_layers
        )
        eq_type = cfg.equation_info.equation_type
        if eq_type in self._registry:
            return self.create(eq_type, hidden_dim=hidden_dim,
                               num_layers=num_layers, **params)
        return self.build_from_config(cfg, **params)

    # ── Registry management ──────────────────────────────────

    def register(self, name: str, cls: Type[nn.Module],
                 defaults: Dict = None):
        """Register a custom PINN class."""
        self._registry[name.lower()] = (cls, defaults or {})
        print(f"[PINNFactory] Registered '{name}' -> {cls.__name__}")

    def unregister(self, name: str):
        self._registry.pop(name.lower(), None)

    def list_domains(self) -> List[str]:
        """Return all registered domain names."""
        return sorted(self._registry.keys())

    def domain_info(self, name: str) -> Dict[str, Any]:
        """Return metadata for a domain."""
        key = name.lower()
        if key not in self._registry:
            return {"error": f"Unknown domain: {key}"}
        cls, defaults = self._registry[key]
        pat = EQUATION_PATTERNS.get(key, {})
        return {
            "domain":      key,
            "class":       cls.__name__,
            "description": pat.get("description", cls.__doc__ or ""),
            "equation":    pat.get("eq_hint", ""),
            "defaults":    defaults,
            "independent": pat.get("independent", []),
            "dependent":   pat.get("dependent", []),
        }

    # ── Batch operations ─────────────────────────────────────

    def batch_create(
        self,
        domains: List[str],
        hidden_dim: int = 64,
        **shared_params,
    ) -> Dict[str, nn.Module]:
        """Create multiple PINNs at once."""
        result = {}
        for d in domains:
            try:
                result[d] = self.create(d, hidden_dim=hidden_dim, **shared_params)
            except ValueError as e:
                print(f"  [PINNFactory] WARNING: {e}")
        return result

    # ── Persistence ──────────────────────────────────────────

    def save(self, model: nn.Module, domain: str, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "domain":     domain,
            "class_name": type(model).__name__,
        }, path)
        print(f"[PINNFactory] Saved {type(model).__name__} -> {path}")

    def load(self, path: str, domain: str, **create_kwargs) -> nn.Module:
        """Load a checkpoint into a fresh model."""
        data  = torch.load(path, map_location="cpu")
        model = self.create(domain, **create_kwargs)
        sd    = data.get("state_dict", data)
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError:
            model.load_state_dict(sd, strict=False)
            print(f"  [PINNFactory] Partial load for {domain}")
        return model

    # ── Convenience ──────────────────────────────────────────

    def generate_code(self, hint: str, output_path: str = None) -> str:
        """Generate standalone PINN class code from a hint."""
        cfg  = self._generator.from_hint(hint)
        code = self._generator.generate_class(cfg)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"[PINNFactory] Code saved to {output_path}")
        return code

    def describe_all(self) -> str:
        """Human-readable summary of all registered domains."""
        lines = [f"PINNFactory -- {len(self._registry)} domains\n"]
        for name in self.list_domains():
            cls, _ = self._registry[name]
            pat    = EQUATION_PATTERNS.get(name, {})
            desc   = pat.get("description", cls.__doc__ or "")
            if desc:
                # Normalise to ASCII-safe string
                desc = (desc.strip().split("\n")[0][:70]
                            .encode("ascii", "replace").decode("ascii"))
            lines.append(f"  {name:<28} {desc}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════

def _smoke_test():
    factory = PINNFactory()

    print("=" * 60)
    print("  PINNFactory Smoke Test")
    print("=" * 60)

    # Test all static domains
    test_domains = [
        "heat", "wave", "burgers", "poisson", "navier_stokes",
        "black_scholes", "sir", "seir", "schrodinger", "kdv",
        "lotka_volterra", "logistic", "arrhenius",
        "advection_diffusion", "allen_cahn", "fitzhugh_nagumo",
        "darcy", "beam", "van_der_pol", "gray_scott",
    ]

    all_ok = True
    for domain in test_domains:
        try:
            model = factory.create(domain)
            model.eval()
            # Build a minimal test tensor
            info  = factory.domain_info(domain)
            n_in  = model.network[0].in_features if hasattr(model, "network") else 2
            x     = torch.rand(8, n_in)
            out   = model(x)
            ploss = model.physics_loss(x).item()
            print(f"  OK  {domain:<28} out={tuple(out.shape)}  "
                  f"phys={ploss:.4f}")
        except Exception as e:
            all_ok = False
            print(f"  FAIL {domain:<28} -> {e}")

    # Dynamic generation test
    print("\n[Dynamic PINN generation]")
    try:
        model = factory.build_from_hint("reaction diffusion Turing patterns")
        print(f"  Dynamic PINN: {type(model).__name__}")
    except Exception as e:
        print(f"  WARNING: Dynamic generation: {e}")

    # Batch creation
    print("\n[Batch creation]")
    models = factory.batch_create(["heat", "wave", "sir"])
    print(f"  Created: {list(models.keys())}")

    # From equation
    print("\n[From equation]")
    m = factory.build_from_equation("du/dt = alpha * d2u/dx2")
    print(f"  {type(m).__name__}  params: alpha={getattr(m,'alpha','?')}")

    # describe_all
    print("\n[Domain registry]")
    print(factory.describe_all()[:600] + "...")

    print(f"\n  All tests {'passed.' if all_ok else 'SOME FAILED.'}")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
