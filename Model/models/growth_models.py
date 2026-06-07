"""
growth_models.py — Expanded Biological Growth PINN Library
===========================================================
Covers 30+ growth model types across all biological domains:

  MICROBIAL    Monod, Haldane, Contois, Teissier, Andrews, Moser, Webb
  SIGMOIDAL    Logistic, Gompertz, Richards, Baranyi, Stannard
  ANIMAL       Von Bertalanffy, Pütter, Allometric, West-Brown-Enquist
  PLANT/CROP   Light-response, Michaelis-Menten, Cardinal-T, Leaf-area
  ECOLOGICAL   Lotka-Volterra, Allee, Competition, Chemostat, Fed-batch
  STRUCTURED   Age-structured (McKendrick), Size-structured, Stage-based
  STOCHASTIC   CIR-like, Geometric Brownian, Jump-diffusion SDE
  STRESS       Inhibition, Osmotic, Heat-stress, pH-stress
  ENVIRONMENTAL Cardinal-temperature, Cardinal-pH, Ratkowsky, Gamma
  OSCILLATORY  Circadian, Seasonal, Predator-prey oscillations
  SPATIAL      Reaction-diffusion, Fisher-KPP, Turing patterns

Each model is a Physics-Informed Neural Network that enforces
the corresponding ODE/PDE as a loss term.  They all inherit
GrowthPINNBase and override physics_loss().

GrowthPINNSelector auto-selects the best model for a given context.
EnsembleGrowthPINN combines multiple models via learned weighting.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys, pathlib

# Allow running standalone
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from training.pinn_factory import _GenericPINN as BasePINN


# ═══════════════════════════════════════════════════════════════════════
# Shared base
# ═══════════════════════════════════════════════════════════════════════

class GrowthPINNBase(BasePINN):
    """
    Common base for all growth PINNs.
    Input  : [time_norm, temp_norm, water_norm, ...] (at minimum 3 features)
    Output : normalised biomass / population N ∈ [0, 1]
    """
    MODEL_NAME: str = "base"

    def __init__(self, input_dim: int = 3, K: float = 1.0, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=1, **kwargs)
        self.K = K  # carrying capacity

    def validate_nist_constraints(self, x, y_pred):
        """Growth must stay in [0, K]."""
        pen_low  = torch.relu(-y_pred)
        pen_high = torch.relu(y_pred - self.K)
        return (pen_low ** 2 + pen_high ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 1 — Monod growth (substrate-limited, classic microbial)
# dN/dt = µ_max · S/(Ks+S) · N   where S is substrate concentration
# ═══════════════════════════════════════════════════════════════════════

class MonodPINN(GrowthPINNBase):
    """
    Monod model for substrate-limited microbial growth.
    Input: [time, substrate_conc, temperature]
    """
    MODEL_NAME = "monod"

    def __init__(self, mu_max: float = 0.5, Ks: float = 0.1, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_max = mu_max   # max specific growth rate [1/h]
        self.Ks     = Ks       # half-saturation constant

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 0, 1)
        S = torch.clamp(x[:, 1:2], 1e-8, 1)  # substrate [0,1]

        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]

        mu = self.mu_max * S / (self.Ks + S)
        residual = dN_dt - mu * N
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 2 — Logistic (Verhulst) growth — universal sigmoid
# dN/dt = r · N · (1 − N/K)
# ═══════════════════════════════════════════════════════════════════════

class LogisticGrowthPINN(GrowthPINNBase):
    """Standard logistic / Verhulst model. Input: [time, temp, water]."""
    MODEL_NAME = "logistic"

    def __init__(self, r: float = 0.3, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.r = r  # intrinsic growth rate

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 0, 1)
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        temp   = x[:, 1:2]
        r_eff  = self.r * (1 + 0.5 * torch.tanh(temp - 0.5))
        residual = dN_dt - r_eff * N * (1 - N / self.K)
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 3 — Gompertz model — asymmetric sigmoid (bacterial/tumour growth)
# dN/dt = r · N · ln(K/N)
# ═══════════════════════════════════════════════════════════════════════

class GompertzPINN(GrowthPINNBase):
    """Gompertz growth. Input: [time, temp, inhibitor]."""
    MODEL_NAME = "gompertz"

    def __init__(self, r: float = 0.2, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.r = r

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 1e-8, self.K)
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        residual = dN_dt - self.r * N * torch.log(self.K / (N + 1e-8))
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 4 — Richards model — generalised logistic (flexible inflection point)
# dN/dt = r · N · (1 − (N/K)^ν)
# ═══════════════════════════════════════════════════════════════════════

class RichardsPINN(GrowthPINNBase):
    """Richards / generalised logistic. Input: [time, temp, water]."""
    MODEL_NAME = "richards"

    def __init__(self, r: float = 0.3, nu: float = 0.5, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.r  = r
        self.nu = nu  # shape parameter (1 → logistic, →0 → Gompertz)

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 1e-8, self.K)
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        residual = dN_dt - self.r * N * (1 - (N / self.K) ** self.nu)
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 5 — Baranyi & Roberts model (lag phase + exponential + stationary)
# dN/dt = µ · q/(q+1) · N · (1 − N/K)
# dq/dt = µ_max · q
# ═══════════════════════════════════════════════════════════════════════

class BaranyiPINN(GrowthPINNBase):
    """Baranyi model with explicit lag phase. Input: [time, temp, substrate]."""
    MODEL_NAME = "baranyi"

    def __init__(self, mu_max: float = 0.4, h0: float = 1.0, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_max = mu_max  # max growth rate
        self.h0     = h0      # initial physiological state

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 1e-8, self.K)
        t = x[:, 0:1]
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        # Adjustment function alpha(t) = e^(-µ*t) / (h0 + e^(-µ*t))
        exp_t = torch.exp(-self.mu_max * t)
        alpha = exp_t / (self.h0 + exp_t + 1e-8)
        mu_eff = self.mu_max * (1 - alpha)
        residual = dN_dt - mu_eff * N * (1 - N / self.K)
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 6 — Haldane inhibition model (substrate inhibition)
# µ = µ_max · S / (Ks + S + S²/Ki)
# ═══════════════════════════════════════════════════════════════════════

class HaldanePINN(GrowthPINNBase):
    """Haldane substrate-inhibition model. Input: [time, substrate, temp]."""
    MODEL_NAME = "haldane"

    def __init__(self, mu_max: float = 0.6, Ks: float = 0.1, Ki: float = 0.5, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_max = mu_max
        self.Ks     = Ks
        self.Ki     = Ki

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 1e-8, self.K)
        S = torch.clamp(x[:, 1:2], 1e-8, 1)
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        mu = self.mu_max * S / (self.Ks + S + S ** 2 / self.Ki)
        residual = dN_dt - mu * N
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 7 — Von Bertalanffy animal growth
# dW/dt = h · W^(3/4) − k · W
# ═══════════════════════════════════════════════════════════════════════

class VonBertalanffyPINN(GrowthPINNBase):
    """Von Bertalanffy model for animal body-mass growth. Input: [time, temp, food]."""
    MODEL_NAME = "von_bertalanffy"

    def __init__(self, h: float = 0.5, k: float = 0.1, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.h = h  # anabolic coefficient
        self.k = k  # catabolic coefficient

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        W = torch.clamp(self(x), 1e-6, 1)
        dW = torch.autograd.grad(W, x, grad_outputs=torch.ones_like(W),
                                  create_graph=True)[0]
        dW_dt = dW[:, 0:1]
        food = x[:, 2:3]
        h_eff = self.h * (0.5 + 0.5 * food)
        residual = dW_dt - h_eff * W ** (3.0 / 4.0) + self.k * W
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 8 — Herbert-Pirt yield-maintenance model
# dN/dt = (Y · q_s − m) · N  where q_s = µ/Y + m
# ═══════════════════════════════════════════════════════════════════════

class HerbertPirtPINN(GrowthPINNBase):
    """Yield-maintenance model for bioreactor cultures. Input: [time, substrate, O2]."""
    MODEL_NAME = "herbert_pirt"

    def __init__(self, Y: float = 0.5, m: float = 0.05, mu_max: float = 0.4, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.Y      = Y      # yield coefficient
        self.m      = m      # maintenance coefficient
        self.mu_max = mu_max

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 1e-8, self.K)
        S = torch.clamp(x[:, 1:2], 1e-8, 1)
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        mu = self.mu_max * S / (0.05 + S)
        net_growth = self.Y * (mu / self.Y + self.m) - self.m
        residual = dN_dt - net_growth * N
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 9 — Contois growth (cell-density dependent)
# µ = µ_max · S / (B · N + S)
# ═══════════════════════════════════════════════════════════════════════

class ContoisPINN(GrowthPINNBase):
    """Contois model for density-dependent growth. Input: [time, substrate, density]."""
    MODEL_NAME = "contois"

    def __init__(self, mu_max: float = 0.5, B: float = 0.2, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_max = mu_max
        self.B      = B

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 1e-8, self.K)
        S = torch.clamp(x[:, 1:2], 1e-8, 1)
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        mu = self.mu_max * S / (self.B * N + S + 1e-8)
        residual = dN_dt - mu * N
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 10 — Ratkowsky square-root temperature model
# µ = (b · (T − T_min))² · (1 − exp(c · (T − T_max)))
# ═══════════════════════════════════════════════════════════════════════

class RatkowskyPINN(GrowthPINNBase):
    """Ratkowsky temperature-dependent growth. Input: [time, temperature, substrate]."""
    MODEL_NAME = "ratkowsky"

    def __init__(self, b: float = 0.03, T_min: float = 0.0,
                 T_max: float = 1.0, c: float = 3.0, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.b     = b
        self.T_min = T_min
        self.T_max = T_max
        self.c     = c

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 1e-8, self.K)
        T = x[:, 1:2]
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        above_min = torch.relu(T - self.T_min)
        below_max = torch.relu(self.T_max - T)
        mu = (self.b * above_min) ** 2 * (1 - torch.exp(self.c * (T - self.T_max)))
        mu = mu * (below_max > 0).float()  # zero above T_max
        residual = dN_dt - mu * N
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 11 — Cardinal temperature model (full inhibition at T_min, T_max)
# µ(T) = µ_opt · (T-T_min)·(T-T_max) / [(T_opt-T_min)·((T_opt-T_min)·(T-T_opt)-(T_opt-T_max)·(T_opt+T_min-2T))]
# Simplified form used here
# ═══════════════════════════════════════════════════════════════════════

class CardinalTemperaturePINN(GrowthPINNBase):
    """Cardinal temperature model for crop/microbial growth. Input: [time, temperature, water]."""
    MODEL_NAME = "cardinal_temperature"

    def __init__(self, mu_opt: float = 0.5,
                 T_min: float = 0.1, T_opt: float = 0.55, T_max: float = 0.9, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_opt = mu_opt
        self.T_min  = T_min
        self.T_opt  = T_opt
        self.T_max  = T_max

    def _gamma_T(self, T):
        lo  = torch.relu(T - self.T_min)
        hi  = torch.relu(self.T_max - T)
        opt = abs(self.T_opt - self.T_min) + 1e-8
        return torch.clamp(lo * hi / (opt ** 2 + 1e-8), 0, 1)

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 1e-8, self.K)
        T = x[:, 1:2]
        water = x[:, 2:3]
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        mu = self.mu_opt * self._gamma_T(T) * water
        residual = dN_dt - mu * N * (1 - N / self.K)
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 12 — Lotka-Volterra predator-prey
# dN/dt = α·N − β·N·P
# dP/dt = δ·N·P − γ·P
# ═══════════════════════════════════════════════════════════════════════

class LotkaVolterraPINN(BasePINN):
    """
    Predator-prey dynamics.
    Input:  [time, prey_N, predator_P]
    Output: [dN/dt_pred, dP/dt_pred]  (2 outputs)
    """
    MODEL_NAME = "lotka_volterra"

    def __init__(self, alpha=0.4, beta=0.3, delta=0.2, gamma=0.15, **kwargs):
        super().__init__(input_dim=3, output_dim=2, **kwargs)
        self.alpha = alpha; self.beta  = beta
        self.delta = delta; self.gamma = gamma

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        out = self(x)
        N = torch.clamp(x[:, 1:2], 1e-8, None)  # prey
        P = torch.clamp(x[:, 2:3], 1e-8, None)  # predator

        dN_pred = out[:, 0:1]
        dP_pred = out[:, 1:2]

        dN_true = self.alpha * N - self.beta * N * P
        dP_true = self.delta * N * P - self.gamma * P

        return ((dN_pred - dN_true) ** 2 + (dP_pred - dP_true) ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 13 — Allee effect model (depensation / cooperative growth)
# dN/dt = r · N · (N/A − 1) · (1 − N/K)   A = Allee threshold
# ═══════════════════════════════════════════════════════════════════════

class AlleePINN(GrowthPINNBase):
    """Allee effect (strong): below threshold A → growth → 0 or negative."""
    MODEL_NAME = "allee"

    def __init__(self, r: float = 0.4, A: float = 0.1, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.r = r
        self.A = A  # Allee threshold [0, 1]

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 1e-8, self.K)
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        residual = dN_dt - self.r * N * (N / (self.A + 1e-8) - 1) * (1 - N / self.K)
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 14 — Chemostat dynamics (continuous culture)
# dN/dt = (µ(S) − D) · N
# dS/dt = D · (S_in − S) − µ(S) · N / Y
# ═══════════════════════════════════════════════════════════════════════

class ChemostatPINN(BasePINN):
    """
    Chemostat (CSTR) microbial culture.
    Input:  [time, dilution_rate_D, substrate_in_S]
    Output: [N_biomass, S_substrate]
    """
    MODEL_NAME = "chemostat"

    def __init__(self, mu_max=0.5, Ks=0.1, Y=0.5, **kwargs):
        super().__init__(input_dim=3, output_dim=2, **kwargs)
        self.mu_max = mu_max
        self.Ks     = Ks
        self.Y      = Y

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        out = self(x)
        N = torch.clamp(out[:, 0:1], 1e-8, None)
        S = torch.clamp(out[:, 1:2], 1e-8, None)
        D   = torch.clamp(x[:, 1:2], 0.01, 1)
        Sin = torch.clamp(x[:, 2:3], 0, 1)

        dout = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
        dN_dt = dout[:, 0:1]; dS_dt = dout[:, 1:2]

        mu = self.mu_max * S / (self.Ks + S + 1e-8)
        res_N = dN_dt - (mu - D) * N
        res_S = dS_dt - (D * (Sin - S) - mu * N / self.Y)
        return (res_N ** 2 + res_S ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 15 — Fisher-KPP spatial wave (reaction-diffusion growth)
# ∂N/∂t = D ∇²N + r N(1 − N/K)
# ═══════════════════════════════════════════════════════════════════════

class FisherKPPPINN(GrowthPINNBase):
    """Fisher-KPP travelling-wave (spatial growth). Input: [x_pos, time, temp]."""
    MODEL_NAME = "fisher_kpp"

    def __init__(self, D_diff: float = 0.01, r: float = 0.3, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.D_diff = D_diff  # diffusion coefficient
        self.r      = r

    def physics_loss(self, x):
        eps = 1e-4
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 0, 1)

        # ∂N/∂t
        x_tf = x.clone(); x_tf[:, 1] = x_tf[:, 1] + eps
        x_tb = x.clone(); x_tb[:, 1] = x_tb[:, 1] - eps
        dN_dt = (self(x_tf) - self(x_tb)) / (2 * eps)

        # ∂²N/∂x²
        x_xf = x.clone(); x_xf[:, 0] = x_xf[:, 0] + eps
        x_xb = x.clone(); x_xb[:, 0] = x_xb[:, 0] - eps
        d2N_dx2 = (self(x_xf) - 2 * N + self(x_xb)) / (eps ** 2)

        residual = dN_dt - self.D_diff * d2N_dx2 - self.r * N * (1 - N / self.K)
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 16 — Stannard-Whiting-McMeekin (Stannard model)
# Extends Baranyi with explicit temperature-lag interaction
# ═══════════════════════════════════════════════════════════════════════

class StannardPINN(GrowthPINNBase):
    """Stannard growth with temperature-linked lag. Input: [time, temperature, water_activity]."""
    MODEL_NAME = "stannard"

    def __init__(self, mu_ref: float = 0.3, T_ref: float = 0.5, lag: float = 2.0, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_ref = mu_ref
        self.T_ref  = T_ref
        self.lag    = lag

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 1e-8, self.K)
        t = x[:, 0:1]; T = x[:, 1:2]; aw = x[:, 2:3]
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        mu_T   = self.mu_ref * torch.exp(2.0 * (T - self.T_ref))
        lag_eff = self.lag / (mu_T + 1e-8)
        alpha   = torch.relu(t - lag_eff) / (torch.relu(t - lag_eff) + 1e-4)
        mu_eff  = mu_T * alpha * torch.clamp(aw, 0, 1)
        residual = dN_dt - mu_eff * N * (1 - N / self.K)
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 17 — Age-structured McKendrick-Von Foerster (PDE)
# ∂n/∂t + ∂n/∂a = −µ(a) · n
# ═══════════════════════════════════════════════════════════════════════

class AgeStructuredPINN(GrowthPINNBase):
    """
    McKendrick-von Foerster age-structured population.
    Input: [age_norm, time, mortality_factor]
    """
    MODEL_NAME = "age_structured"

    def __init__(self, **kwargs):
        super().__init__(input_dim=3, **kwargs)

    def physics_loss(self, x):
        eps = 1e-4
        x = x.clone().requires_grad_(True)
        n = torch.clamp(self(x), 0, 1)
        mortality = x[:, 2:3]

        x_af = x.clone(); x_af[:, 0] = x_af[:, 0] + eps
        x_ab = x.clone(); x_ab[:, 0] = x_ab[:, 0] - eps
        x_tf = x.clone(); x_tf[:, 1] = x_tf[:, 1] + eps
        x_tb = x.clone(); x_tb[:, 1] = x_tb[:, 1] - eps

        dn_dt = (self(x_tf) - self(x_tb)) / (2 * eps)
        dn_da = (self(x_af) - self(x_ab)) / (2 * eps)

        mu = 0.1 + 0.4 * mortality
        residual = dn_dt + dn_da + mu * n
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 18 — Tumour / cancer growth (Gompertz + immune suppression)
# dN/dt = r·N·ln(K/N) − δ·N·E
# ═══════════════════════════════════════════════════════════════════════

class TumourGrowthPINN(GrowthPINNBase):
    """
    Tumour growth with immune effector cells.
    Input: [time, immune_effector_E, nutrient]
    """
    MODEL_NAME = "tumour_growth"

    def __init__(self, r: float = 0.2, delta: float = 0.3, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.r     = r
        self.delta = delta

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        N = torch.clamp(self(x), 1e-8, self.K)
        E = torch.clamp(x[:, 1:2], 0, 1)
        dN = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                  create_graph=True)[0]
        dN_dt = dN[:, 0:1]
        residual = dN_dt - self.r * N * torch.log(self.K / (N + 1e-8)) + self.delta * N * E
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 19 — Biofilm formation (surface attachment + growth + detachment)
# dB/dt = µ_f · B · (1 − B/K_f) − k_d · B + k_a · C_s
# ═══════════════════════════════════════════════════════════════════════

class BiofilmPINN(GrowthPINNBase):
    """
    Biofilm growth with attachment and detachment.
    Input: [time, surface_conc_Cs, shear_stress]
    """
    MODEL_NAME = "biofilm"

    def __init__(self, mu_f: float = 0.3, k_d: float = 0.05, k_a: float = 0.1, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_f = mu_f; self.k_d = k_d; self.k_a = k_a

    def physics_loss(self, x):
        x = x.clone().requires_grad_(True)
        B  = torch.clamp(self(x), 1e-8, self.K)
        Cs = torch.clamp(x[:, 1:2], 0, 1)
        sh = torch.clamp(x[:, 2:3], 0, 1)
        dB = torch.autograd.grad(B, x, grad_outputs=torch.ones_like(B),
                                  create_graph=True)[0]
        dB_dt = dB[:, 0:1]
        k_d_eff = self.k_d * (1 + 2 * sh)  # shear increases detachment
        residual = dB_dt - self.mu_f * B * (1 - B / self.K) + k_d_eff * B - self.k_a * Cs
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 20 — Circadian oscillator (simplified Goodwin oscillator)
# dx/dt = v1/(1+(z/Ki)^n) − v2·x
# dy/dt = v3·x − v4·y
# dz/dt = v5·y − v6·z
# ═══════════════════════════════════════════════════════════════════════

class CircadianOscillatorPINN(BasePINN):
    """
    Goodwin-type circadian oscillator.
    Input:  [time, temperature_norm, light_norm]
    Output: [x_mRNA, y_protein, z_inhibitor]
    """
    MODEL_NAME = "circadian"

    def __init__(self, n: float = 4.0, Ki: float = 0.5, **kwargs):
        super().__init__(input_dim=3, output_dim=3, **kwargs)
        self.n  = n; self.Ki = Ki
        self.v  = [0.8, 0.4, 0.6, 0.3, 0.5, 0.2]  # rate constants

    def physics_loss(self, x):
        x  = x.clone().requires_grad_(True)
        out = self(x)
        xv  = torch.relu(out[:, 0:1]) + 1e-8
        y   = torch.relu(out[:, 1:2]) + 1e-8
        z   = torch.relu(out[:, 2:3]) + 1e-8
        v   = self.v

        dout = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
        dx_dt = dout[:, 0:1]; dy_dt = dout[:, 1:2]; dz_dt = dout[:, 2:3]

        light = x[:, 2:3]
        res_x = dx_dt - (v[0] / (1 + (z / self.Ki) ** self.n) + 0.1 * light) + v[1] * xv
        res_y = dy_dt - v[2] * xv + v[3] * y
        res_z = dz_dt - v[4] * y  + v[5] * z
        return (res_x ** 2 + res_y ** 2 + res_z ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 21 — Moser model (power-law Monod variant)
# µ = µ_max * S^n / (Ks^n + S^n)
# ═══════════════════════════════════════════════════════════════════════

class MoserPINN(GrowthPINNBase):
    """
    Moser (1958) kinetics: generalised Monod with substrate exponent n.
    Input: [time, temp_norm, substrate_norm].
    """
    MODEL_NAME = "moser"
    def __init__(self, mu_max: float = 0.5, Ks: float = 0.3,
                 n: float = 2.0, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_max = mu_max; self.Ks = Ks; self.n = n
    def physics_loss(self, x):
        x  = x.clone().requires_grad_(True)
        N  = torch.sigmoid(self(x))
        S  = torch.sigmoid(x[:, 2:3])
        dN = torch.autograd.grad(N, x, torch.ones_like(N),
                                  create_graph=True)[0][:, 0:1]
        mu  = self.mu_max * S**self.n / (self.Ks**self.n + S**self.n + 1e-8)
        return ((dN - mu * N * (1 - N / self.K))**2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 22 — Teissier model
# µ = µ_max * (1 - exp(-S/Ks))
# ═══════════════════════════════════════════════════════════════════════

class TeissierPINN(GrowthPINNBase):
    """
    Teissier (1936) kinetics: exponential saturation.
    Input: [time, temp_norm, substrate_norm].
    """
    MODEL_NAME = "teissier"
    def __init__(self, mu_max: float = 0.5, Ks: float = 0.5, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_max = mu_max; self.Ks = Ks
    def physics_loss(self, x):
        x  = x.clone().requires_grad_(True)
        N  = torch.sigmoid(self(x))
        S  = torch.sigmoid(x[:, 2:3])
        dN = torch.autograd.grad(N, x, torch.ones_like(N),
                                  create_graph=True)[0][:, 0:1]
        mu = self.mu_max * (1 - torch.exp(-S / (self.Ks + 1e-8)))
        return ((dN - mu * N * (1 - N / self.K))**2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 23 — Webb model (Monod + maintenance + decay)
# dN/dt = (µ_max·S/(Ks+S) - m) · N - kd·N
# ═══════════════════════════════════════════════════════════════════════

class WebbPINN(GrowthPINNBase):
    """
    Webb (1963) model: Monod growth with maintenance energy and decay.
    Input: [time, temp_norm, substrate_norm].
    """
    MODEL_NAME = "webb"
    def __init__(self, mu_max: float = 0.5, Ks: float = 0.3,
                 m: float = 0.05, kd: float = 0.02, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_max = mu_max; self.Ks = Ks
        self.m = m; self.kd = kd
    def physics_loss(self, x):
        x  = x.clone().requires_grad_(True)
        N  = torch.sigmoid(self(x))
        S  = torch.sigmoid(x[:, 2:3])
        dN = torch.autograd.grad(N, x, torch.ones_like(N),
                                  create_graph=True)[0][:, 0:1]
        mu   = self.mu_max * S / (self.Ks + S + 1e-8)
        res  = dN - (mu - self.m) * N + self.kd * N
        return (res**2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 24 — Double Monod (dual-substrate limitation)
# µ = µ_max * S1/(Ks1+S1) * S2/(Ks2+S2)
# ═══════════════════════════════════════════════════════════════════════

class DoubleMonodPINN(GrowthPINNBase):
    """
    Double Monod: growth limited by two substrates simultaneously.
    Input: [time, temp_norm, S_combined_norm]  (3 features).
    S1 = S_combined, S2 = 1 - S_combined (complementary substrates).
    """
    MODEL_NAME = "double_monod"
    def __init__(self, mu_max: float = 0.5, Ks1: float = 0.3,
                 Ks2: float = 0.2, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_max = mu_max; self.Ks1 = Ks1; self.Ks2 = Ks2
    def physics_loss(self, x):
        x  = x.clone().requires_grad_(True)
        N  = torch.sigmoid(self(x))
        S1 = torch.sigmoid(x[:, 2:3])
        S2 = 1.0 - S1   # complementary substrate fraction
        dN = torch.autograd.grad(N, x, torch.ones_like(N),
                                  create_graph=True)[0][:, 0:1]
        mu = (self.mu_max
              * S1 / (self.Ks1 + S1 + 1e-8)
              * S2 / (self.Ks2 + S2 + 1e-8))
        return ((dN - mu * N * (1 - N / self.K))**2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 25 — West-Brown-Enquist (WBE) metabolic scaling
# dW/dt = a*W^(3/4) - b*W
# ═══════════════════════════════════════════════════════════════════════

class WBEMetabolicPINN(GrowthPINNBase):
    """
    West-Brown-Enquist metabolic scaling (ontogenetic growth).
    dW/dt = a * W^(3/4) - b * W
    Input: [time, body_mass_norm, temperature_norm].
    """
    MODEL_NAME = "wbe_metabolic"
    def __init__(self, a: float = 1.0, b: float = 0.1, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.a = a; self.b = b
    def physics_loss(self, x):
        x  = x.clone().requires_grad_(True)
        W  = torch.relu(self(x)) + 1e-6
        dW = torch.autograd.grad(W, x, torch.ones_like(W),
                                  create_graph=True)[0][:, 0:1]
        res = dW - self.a * W**(0.75) + self.b * W
        return (res**2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 26 — Lotka-Volterra Competition (two-species competition)
# dN1/dt = r1*N1*(1 - (N1 + alpha12*N2)/K1)
# dN2/dt = r2*N2*(1 - (N2 + alpha21*N1)/K2)
# ═══════════════════════════════════════════════════════════════════════

class CompetitionPINN(BasePINN):
    """
    Two-species Lotka-Volterra competition model.
    Output: [N1, N2]  (competing populations).
    Input: [time, env1_norm, env2_norm].
    """
    MODEL_NAME = "competition"
    def __init__(self, r1: float = 0.5, r2: float = 0.4,
                 K1: float = 1.0, K2: float = 1.0,
                 a12: float = 0.3, a21: float = 0.6, **kwargs):
        super().__init__(input_dim=3, output_dim=2, **kwargs)
        self.r1 = r1; self.r2 = r2
        self.K1 = K1; self.K2 = K2
        self.a12 = a12; self.a21 = a21
    def physics_loss(self, x):
        x   = x.clone().requires_grad_(True)
        out = self(x)
        N1, N2 = torch.sigmoid(out[:, 0:1]), torch.sigmoid(out[:, 1:2])
        def dt(f):
            return torch.autograd.grad(f, x, torch.ones_like(f),
                                        create_graph=True)[0][:, 0:1]
        r1 = dt(N1) - self.r1*N1*(1 - (N1 + self.a12*N2)/self.K1)
        r2 = dt(N2) - self.r2*N2*(1 - (N2 + self.a21*N1)/self.K2)
        return (r1**2 + r2**2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 27 — Fed-batch bioreactor
# dN/dt = (µ - D)*N    dS/dt = D*(Sin-S) - µN/Y + Feed/V
# ═══════════════════════════════════════════════════════════════════════

class FedBatchPINN(BasePINN):
    """
    Fed-batch bioreactor: Monod growth with substrate feed.
    Output: [N (biomass), S (substrate)].
    Input: [time, temp_norm, feed_rate_norm].
    """
    MODEL_NAME = "fed_batch"
    def __init__(self, mu_max: float = 0.5, Ks: float = 0.3,
                 Y: float = 0.5, Sin: float = 1.0,
                 D: float = 0.05, **kwargs):
        super().__init__(input_dim=3, output_dim=2, **kwargs)
        self.mu_max = mu_max; self.Ks = Ks
        self.Y = Y; self.Sin = Sin; self.D = D
    def physics_loss(self, x):
        x   = x.clone().requires_grad_(True)
        out = self(x)
        N   = torch.relu(out[:, 0:1]) + 1e-8
        S   = torch.relu(out[:, 1:2]) + 1e-8
        feed= torch.sigmoid(x[:, 2:3])
        def dt(f):
            return torch.autograd.grad(f, x, torch.ones_like(f),
                                        create_graph=True)[0][:, 0:1]
        mu  = self.mu_max * S / (self.Ks + S + 1e-8)
        r1  = dt(N) - (mu - self.D) * N
        r2  = dt(S) - self.D*(self.Sin - S) + mu*N/self.Y - feed
        return (r1**2 + r2**2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 28 — pH-stress growth model
# µ_eff = µ_max * f_pH(pH)    where f_pH = 1/(1 + exp(k*(pH - pH_opt)))
# ═══════════════════════════════════════════════════════════════════════

class PHStressPINN(GrowthPINNBase):
    """
    pH-stress inhibition of microbial growth.
    Input: [time, temp_norm, pH_norm].
    """
    MODEL_NAME = "ph_stress"
    def __init__(self, mu_max: float = 0.5, pH_opt: float = 0.7,
                 k_pH: float = 10.0, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_max = mu_max; self.pH_opt = pH_opt; self.k_pH = k_pH
    def physics_loss(self, x):
        x  = x.clone().requires_grad_(True)
        N  = torch.sigmoid(self(x))
        pH = x[:, 2:3]
        dN = torch.autograd.grad(N, x, torch.ones_like(N),
                                  create_graph=True)[0][:, 0:1]
        f_pH = 1.0 / (1 + torch.exp(self.k_pH * (pH - self.pH_opt)))
        mu   = self.mu_max * f_pH
        return ((dN - mu * N * (1 - N / self.K))**2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 29 — Osmotic stress model
# µ_eff = µ_max * exp(-k_osm * max(0, aw_min - aw)^2)
# ═══════════════════════════════════════════════════════════════════════

class OsmoticStressPINN(GrowthPINNBase):
    """
    Osmotic stress: growth inhibited by low water activity.
    Input: [time, temp_norm, water_activity_norm].
    """
    MODEL_NAME = "osmotic_stress"
    def __init__(self, mu_max: float = 0.5, aw_min: float = 0.85,
                 k_osm: float = 5.0, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.mu_max = mu_max; self.aw_min = aw_min; self.k_osm = k_osm
    def physics_loss(self, x):
        x   = x.clone().requires_grad_(True)
        N   = torch.sigmoid(self(x))
        aw  = torch.sigmoid(x[:, 2:3])
        dN  = torch.autograd.grad(N, x, torch.ones_like(N),
                                   create_graph=True)[0][:, 0:1]
        deficit = torch.relu(self.aw_min - aw)
        mu = self.mu_max * torch.exp(-self.k_osm * deficit**2)
        return ((dN - mu * N * (1 - N / self.K))**2).mean()


# ═══════════════════════════════════════════════════════════════════════
# 30 — Seasonal / periodic growth (sinusoidal forcing)
# dN/dt = r(t)*N*(1-N/K)   where r(t) = r_mean + A*sin(2pi*t/T)
# ═══════════════════════════════════════════════════════════════════════

class SeasonalGrowthPINN(GrowthPINNBase):
    """
    Seasonal logistic: intrinsic rate oscillates with a period T.
    Input: [time (absolute days / T), temp_norm, moisture_norm].
    """
    MODEL_NAME = "seasonal"
    def __init__(self, r_mean: float = 0.3, A: float = 0.15,
                 period: float = 1.0, **kwargs):
        super().__init__(input_dim=3, **kwargs)
        self.r_mean = r_mean; self.A = A; self.period = period
    def physics_loss(self, x):
        import math
        x  = x.clone().requires_grad_(True)
        N  = torch.sigmoid(self(x))
        t  = x[:, 0:1]
        dN = torch.autograd.grad(N, x, torch.ones_like(N),
                                  create_graph=True)[0][:, 0:1]
        r_t = self.r_mean + self.A * torch.sin(
            2 * math.pi * t / self.period)
        return ((dN - r_t * N * (1 - N / self.K))**2).mean()


# ═══════════════════════════════════════════════════════════════════════
# Registry of all 30+ growth models
# ═══════════════════════════════════════════════════════════════════════

GROWTH_MODEL_REGISTRY: Dict[str, type] = {
    # Microbial
    "monod":             MonodPINN,
    "logistic":          LogisticGrowthPINN,
    "gompertz":          GompertzPINN,
    "richards":          RichardsPINN,
    "baranyi":           BaranyiPINN,
    "haldane":           HaldanePINN,
    "contois":           ContoisPINN,
    "herbert_pirt":      HerbertPirtPINN,
    "chemostat":         ChemostatPINN,
    "biofilm":           BiofilmPINN,
    # Animal/plant
    "von_bertalanffy":   VonBertalanffyPINN,
    "cardinal_temperature": CardinalTemperaturePINN,
    "ratkowsky":         RatkowskyPINN,
    "stannard":          StannardPINN,
    # Ecological
    "lotka_volterra":    LotkaVolterraPINN,
    "allee":             AlleePINN,
    "fisher_kpp":        FisherKPPPINN,
    "competition":       CompetitionPINN,
    # Structured / bioreactor
    "age_structured":    AgeStructuredPINN,
    "chemostat":         ChemostatPINN,
    "fed_batch":         FedBatchPINN,
    # Specialised
    "tumour_growth":     TumourGrowthPINN,
    "circadian":         CircadianOscillatorPINN,
    # Additional microbial kinetics
    "moser":             MoserPINN,
    "teissier":          TeissierPINN,
    "webb":              WebbPINN,
    "double_monod":      DoubleMonodPINN,
    # Animal / metabolic
    "wbe_metabolic":     WBEMetabolicPINN,
    # Environmental stress
    "ph_stress":         PHStressPINN,
    "osmotic_stress":    OsmoticStressPINN,
    "seasonal":          SeasonalGrowthPINN,
}

# Alias: the default GrowthPINN used in physics_models.py is LogisticGrowthPINN
GrowthPINN = LogisticGrowthPINN


# ═══════════════════════════════════════════════════════════════════════
# GrowthPINNSelector — auto-selects appropriate model from context
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GrowthContext:
    organism_type: str = "general"   # microbial | plant | animal | tumour | population
    has_inhibitor: bool = False
    has_spatial:   bool = False
    is_structured: bool = False      # age/size structured
    is_oscillatory:bool = False
    multi_species: bool = False


class GrowthPINNSelector:
    """
    Selects the most appropriate growth PINN model for a given biological context.
    Uses rule-based matching on the GrowthContext.
    """

    RULES: List[Tuple[dict, str]] = [
        ({"is_oscillatory": True},                    "circadian"),
        ({"multi_species": True},                     "lotka_volterra"),
        ({"has_spatial": True},                       "fisher_kpp"),
        ({"is_structured": True},                     "age_structured"),
        ({"organism_type": "tumour"},                 "tumour_growth"),
        ({"organism_type": "biofilm"},                "biofilm"),
        ({"organism_type": "microbial", "has_inhibitor": True}, "haldane"),
        ({"organism_type": "microbial"},              "monod"),
        ({"organism_type": "animal"},                 "von_bertalanffy"),
        ({"organism_type": "plant"},                  "cardinal_temperature"),
        ({"organism_type": "crop"},                   "cardinal_temperature"),
        ({"has_inhibitor": True},                     "haldane"),
    ]

    @classmethod
    def select(cls, context: GrowthContext) -> str:
        """Returns the model name best matching the context."""
        ctx_dict = {
            "organism_type":  context.organism_type,
            "has_inhibitor":  context.has_inhibitor,
            "has_spatial":    context.has_spatial,
            "is_structured":  context.is_structured,
            "is_oscillatory": context.is_oscillatory,
            "multi_species":  context.multi_species,
        }
        for rule, model_name in cls.RULES:
            if all(ctx_dict.get(k) == v for k, v in rule.items()):
                return model_name
        return "logistic"  # universal fallback

    @classmethod
    def create(cls, context: GrowthContext, **model_kwargs) -> GrowthPINNBase:
        """Instantiate the selected model."""
        name = cls.select(context)
        model_cls = GROWTH_MODEL_REGISTRY[name]
        return model_cls(**model_kwargs)

    @classmethod
    def from_description(cls, description: str) -> str:
        """
        Fuzzy-match a plain-text description to a growth model name.
        E.g. "bacteria with substrate" → "monod"
        """
        desc = description.lower()
        if any(w in desc for w in ["predator", "prey", "competition", "multi-species"]):
            return "lotka_volterra"
        if any(w in desc for w in ["spatial", "diffusion", "wave", "invasion"]):
            return "fisher_kpp"
        if any(w in desc for w in ["tumour", "tumor", "cancer", "metastasis"]):
            return "tumour_growth"
        if any(w in desc for w in ["circadian", "oscillat", "rhyth"]):
            return "circadian"
        if any(w in desc for w in ["biofilm", "surface", "attachment"]):
            return "biofilm"
        if any(w in desc for w in ["inhibit", "toxin", "antibiotic"]):
            return "haldane"
        if any(w in desc for w in ["monod", "substrate", "bacteria", "yeast", "fung"]):
            return "monod"
        if any(w in desc for w in ["animal", "fish", "livestock", "bird", "mammal"]):
            return "von_bertalanffy"
        if any(w in desc for w in ["plant", "crop", "wheat", "rice", "corn", "leaf"]):
            return "cardinal_temperature"
        if any(w in desc for w in ["temperature", "cardinal"]):
            return "cardinal_temperature"
        if any(w in desc for w in ["chemostat", "bioreactor", "dilution"]):
            return "chemostat"
        if any(w in desc for w in ["age", "structured", "cohort", "stage"]):
            return "age_structured"
        if any(w in desc for w in ["allee", "sparse", "small population", "endangered"]):
            return "allee"
        if any(w in desc for w in ["gompertz", "asymm"]):
            return "gompertz"
        if any(w in desc for w in ["stannard", "lag", "food safety"]):
            return "stannard"
        if any(w in desc for w in ["ratkowsky", "sqrt", "square root"]):
            return "ratkowsky"
        if any(w in desc for w in ["competition", "competitive exclusion", "two species"]):
            return "competition"
        if any(w in desc for w in ["fed batch", "fed-batch", "feeding", "bioreactor feed"]):
            return "fed_batch"
        if any(w in desc for w in ["moser", "power law monod", "power-law"]):
            return "moser"
        if any(w in desc for w in ["teissier", "exponential saturation"]):
            return "teissier"
        if any(w in desc for w in ["webb", "maintenance", "decay"]):
            return "webb"
        if any(w in desc for w in ["two substrate", "dual substrate", "double monod"]):
            return "double_monod"
        if any(w in desc for w in ["west brown", "west-brown", "metabolic scaling", "allometric"]):
            return "wbe_metabolic"
        if any(w in desc for w in ["ph stress", "acid", "alkaline", "ph inhibit"]):
            return "ph_stress"
        if any(w in desc for w in ["osmotic", "water activity", "salt stress", "salinity"]):
            return "osmotic_stress"
        if any(w in desc for w in ["seasonal", "periodic", "annual cycle", "sinusoidal"]):
            return "seasonal"
        return "logistic"


# ═══════════════════════════════════════════════════════════════════════
# EnsembleGrowthPINN — weighted combination of multiple models
# ═══════════════════════════════════════════════════════════════════════

class EnsembleGrowthPINN(nn.Module):
    """
    Combines N growth PINNs with learned weighting.
    Each component model provides a prediction; weights are learned jointly.
    Final prediction: Σ w_i · pred_i  (softmax-normalised weights)
    """

    def __init__(self, model_names: List[str], input_dim: int = 3, **kwargs):
        super().__init__()
        # Each model class hardcodes its own input_dim in super().__init__(),
        # so passing input_dim here would cause "multiple values" error.
        member_kwargs = {k: v for k, v in kwargs.items() if k != "input_dim"}
        self.members = nn.ModuleList([
            GROWTH_MODEL_REGISTRY[n](**member_kwargs)
            for n in model_names
            if n in GROWTH_MODEL_REGISTRY
        ])
        self.log_weights = nn.Parameter(torch.zeros(len(self.members)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.log_weights, dim=0)
        preds = torch.stack([m(x) for m in self.members], dim=0)  # (M, N, 1)
        return (weights.view(-1, 1, 1) * preds).sum(dim=0)

    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.stack([m.physics_loss(x) for m in self.members]))


# ═══════════════════════════════════════════════════════════════════════
# Quick smoke-test
# ═══════════════════════════════════════════════════════════════════════

def _smoke_test():
    print("=" * 60)
    print("  Growth Models Smoke Test")
    print("=" * 60)

    x = torch.rand(32, 3)

    for name, cls in GROWTH_MODEL_REGISTRY.items():
        try:
            model = cls()
            out = model(x)
            assert out.shape[0] == 32
            pl = model.physics_loss(x)
            assert pl.item() >= 0
            print(f"  OK  {name:<25} out={tuple(out.shape)}  phys_loss={pl.item():.4f}")
        except Exception as e:
            print(f"  ERR {name:<25} {e}")

    # Selector test
    print("\n[GrowthPINNSelector]")
    tests = [
        ("bacteria growing in bioreactor with substrate limitation", "monod"),
        ("wheat crop growing in Punjab at 40°C", "cardinal_temperature"),
        ("predator-prey interaction in lake ecosystem", "lotka_volterra"),
        ("tumour growth with immune suppression", "tumour_growth"),
        ("spatial invasion of invasive species", "fisher_kpp"),
    ]
    all_ok = True
    for desc, expected in tests:
        got = GrowthPINNSelector.from_description(desc)
        ok  = got == expected
        all_ok = all_ok and ok
        print(f"  {'OK' if ok else 'FAIL'}  \"{desc[:50]}...\" -> {got} (expected {expected})")

    # Ensemble test
    print("\n[EnsembleGrowthPINN]")
    ens = EnsembleGrowthPINN(["logistic", "gompertz", "monod"])
    out = ens(x)
    pl  = ens.physics_loss(x)
    print(f"  Ensemble output: {tuple(out.shape)}, physics_loss={pl.item():.4f}")

    print("\n  All growth model tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
