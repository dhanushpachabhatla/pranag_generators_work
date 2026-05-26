"""
loss_generator.py — Automated Loss Function Generator
======================================================
Generates loss components for all 7 categories per the PRANA-G spec:
  1. Data loss (match with real data)
  2. Physics loss (PDE residuals)
  3. Boundary loss (constraints)
  4. Biology loss (genetic/cellular rules)
  5. Ecology loss (environmental safety)
  6. Economics loss (cost viability)
  7. Safety loss (toxicity/containment)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass


@dataclass
class LossComponent:
    name: str
    weight: float
    compute_fn: Callable
    description: str


class LossGenerator:
    """Generates loss functions dynamically for any domain/constraint."""

    def __init__(self):
        self.components: Dict[str, LossComponent] = {}
        self.weights: Dict[str, float] = self._default_weights()

    def _default_weights(self) -> Dict[str, float]:
        """Initial weights per PRANA-G spec."""
        return {
            "data":     1.0,      # λ₁
            "physics":  1.5,      # λ₂
            "boundary": 1.2,      # λ₃
            "biology":  1.8,      # λ₄
            "ecology":  1.6,      # λ₅
            "economics": 0.8,     # λ₆
            "safety":   2.0,      # λ₇ (highest priority)
        }

    # ────────────────────────────────────────────────────────────
    # COMPONENT BUILDERS
    # ────────────────────────────────────────────────────────────

    def add_data_loss(self, observed: Optional[torch.Tensor] = None, weight_fn: Optional[Callable] = None):
        """Loss_Data: MSE to real observations, weighted by data quality."""
        def compute(pred):
            if pred is None: return torch.tensor(0.0)
            # If observed is None, assume pred is already the residual
            obs = observed if observed is not None else torch.zeros_like(pred)
            mse = (pred - obs) ** 2
            if weight_fn is not None:
                mse = mse * weight_fn()
            return mse.mean()

        self.components["data"] = LossComponent(
            name="Data",
            weight=self.weights["data"],
            compute_fn=compute,
            description="Match real-world observations"
        )

    def add_physics_loss(self, domain: str = "general"):
        """Loss_Physics: PDE residual penalty for fundamental laws."""
        def compute(pred):
            if pred is None: return torch.tensor(0.0)
            # pred is expected to be the pde_residuals
            residual_norm = torch.norm(pred, p=2)
            return residual_norm ** 2

        self.components["physics"] = LossComponent(
            name="Physics",
            weight=self.weights["physics"],
            compute_fn=compute,
            description=f"Enforce PDE laws ({domain})"
        )

    def add_boundary_loss(
        self,
        lower_bounds: Optional[torch.Tensor] = None,
        upper_bounds: Optional[torch.Tensor] = None
    ):
        """Loss_Boundary: Penalize predictions outside feasible range."""
        def compute(pred):
            if pred is None: return torch.tensor(0.0)
            # If bounds aren't passed, assume pred is the boundary residual
            if lower_bounds is None and upper_bounds is None:
                return (pred ** 2).mean()
            
            loss = torch.tensor(0.0, device=pred.device)
            if lower_bounds is not None:
                lb = lower_bounds.to(pred.device)
                loss += (torch.relu(lb - pred) ** 2).mean()
            if upper_bounds is not None:
                ub = upper_bounds.to(pred.device)
                loss += (torch.relu(pred - ub) ** 2).mean()
            return loss

        self.components["boundary"] = LossComponent(
            name="Boundary",
            weight=self.weights["boundary"],
            compute_fn=compute,
            description="Enforce physical constraints"
        )

    def add_biology_loss(
        self,
        constraints: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Loss_Biology: Genetic code, protein folding, metabolic rules."""
        def compute(pred):
            if pred is None: return torch.tensor(0.0)
            loss = torch.tensor(0.0, device=pred.device)
            c = constraints or {}
            # Codon optimization penalty
            if "codon_rarity" in c:
                loss += c["codon_rarity"].mean()
            # Protein stability (ΔG)
            if "protein_stability" in c:
                stability = c["protein_stability"]
                loss += torch.relu(-stability).mean()  # Penalize unstable (ΔG > 0)
            # Metabolic burden
            if "metabolic_burden" in c:
                burden = c["metabolic_burden"]
                loss += torch.relu(burden - 0.5).mean()  # Penalize >50% burden
                
            # If no constraints dict, assume pred is the bio penalty directly
            if not c:
                loss += pred.mean()
            return loss if loss > 0 else torch.tensor(0.0, device=pred.device)

        self.components["biology"] = LossComponent(
            name="Biology",
            weight=self.weights["biology"],
            compute_fn=compute,
            description="Enforce genetic/cellular rules"
        )

    def add_ecology_loss(
        self,
        escape_prob: float = 0.001,
        impact_threshold: float = 0.1
    ):
        """Loss_Ecology: Containment, invasiveness, ecosystem harm."""
        def compute(pred):
            if pred is None: return torch.tensor(0.0)
            # P_escape × (Impact_ecological + Impact_economic)
            ecological_impact = torch.clamp(pred, 0, 1)
            total_loss = escape_prob * ecological_impact
            if (total_loss > impact_threshold).any():
                penalty = (total_loss - impact_threshold) ** 2
                return penalty.mean()
            return torch.tensor(0.0)

        self.components["ecology"] = LossComponent(
            name="Ecology",
            weight=self.weights["ecology"],
            compute_fn=compute,
            description="Prevent ecological harm/invasiveness"
        )

    def add_economics_loss(
        self,
        manufacturing_cost: float = 0.0,
        operating_cost: float = 0.0,
        budget: float = 1.0
    ):
        """Loss_Economics: Ensure manufacturing viability."""
        def compute(pred):
            if pred is None: return torch.tensor(0.0)
            # Allow pred to be the operating cost if dynamic
            total_cost = manufacturing_cost + pred
            cost_ratio = total_cost / budget if budget > 0 else float('inf')
            # Penalize if over budget
            overbudget = torch.relu(cost_ratio - 1.0) ** 2
            return overbudget.mean()

        self.components["economics"] = LossComponent(
            name="Economics",
            weight=self.weights["economics"],
            compute_fn=compute,
            description="Ensure economic viability"
        )

    def add_safety_loss(
        self,
        toxicity_score: float = 0.0,
        pathogenicity: float = 0.0,
        allergenicity: float = 0.0
    ):
        """Loss_Safety: Toxicity, pathogenicity, allergenicity."""
        def compute(pred):
            if pred is None: return torch.tensor(0.0)
            # Allow pred to be the toxicity_score
            exposure = 1.0
            loss = (
                pred * exposure +
                torch.tensor(pathogenicity) * exposure +
                torch.tensor(allergenicity) * exposure
            )
            return torch.relu(loss).mean()

        self.components["safety"] = LossComponent(
            name="Safety",
            weight=self.weights["safety"],
            compute_fn=compute,
            description="Enforce safety constraints"
        )

    # ────────────────────────────────────────────────────────────
    # COMPOSITE LOSS
    # ────────────────────────────────────────────────────────────

    def compute_total_loss(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of all components."""
        total = torch.tensor(0.0, dtype=torch.float32)
        for name, component in self.components.items():
            try:
                loss_val = component.compute_fn(inputs.get(name))
                weighted = self.weights[name] * loss_val
                total = total + weighted
            except Exception as e:
                print(f"Error in {name} loss: {e}")
                continue
        return total

    def adapt_weights(self, scenario: str):
        """Adaptively adjust weights based on failure patterns."""
        adaptations = {
            "too_creative": {
                "data": 1.5,     # Increase data weight
                "physics": 1.0,
            },
            "violates_physics": {
                "physics": 2.0,   # Increase physics weight
                "data": 0.8,
            },
            "ignores_extremes": {
                "boundary": 2.0,  # Increase boundary weight
                "physics": 1.2,
            },
            "biologically_impossible": {
                "biology": 2.5,   # Increase biology weight
                "data": 1.0,
            },
            "ecologically_unsafe": {
                "ecology": 2.5,   # Increase ecology weight
                "safety": 2.2,
            },
            "too_expensive": {
                "economics": 1.5, # Increase economics weight
                "data": 0.9,
            },
            "unsafe": {
                "safety": 3.0,    # Max safety weight
                "toxicity": 2.5,
            },
        }
        if scenario in adaptations:
            self.weights.update(adaptations[scenario])
            print(f"✅ Weights adapted for: {scenario}")

    def to_dict(self) -> Dict:
        """Export all components for serialization."""
        return {
            "components": {
                name: {
                    "weight": comp.weight,
                    "description": comp.description
                }
                for name, comp in self.components.items()
            },
            "current_weights": self.weights,
        }


# ────────────────────────────────────────────────────────────
# FACTORY FUNCTIONS
# ────────────────────────────────────────────────────────────

def create_biology_loss_generator() -> LossGenerator:
    """Pre-configured for biology simulations."""
    gen = LossGenerator()
    gen.add_biology_loss({})
    gen.add_safety_loss()
    gen.weights["biology"] = 2.0
    return gen


def create_physics_loss_generator() -> LossGenerator:
    """Pre-configured for physics/mechanics simulations."""
    gen = LossGenerator()
    gen.add_physics_loss("mechanics")
    gen.add_boundary_loss(torch.tensor(0.0), torch.tensor(1.0))
    gen.weights["physics"] = 2.0
    return gen


def create_cross_domain_loss_generator() -> LossGenerator:
    """Full 7-component loss for multi-domain designs."""
    gen = LossGenerator()
    gen.add_data_loss(torch.tensor(0.5))
    gen.add_physics_loss("multi-domain")
    gen.add_boundary_loss(torch.tensor(0.0), torch.tensor(1.0))
    gen.add_biology_loss({})
    gen.add_ecology_loss()
    gen.add_economics_loss(100, 50, 1000)
    gen.add_safety_loss()
    return gen


if __name__ == "__main__":
    # Test loss generator
    gen = create_cross_domain_loss_generator()
    print("✅ Loss Generator created with 7 components:")
    for name, comp in gen.components.items():
        print(f"  {name}: {comp.description} (w={comp.weight})")

    # Test weight adaptation
    gen.adapt_weights("ecologically_unsafe")
    print("\n✅ Weights after adaptation:")
    for name, w in gen.weights.items():
        print(f"  {name}: {w}")
