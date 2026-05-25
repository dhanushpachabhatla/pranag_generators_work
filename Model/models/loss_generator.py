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

    def add_data_loss(self, observed: torch.Tensor, weight_fn: Optional[Callable] = None):
        """Loss_Data: MSE to real observations, weighted by data quality."""
        def compute(pred):
            mse = (pred - observed) ** 2
            if weight_fn is not None:
                mse = mse * weight_fn()
            return mse.mean()

        self.components["data"] = LossComponent(
            name="Data",
            weight=self.weights["data"],
            compute_fn=compute,
            description="Match real-world observations"
        )

    def add_physics_loss(self, pde_residuals: torch.Tensor, domain: str = "general"):
        """Loss_Physics: PDE residual penalty for fundamental laws."""
        def compute(pred):
            residual_norm = torch.norm(pde_residuals, p=2)
            return residual_norm ** 2

        self.components["physics"] = LossComponent(
            name="Physics",
            weight=self.weights["physics"],
            compute_fn=compute,
            description=f"Enforce PDE laws ({domain})"
        )

    def add_boundary_loss(
        self,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor
    ):
        """Loss_Boundary: Penalize predictions outside feasible range."""
        def compute(pred):
            below = torch.relu(lower_bounds - pred) ** 2
            above = torch.relu(pred - upper_bounds) ** 2
            return (below + above).mean()

        self.components["boundary"] = LossComponent(
            name="Boundary",
            weight=self.weights["boundary"],
            compute_fn=compute,
            description="Enforce physical constraints"
        )

    def add_biology_loss(
        self,
        constraints: Dict[str, torch.Tensor]
    ):
        """Loss_Biology: Genetic code, protein folding, metabolic rules."""
        def compute(pred):
            loss = 0.0
            # Codon optimization penalty
            if "codon_rarity" in constraints:
                loss += constraints["codon_rarity"].mean()
            # Protein stability (ΔG)
            if "protein_stability" in constraints:
                stability = constraints["protein_stability"]
                loss += torch.relu(-stability).mean()  # Penalize unstable (ΔG > 0)
            # Metabolic burden
            if "metabolic_burden" in constraints:
                burden = constraints["metabolic_burden"]
                loss += torch.relu(burden - 0.5).mean()  # Penalize >50% burden
            return loss if loss > 0 else torch.tensor(0.0)

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
            # P_escape × (Impact_ecological + Impact_economic)
            ecological_impact = torch.clamp(torch.tensor(pred), 0, 1)
            total_loss = escape_prob * ecological_impact
            if total_loss > impact_threshold:
                penalty = (total_loss - impact_threshold) ** 2
                return penalty
            return torch.tensor(0.0)

        self.components["ecology"] = LossComponent(
            name="Ecology",
            weight=self.weights["ecology"],
            compute_fn=compute,
            description="Prevent ecological harm/invasiveness"
        )

    def add_economics_loss(
        self,
        manufacturing_cost: float,
        operating_cost: float,
        budget: float
    ):
        """Loss_Economics: Ensure manufacturing viability."""
        def compute(pred):
            total_cost = manufacturing_cost + operating_cost
            cost_ratio = total_cost / budget if budget > 0 else float('inf')
            # Penalize if over budget
            overbudget = torch.relu(torch.tensor(cost_ratio - 1.0)) ** 2
            return overbudget

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
            # Each violation multiplied by exposure
            exposure = 1.0
            loss = (
                torch.tensor(toxicity_score) * exposure +
                torch.tensor(pathogenicity) * exposure +
                torch.tensor(allergenicity) * exposure
            )
            return torch.relu(loss)

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
                print(f"⚠️  Error in {name} loss: {e}")
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
    gen.add_physics_loss(torch.tensor(0.0), "mechanics")
    gen.add_boundary_loss(torch.tensor(0.0), torch.tensor(1.0))
    gen.weights["physics"] = 2.0
    return gen


def create_cross_domain_loss_generator() -> LossGenerator:
    """Full 7-component loss for multi-domain designs."""
    gen = LossGenerator()
    gen.add_data_loss(torch.tensor(0.5))
    gen.add_physics_loss(torch.tensor(0.0), "multi-domain")
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
