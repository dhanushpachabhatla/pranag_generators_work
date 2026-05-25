"""
domain_losses.py — Extended PRANA-G Loss Components (Tasks 3–6)
================================================================
Implements the four missing loss components from the 7-component spec:

    L_total = λ₁L_data + λ₂L_physics + λ₃L_boundary
            + λ₄L_biology + λ₅L_ecology + λ₆L_economics + λ₇L_safety

Each class is self-contained and can be passed to BasePINN.total_loss()
via the `extended_losses` parameter:

    extended = {
        "biology":   (bio_loss.compute(**bio_inputs),   1.8),
        "ecology":   (eco_loss.compute(**eco_inputs),   1.6),
        "economics": (econ_loss.compute(**econ_inputs), 0.8),
        "safety":    (safe_loss.compute(**safe_inputs), 2.0),
    }
    total, breakdown = model.total_loss(..., extended_losses=extended)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


# ═══════════════════════════════════════════════════════════════════════
# Task 3 — Biology Loss
# L_biology = E_folding + M_burden + G_constraint
# ═══════════════════════════════════════════════════════════════════════

class BiologyLoss:
    """
    Penalises biologically impossible designs.

    Components:
        E_folding   : Protein folding instability (ΔG > 0 → unstable)
        M_burden    : Metabolic burden penalty (>50% capacity is harmful)
        G_constraint: Gene constraint violations (codon rarity, GC content)

    All inputs are normalised tensors in [0, 1] unless noted.
    A score of 0.0 means no penalty; higher values indicate worse designs.
    """

    def __init__(
        self,
        folding_weight: float = 1.0,
        burden_weight:  float = 1.0,
        constraint_weight: float = 1.0,
        max_metabolic_burden: float = 0.5,   # 50% capacity threshold
        max_codon_rarity: float = 0.3,        # 30% rare codons is problematic
    ):
        self.folding_weight    = folding_weight
        self.burden_weight     = burden_weight
        self.constraint_weight = constraint_weight
        self.max_burden        = max_metabolic_burden
        self.max_codon_rarity  = max_codon_rarity

    def folding_penalty(self, folding_energy: torch.Tensor) -> torch.Tensor:
        """
        E_folding: penalise positive ΔG (unstable protein).
        folding_energy: normalised ΔG values; positive = unstable.
        """
        return torch.relu(folding_energy).mean()

    def metabolic_burden_penalty(self, metabolic_burden: torch.Tensor) -> torch.Tensor:
        """
        M_burden: penalise burden > threshold (cell overloaded).
        metabolic_burden: fraction of cell capacity consumed [0, 1].
        """
        excess = torch.relu(metabolic_burden - self.max_burden)
        return (excess ** 2).mean()

    def gene_constraint_penalty(
        self,
        codon_rarity: Optional[torch.Tensor] = None,
        gc_content:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        G_constraint: penalise rare codons and out-of-range GC content.
        codon_rarity : fraction of rare codons [0, 1]; high → expression issues
        gc_content   : GC fraction [0, 1]; valid window is [0.40, 0.65]
        """
        loss = torch.tensor(0.0)

        if codon_rarity is not None:
            loss = loss + torch.relu(codon_rarity - self.max_codon_rarity).mean()

        if gc_content is not None:
            gc_low  = torch.relu(0.40 - gc_content)
            gc_high = torch.relu(gc_content - 0.65)
            loss = loss + (gc_low + gc_high).mean()

        return loss

    def compute(
        self,
        folding_energy:   Optional[torch.Tensor] = None,
        metabolic_burden: Optional[torch.Tensor] = None,
        codon_rarity:     Optional[torch.Tensor] = None,
        gc_content:       Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L_biology = E_folding + M_burden + G_constraint
        Returns scalar loss tensor. Zero if no inputs provided.
        """
        total = torch.tensor(0.0)

        if folding_energy is not None:
            total = total + self.folding_weight * self.folding_penalty(folding_energy)

        if metabolic_burden is not None:
            total = total + self.burden_weight * self.metabolic_burden_penalty(metabolic_burden)

        if codon_rarity is not None or gc_content is not None:
            total = total + self.constraint_weight * self.gene_constraint_penalty(
                codon_rarity, gc_content
            )

        return total

    def validate(
        self,
        folding_energy: Optional[torch.Tensor] = None,
        metabolic_burden: Optional[torch.Tensor] = None,
        codon_rarity: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Returns penalty breakdown for inspection."""
        return {
            "folding_penalty":  self.folding_penalty(folding_energy).item()
                                if folding_energy is not None else 0.0,
            "burden_penalty":   self.metabolic_burden_penalty(metabolic_burden).item()
                                if metabolic_burden is not None else 0.0,
            "gene_penalty":     self.gene_constraint_penalty(codon_rarity).item()
                                if codon_rarity is not None else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════
# Task 4 — Ecology Loss
# L_ecology = P_escape × Impact
# ═══════════════════════════════════════════════════════════════════════

class EcologyLoss:
    """
    Prevents unsafe ecological escape risk.

    Formulation:
        L_ecology = P_escape × (α·ecological_impact + β·economic_impact)

    Components:
        P_escape          : probability of organism escaping containment [0, 1]
        ecological_impact : severity of ecological disruption if escaped [0, 1]
        economic_impact   : downstream agricultural/economic damage [0, 1]

    Penalty is zero when P_escape is below safe_escape_threshold.
    """

    def __init__(
        self,
        safe_escape_threshold: float = 0.01,   # <1% escape probability is safe
        ecological_weight: float = 0.7,
        economic_weight:   float = 0.3,
        invasion_factor:   float = 2.0,         # multiplier for invasive species risk
    ):
        self.safe_escape_threshold = safe_escape_threshold
        self.ecological_weight     = ecological_weight
        self.economic_weight       = economic_weight
        self.invasion_factor       = invasion_factor

    def escape_penalty(
        self,
        p_escape:           torch.Tensor,
        ecological_impact:  torch.Tensor,
        economic_impact:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Core penalty: P_escape × weighted impact.
        Only fires when escape probability exceeds safe threshold.
        """
        p_above_safe = torch.relu(p_escape - self.safe_escape_threshold)
        combined_impact = self.ecological_weight * ecological_impact
        if economic_impact is not None:
            combined_impact = combined_impact + self.economic_weight * economic_impact
        return (p_above_safe * combined_impact).mean()

    def invasion_risk(self, invasiveness_score: torch.Tensor) -> torch.Tensor:
        """
        Additional penalty for designs with high invasiveness potential.
        invasiveness_score: probability-like score [0, 1] of becoming invasive.
        """
        return (self.invasion_factor * torch.relu(invasiveness_score - 0.3) ** 2).mean()

    def compute(
        self,
        p_escape:           Optional[torch.Tensor] = None,
        ecological_impact:  Optional[torch.Tensor] = None,
        economic_impact:    Optional[torch.Tensor] = None,
        invasiveness_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L_ecology = P_escape × Impact + invasion_risk
        Returns scalar loss tensor. Zero if no inputs provided.
        """
        total = torch.tensor(0.0)

        if p_escape is not None and ecological_impact is not None:
            total = total + self.escape_penalty(p_escape, ecological_impact, economic_impact)

        if invasiveness_score is not None:
            total = total + self.invasion_risk(invasiveness_score)

        return total

    def normalised_penalty(
        self,
        p_escape: torch.Tensor,
        ecological_impact: torch.Tensor,
    ) -> torch.Tensor:
        """Returns penalty normalised to [0, 1] for reporting."""
        raw = self.escape_penalty(p_escape, ecological_impact)
        return torch.clamp(raw, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Task 5 — Economics Loss
# L_economics = max(0, Cost_mfg + Cost_op − Budget)
# ═══════════════════════════════════════════════════════════════════════

class EconomicsLoss:
    """
    Prevents economically infeasible designs.

    Formulation:
        L_economics = max(0, Cost_manufacturing + Cost_operation − Budget)²

    Supports scalar costs (single design) or batch tensors (multiple designs).
    All costs and budget should be in the same currency unit (e.g. USD×10³).
    """

    def __init__(
        self,
        budget: float = 1000.0,           # default budget per design
        scale: float = 1e-3,              # normalise large costs
        quadratic_penalty: bool = True,   # squared penalty for smoother gradient
    ):
        self.budget            = budget
        self.scale             = scale
        self.quadratic_penalty = quadratic_penalty

    def cost_penalty(
        self,
        manufacturing_cost: torch.Tensor,
        operation_cost:     torch.Tensor,
        budget:             Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        max(0, total_cost − budget)  or  max(0, total_cost − budget)²
        """
        b = budget if budget is not None else torch.tensor(self.budget)
        total_cost = manufacturing_cost + operation_cost
        overbudget = torch.relu(total_cost - b) * self.scale
        if self.quadratic_penalty:
            return (overbudget ** 2).mean()
        return overbudget.mean()

    def compute(
        self,
        manufacturing_cost: Optional[torch.Tensor] = None,
        operation_cost:     Optional[torch.Tensor] = None,
        budget:             Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L_economics = max(0, Cost_mfg + Cost_op − Budget)²
        Returns scalar loss tensor. Zero if no inputs provided.
        """
        if manufacturing_cost is None or operation_cost is None:
            return torch.tensor(0.0)

        return self.cost_penalty(manufacturing_cost, operation_cost, budget)

    def budget_utilisation(
        self,
        manufacturing_cost: torch.Tensor,
        operation_cost:     torch.Tensor,
    ) -> torch.Tensor:
        """Returns cost-to-budget ratio for reporting [0, ∞)."""
        total = manufacturing_cost + operation_cost
        return total / max(self.budget, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Task 6 — Safety Loss
# L_safety = Toxicity + Pathogenicity + Allergenicity
# ═══════════════════════════════════════════════════════════════════════

class SafetyLoss:
    """
    Rejects unsafe biological outputs.

    Formulation:
        L_safety = w_t·Toxicity + w_p·Pathogenicity + w_a·Allergenicity

    All three components are normalised scores in [0, 1] where 0 = safe.
    Penalty fires only when a score exceeds its safe threshold.

    Database integration: replace placeholder `score_from_db()` with
    real TOXNET / RegulationDB / allergen database lookups.
    """

    SAFE_TOXICITY      = 0.1   # LD50-based normalised threshold
    SAFE_PATHOGENICITY = 0.05  # BSL-1 equivalent upper bound
    SAFE_ALLERGENICITY = 0.15  # WHO/FAO allergenicity threshold

    def __init__(
        self,
        toxicity_weight:      float = 1.0,
        pathogenicity_weight: float = 1.5,  # higher — containment risk
        allergenicity_weight: float = 1.0,
    ):
        self.w_tox   = toxicity_weight
        self.w_path  = pathogenicity_weight
        self.w_allerg = allergenicity_weight

    def toxicity_penalty(self, toxicity_score: torch.Tensor) -> torch.Tensor:
        """Penalise toxicity above safe threshold."""
        return torch.relu(toxicity_score - self.SAFE_TOXICITY).mean()

    def pathogenicity_penalty(self, pathogenicity_score: torch.Tensor) -> torch.Tensor:
        """Penalise pathogenicity above safe threshold (strict — biosafety)."""
        return torch.relu(pathogenicity_score - self.SAFE_PATHOGENICITY).mean()

    def allergenicity_penalty(self, allergenicity_score: torch.Tensor) -> torch.Tensor:
        """Penalise allergenicity above WHO/FAO threshold."""
        return torch.relu(allergenicity_score - self.SAFE_ALLERGENICITY).mean()

    def compute(
        self,
        toxicity_score:      Optional[torch.Tensor] = None,
        pathogenicity_score: Optional[torch.Tensor] = None,
        allergenicity_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L_safety = Toxicity + Pathogenicity + Allergenicity
        Returns scalar loss tensor. Zero if no inputs provided.
        """
        total = torch.tensor(0.0)

        if toxicity_score is not None:
            total = total + self.w_tox * self.toxicity_penalty(toxicity_score)

        if pathogenicity_score is not None:
            total = total + self.w_path * self.pathogenicity_penalty(pathogenicity_score)

        if allergenicity_score is not None:
            total = total + self.w_allerg * self.allergenicity_penalty(allergenicity_score)

        return total

    def is_safe(
        self,
        toxicity_score:      float,
        pathogenicity_score: float,
        allergenicity_score: float,
    ) -> bool:
        """Hard rejection: returns False if any score exceeds its threshold."""
        return (
            toxicity_score      <= self.SAFE_TOXICITY
            and pathogenicity_score <= self.SAFE_PATHOGENICITY
            and allergenicity_score <= self.SAFE_ALLERGENICITY
        )

    def score_from_db(self, sequence_id: str) -> Dict[str, float]:
        """
        Placeholder database integration layer.
        Replace with real TOXNET / RegulationDB / allergen database calls.
        Returns normalised [0, 1] scores for a given biological sequence.
        """
        # TODO: integrate with TOXNET, PubChem, AllerHunter, etc.
        return {
            "toxicity":      0.0,
            "pathogenicity": 0.0,
            "allergenicity": 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════
# Convenience factory
# ═══════════════════════════════════════════════════════════════════════

def default_extended_losses(
    bio_inputs:  Optional[Dict] = None,
    eco_inputs:  Optional[Dict] = None,
    econ_inputs: Optional[Dict] = None,
    safe_inputs: Optional[Dict] = None,
) -> Dict[str, tuple]:
    """
    Build the extended_losses dict for BasePINN.total_loss() from raw inputs.

    Usage:
        ext = default_extended_losses(
            bio_inputs  = {"folding_energy": delta_g_tensor},
            eco_inputs  = {"p_escape": escape_prob, "ecological_impact": impact},
            econ_inputs = {"manufacturing_cost": mfg_cost, "operation_cost": op_cost},
            safe_inputs = {"toxicity_score": tox},
        )
        loss, breakdown = model.total_loss(..., extended_losses=ext)
    """
    bio_loss  = BiologyLoss()
    eco_loss  = EcologyLoss()
    econ_loss = EconomicsLoss()
    safe_loss = SafetyLoss()

    result = {}
    if bio_inputs:
        result["biology"]   = (bio_loss.compute(**bio_inputs),   1.8)
    if eco_inputs:
        result["ecology"]   = (eco_loss.compute(**eco_inputs),   1.6)
    if econ_inputs:
        result["economics"] = (econ_loss.compute(**econ_inputs), 0.8)
    if safe_inputs:
        result["safety"]    = (safe_loss.compute(**safe_inputs), 2.0)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Unit tests / sample penalty validation
# ═══════════════════════════════════════════════════════════════════════

def _run_tests():
    print("=" * 60)
    print("  Domain Loss Unit Tests")
    print("=" * 60)

    # ── Biology Loss ──────────────────────────────────────────────
    print("\n[BiologyLoss]")
    bio = BiologyLoss()

    # Stable protein (ΔG < 0) → no penalty
    stable   = torch.tensor([-1.5, -0.8, -2.0])
    unstable = torch.tensor([ 0.5,  1.2,  0.0])
    assert bio.folding_penalty(stable).item() == 0.0,   "Stable protein should give 0 penalty"
    assert bio.folding_penalty(unstable).item() > 0.0,  "Unstable protein should give > 0 penalty"
    print(f"  folding_penalty(stable)   = {bio.folding_penalty(stable).item():.4f}  [expected 0.0]")
    print(f"  folding_penalty(unstable) = {bio.folding_penalty(unstable).item():.4f}  [expected > 0]")

    # Metabolic burden at 30% (safe) vs 80% (overloaded)
    safe_burden  = torch.tensor([0.3, 0.25, 0.4])
    heavy_burden = torch.tensor([0.8, 0.9,  0.7])
    assert bio.metabolic_burden_penalty(safe_burden).item() == 0.0,  "Safe burden: no penalty"
    assert bio.metabolic_burden_penalty(heavy_burden).item() > 0.0,  "Heavy burden: penalty"
    print(f"  burden_penalty(safe=0.3)  = {bio.metabolic_burden_penalty(safe_burden).item():.4f}  [expected 0.0]")
    print(f"  burden_penalty(high=0.8)  = {bio.metabolic_burden_penalty(heavy_burden).item():.4f}  [expected > 0]")

    # compute() combines all components
    total_bio = bio.compute(folding_energy=unstable, metabolic_burden=heavy_burden)
    assert total_bio.item() > 0.0
    print(f"  compute(unstable+heavy)   = {total_bio.item():.4f}  [expected > 0]")

    # ── Ecology Loss ──────────────────────────────────────────────
    print("\n[EcologyLoss]")
    eco = EcologyLoss()

    safe_escape = torch.tensor([0.005, 0.008])   # below 1% threshold
    high_escape = torch.tensor([0.05,  0.10])    # above threshold
    impact      = torch.tensor([0.8,   0.9])

    assert eco.escape_penalty(safe_escape, impact).item() == 0.0,  "Safe escape: no penalty"
    assert eco.escape_penalty(high_escape, impact).item() > 0.0,   "High escape: penalty"
    print(f"  escape_penalty(safe)      = {eco.escape_penalty(safe_escape, impact).item():.6f}  [expected 0.0]")
    print(f"  escape_penalty(high)      = {eco.escape_penalty(high_escape, impact).item():.6f}  [expected > 0]")

    norm_penalty = eco.normalised_penalty(high_escape, impact)
    assert 0.0 <= norm_penalty.item() <= 1.0
    print(f"  normalised_penalty        = {norm_penalty.item():.4f}  [expected in [0,1]]")

    # ── Economics Loss ────────────────────────────────────────────
    print("\n[EconomicsLoss]")
    econ = EconomicsLoss(budget=1000.0)

    mfg_ok  = torch.tensor([300.0, 400.0])
    op_ok   = torch.tensor([200.0, 300.0])    # total: 500, 700 — under budget
    mfg_over= torch.tensor([600.0, 800.0])
    op_over = torch.tensor([600.0, 400.0])    # total: 1200, 1200 — over budget

    assert econ.cost_penalty(mfg_ok, op_ok).item() == 0.0,   "Under budget: no penalty"
    assert econ.cost_penalty(mfg_over, op_over).item() > 0.0, "Over budget: penalty"
    print(f"  cost_penalty(under budget) = {econ.cost_penalty(mfg_ok, op_ok).item():.6f}  [expected 0.0]")
    print(f"  cost_penalty(over budget)  = {econ.cost_penalty(mfg_over, op_over).item():.6f}  [expected > 0]")

    # ── Safety Loss ───────────────────────────────────────────────
    print("\n[SafetyLoss]")
    safe = SafetyLoss()

    safe_tox   = torch.tensor([0.05, 0.08])  # below 0.10 threshold
    danger_tox = torch.tensor([0.30, 0.50])  # above threshold

    assert safe.toxicity_penalty(safe_tox).item()   == 0.0, "Safe tox: no penalty"
    assert safe.toxicity_penalty(danger_tox).item() > 0.0,  "Dangerous tox: penalty"
    print(f"  toxicity_penalty(safe)     = {safe.toxicity_penalty(safe_tox).item():.4f}  [expected 0.0]")
    print(f"  toxicity_penalty(danger)   = {safe.toxicity_penalty(danger_tox).item():.4f}  [expected > 0]")

    # is_safe() hard rejection
    assert safe.is_safe(0.05, 0.02, 0.10) is True,  "Should be safe"
    assert safe.is_safe(0.20, 0.02, 0.10) is False, "Toxic: should reject"
    print(f"  is_safe(tox=0.05, ...)     = {safe.is_safe(0.05, 0.02, 0.10)}  [expected True]")
    print(f"  is_safe(tox=0.20, ...)     = {safe.is_safe(0.20, 0.02, 0.10)}  [expected False]")

    # ── Factory ───────────────────────────────────────────────────
    print("\n[default_extended_losses factory]")
    ext = default_extended_losses(
        bio_inputs  = {"folding_energy": unstable},
        eco_inputs  = {"p_escape": high_escape, "ecological_impact": impact},
        econ_inputs = {"manufacturing_cost": mfg_over, "operation_cost": op_over},
        safe_inputs = {"toxicity_score": danger_tox},
    )
    assert set(ext.keys()) == {"biology", "ecology", "economics", "safety"}
    for name, (loss_val, weight) in ext.items():
        print(f"  {name:<12} loss={loss_val.item():.4f}  weight={weight}")

    print("\n  All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
