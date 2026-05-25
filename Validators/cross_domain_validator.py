"""
cross_domain_validator.py  —  DIVYANSHU  (Task 2)
Sequential: Biology → Materials → Physics → Chemistry
If any domain fails → STOP and return failure reason.
"""

from dataclasses import dataclass, field
from typing import Optional
from validator import SimulationResult


DOMAIN_THRESHOLDS = {
    # Thresholds calibrated to the score distributions produced by the
    # surrogate models.  Each maps to a concrete SimulationResult field
    # via SCORE_ATTRIBUTE_MAP.  Set at ~60th-percentile of each distribution
    # so the top-performing designs pass while low-quality ones are filtered.
    "quantum":         0.55,   # maps to physics_score  (max≈0.63, mean≈0.48)
    "nuclear":         0.55,   # maps to physics_score
    "chemical":        0.65,   # maps to chemistry_score (max≈0.86, mean≈0.65)
    "materials":       0.55,   # maps to materials_score (max≈0.84, mean≈0.49)
    "molecular_bio":   0.70,   # maps to biology_score   (min≈0.82 → all pass)
    "cellular":        0.70,   # maps to biology_score
    "organismal":      0.65,   # maps to biology_score
    "ecological":      0.65,   # maps to biology_score
    "physics":         0.55,   # maps to physics_score
    "earth_planetary": 0.60,   # maps to overall score   (max≈0.80, mean≈0.65)
    "space":           0.55,   # maps to physics_score
    "human_social":    0.60,   # maps to overall score
    "economic":        0.55,   # maps to overall score
}

# Maps abstract domain names to concrete SimulationResult attributes.
# SimulationResult fields: design_id, score, biology_score,
#   materials_score, physics_score, chemistry_score
# These are reused across the 13 hierarchical domain checks.
SCORE_ATTRIBUTE_MAP = {
    "quantum":         "physics_score",     # quantum physics → physics
    "nuclear":         "physics_score",     # nuclear → physics
    "chemical":        "chemistry_score",   # chemical reactions
    "materials":       "materials_score",   # NOTE: plural (SimulationResult field)
    "molecular_bio":   "biology_score",     # DNA/RNA/proteins
    "cellular":        "biology_score",     # cell biology
    "organismal":      "biology_score",     # whole organism
    "ecological":      "biology_score",     # population/ecosystem
    "physics":         "physics_score",     # forces/energy
    "earth_planetary": "score",             # climate/soil → overall score
    "space":           "physics_score",     # orbital mechanics
    "human_social":    "score",             # health/agriculture → overall
    "economic":        "score",             # supply chains → overall
}
DOMAIN_ORDER = [
    "quantum", "nuclear", "chemical", "materials",
    "molecular_bio", "cellular", "organismal", "ecological",
    "physics", "earth_planetary", "space", "human_social", "economic"
]


@dataclass
class DomainCheck:
    domain:    str
    score:     float
    threshold: float
    passed:    bool
    reason:    Optional[str] = None


@dataclass
class CrossDomainResult:
    design_id:      str
    overall_passed: bool
    checks:         list = field(default_factory=list)
    failure_domain: Optional[str] = None
    failure_reason: Optional[str] = None
    domains_checked:int = 0

    def to_dict(self):
        return {
            "design_id":      self.design_id,
            "overall_passed": self.overall_passed,
            "failure_domain": self.failure_domain,
            "failure_reason": self.failure_reason,
            "domains_checked":self.domains_checked,
            "checks": [{
                "domain":    c.domain,
                "score":     round(c.score, 4),
                "threshold": c.threshold,
                "passed":    c.passed,
                "reason":    c.reason,
            } for c in self.checks],
        }


class CrossDomainValidator:
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or DOMAIN_THRESHOLDS

    def _check(self, sim: SimulationResult, domain: str) -> DomainCheck:
        thresh = self.thresholds.get(domain, 0.70)   # always defined first

        # Try mapped attribute first, then direct domain_score attribute
        mapped_attr = SCORE_ATTRIBUTE_MAP.get(domain, f"{domain}_score")
        score = getattr(sim, mapped_attr, None)
        if score is None:
            score = getattr(sim, f"{domain}_score", None)

        if score is None:
            return DomainCheck(domain=domain, score=0.0, threshold=thresh, passed=False,
                               reason=f"{domain.capitalize()} score not evaluated (domain not computed)")
        score = float(score)
        passed = score >= thresh
        return DomainCheck(
            domain    = domain,
            score     = score,
            threshold = thresh,
            passed    = passed,
            reason    = None if passed else
                        f"{domain.capitalize()} score {score:.4f} < min {thresh}",
        )

    def validate(self, sim: SimulationResult) -> CrossDomainResult:
        result = CrossDomainResult(design_id=sim.design_id, overall_passed=False)
        for domain in DOMAIN_ORDER:
            check = self._check(sim, domain)
            result.checks.append(check)
            result.domains_checked += 1
            if not check.passed:
                result.failure_domain = domain
                result.failure_reason = check.reason
                return result   # ← STOP immediately
        result.overall_passed = True
        return result

    def validate_batch(self, simulations: list) -> dict:
        results  = [self.validate(s) for s in simulations]
        passed   = [r for r in results if r.overall_passed]
        failed   = [r for r in results if not r.overall_passed]
        fail_dist= {d: 0 for d in DOMAIN_ORDER}
        for r in failed:
            if r.failure_domain:
                fail_dist[r.failure_domain] += 1

        return {
            "total":                len(results),
            "passed":               len(passed),
            "failed":               len(failed),
            "failure_distribution": fail_dist,
            "results":              [r.to_dict() for r in results],
        }


if __name__ == "__main__":
    from validator import load_mock_simulations
    sims   = load_mock_simulations(40)
    cdv    = CrossDomainValidator()
    report = cdv.validate_batch(sims)
    print(f"Cross-Domain → Passed: {report['passed']}/{report['total']}")
    print(f"Failure dist: {report['failure_distribution']}")
