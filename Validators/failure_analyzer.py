"""
failure_analyzer.py  —  DIVYANSHU  (Task 4)
Identifies WHY a design failed: Physics / Boundary / Data issue.
Output: % distribution + suggestions.
"""

from dataclasses import dataclass, field
from typing import Optional
from validator import SimulationResult


FAILURE_CATEGORIES = {
    "physics_issue":    {
        "label": "Physics Issue",
        "desc":  "Physics domain is a clear outlier — violates physical feasibility constraints",
        "suggestions": [
            "Retrain the physics PINN with a wider temperature/force range",
            "Reduce operating temperature and recalibrate force boundary conditions",
            "Apply finite element mesh refinement on the physics sub-model",
        ]
    },
    "boundary_issue":   {
        "label": "Materials Bottleneck",
        "desc":  "Materials domain is a clear outlier — structural or composition constraints not met",
        "suggestions": [
            "Improve materials score: it is the single weakest domain dragging overall viability down",
            "Review band-gap and density parameter ranges in the materials PINN",
            "Augment materials training data with mid-range conductivity samples",
        ]
    },
    "data_issue":       {
        "label": "Data Quality Issue",
        "desc":  "Chemistry or biology domain is anomalously low — likely noisy or insufficient training data",
        "suggestions": [
            "Augment training dataset with edge cases in the flagged domain",
            "Apply data smoothing and outlier removal to chemistry/biology inputs",
            "Increase simulation resolution for the underperforming domain",
        ]
    },
    "domain_mismatch":  {
        "label": "Multi-Domain Conflict",
        "desc":  "Two or more domains are simultaneously below their individual baselines",
        "suggestions": [
            "Re-run cross-domain optimisation — multiple domains are conflicting",
            "Apply multi-objective Pareto optimisation across failing domains",
            "Decouple domain training and retrain each PINN independently",
        ]
    },
    "threshold_breach": {
        "label": "Marginal Threshold Breach",
        "desc":  "Overall score is close to the 0.70 viability threshold — small improvement needed",
        "suggestions": [
            "Tune hyperparameters near the decision boundary",
            "Apply ensemble scoring to reduce score variance",
            "Target the single weakest sub-domain for a small targeted gain",
        ]
    },
}


@dataclass
class FailureAnalysis:
    design_id:           str
    score:               float
    failure_category:    str
    failure_label:       str
    failure_description: str
    severity:            str
    sub_scores:          dict = field(default_factory=dict)
    suggestions:         list = field(default_factory=list)
    confidence:          float = 0.0

    def to_dict(self):
        return {
            "design_id": self.design_id,
            "score":     round(self.score, 4),
            "failure":   {
                "category":    self.failure_category,
                "label":       self.failure_label,
                "description": self.failure_description,
                "severity":    self.severity,
                "confidence":  round(self.confidence, 3),
            },
            "sub_scores":  {k: round(v,4) for k,v in self.sub_scores.items()},
            "suggestions": self.suggestions,
        }


class FailureAnalyzer:
    PASS_THRESHOLD = 0.70   # Spec: viability < 0.7 → KILL
    OUTLIER_GAP    = 0.15   # domain is an outlier if it's > 0.15 below the design's own mean

    def _severity(self, score: float) -> str:
        if score >= 0.65: return "low"
        if score >= 0.55: return "medium"
        if score >= 0.40: return "high"
        return "critical"

    def _std(self, values: list) -> float:
        if not values: return 0.0
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    def _classify(self, sub: dict, overall: float):
        """
        Per-design outlier classification.
        A domain is an 'outlier' only when it sits > OUTLIER_GAP below
        this particular design's own mean score — not a global std cutoff.
        Returns (category, confidence).
        """
        worst   = min(sub, key=sub.get)
        scores  = list(sub.values())
        mean_sc = sum(scores) / len(scores)

        # 1. Threshold breach — overall is just below the 0.70 bar
        if overall >= 0.62:
            return "threshold_breach", 0.88

        # 2. Outlier detection: which domains are significantly below
        #    THIS design's own average (not a global threshold)?
        outliers = [d for d, s in sub.items() if (mean_sc - s) > self.OUTLIER_GAP]

        if len(outliers) == 1:
            bottleneck = outliers[0]
            if bottleneck == "physics":
                return "physics_issue",  0.90
            if bottleneck == "materials":
                return "boundary_issue", 0.88
            # chemistry or biology anomaly
            return "data_issue", 0.82

        if len(outliers) >= 2:
            # Genuine multi-domain conflict — two or more domains are outliers
            return "domain_mismatch", 0.80

        # 3. No single outlier: all domains are close together but uniformly low
        if mean_sc < 0.52:
            return "data_issue", 0.72

        # 4. Soft bottleneck — no outlier, but one domain is still the weakest
        if worst == "physics":
            return "physics_issue",  0.70
        if worst == "materials":
            return "boundary_issue", 0.68
        if worst in ("chemistry", "biology"):
            return "data_issue", 0.65

        return "physics_issue", 0.60

    def analyze(self, sim: SimulationResult) -> Optional[FailureAnalysis]:
        if sim.score >= self.PASS_THRESHOLD:
            return None

        sub = {
            "biology":   sim.biology_score,
            "materials": sim.materials_score,
            "physics":   sim.physics_score,
            "chemistry": sim.chemistry_score,
        }
        worst     = min(sub, key=sub.get)
        min_score = sub[worst]

        cat, conf = self._classify(sub, sim.score)

        # Build domain-aware suggestions
        cat_suggestions = list(FAILURE_CATEGORIES[cat]["suggestions"])
        domain_tip = f"Bottleneck domain: {worst} (score: {min_score:.3f}) — target this domain first"
        suggestions = cat_suggestions + [domain_tip]
        if cat == "threshold_breach":
            gap = self.PASS_THRESHOLD - sim.score
            suggestions.insert(0, f"Score {sim.score:.3f} is only {gap:.3f} below threshold — small gain in {worst} may be enough")

        return FailureAnalysis(
            design_id           = sim.design_id,
            score               = sim.score,
            failure_category    = cat,
            failure_label       = FAILURE_CATEGORIES[cat]["label"],
            failure_description = FAILURE_CATEGORIES[cat]["desc"],
            severity            = self._severity(sim.score),
            sub_scores          = sub,
            suggestions         = suggestions,
            confidence          = conf,
        )

    def analyze_batch(self, simulations: list) -> dict:
        analyses = [a for s in simulations if (a := self.analyze(s))]
        total    = len(analyses)
        dist     = {k: 0 for k in FAILURE_CATEGORIES}
        for a in analyses:
            dist[a.failure_category] += 1
        dist_pct = {k: round(v/total*100, 1) if total else 0 for k, v in dist.items()}

        seen = set()
        top_suggestions = []
        for a in analyses:
            for s in a.suggestions:
                if s not in seen:
                    seen.add(s)
                    top_suggestions.append(s)
            if len(top_suggestions) >= 8:
                break

        return {
            "total_failures": total,
            "distribution":   dist_pct,
            "severity_breakdown": {
                "critical": sum(1 for a in analyses if a.severity == "critical"),
                "high":     sum(1 for a in analyses if a.severity == "high"),
                "medium":   sum(1 for a in analyses if a.severity == "medium"),
                "low":      sum(1 for a in analyses if a.severity == "low"),
            },
            "top_suggestions": top_suggestions,
            "analyses":        [a.to_dict() for a in analyses],
        }


if __name__ == "__main__":
    import json
    from validator import load_mock_simulations, Validator
    sims = load_mock_simulations(40)
    v    = Validator()
    v.validate_batch(sims)
    failed = [s for s, r in zip(sims, v.results) if not r.passed]
    fa     = FailureAnalyzer()
    report = fa.analyze_batch(failed)
    print(f"Failures: {report['total_failures']}")
    print(f"Distribution: {report['distribution']}")
    print(json.dumps(report['severity_breakdown'], indent=2))
