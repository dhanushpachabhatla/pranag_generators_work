import json
import os
import sys

# Ensure Model directories are in path to import registry
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model', 'models')))
from training.pinn_factory import PINNFactory


class DAGRouter:
    """
    Parses a spec.json and uses an LLM (or mock rules) to determine the Execution DAG.
    It decides WHICH surrogates to activate, in WHAT order, and WHAT weights to apply.
    """
    
    @property
    def LLM_SYSTEM_PROMPT(self):
        return f"""
You are the Master Orchestrator for the PRANA-G Physical Simulation Pipeline.
You will receive a user specification for a biological or material design.
Your job is to output a JSON defining the Directed Acyclic Graph (DAG) for surrogate execution.

Available Surrogates & Strict Input Schemas:
(Note: `intrinsic_property_1` represents the entity's normalized Capability/Baseline Performance like Yield or Strength. `intrinsic_property_2` represents the entity's normalized Resilience/Stability like Drought Tolerance or Weathering Resistance).
{{
  "arrhenius": {{"inputs": ["temperature", "intrinsic_property_1", "intrinsic_property_2", "boundary_rate", "initial_rate"]}},
  "biology": {{"inputs": ["time", "intrinsic_property_1", "intrinsic_property_2", "boundary_biomass", "initial_biomass"]}},
  "darcy": {{"inputs": ["space_x", "space_y", "intrinsic_property_1", "intrinsic_property_2", "boundary_pressure", "initial_pressure"]}},
  "economics": {{"inputs": ["time", "intrinsic_property_1", "intrinsic_property_2", "boundary_cost", "initial_cost"]}},
  "heat": {{"inputs": ["time", "space", "intrinsic_property_1", "intrinsic_property_2", "boundary_temperature", "initial_temperature"]}},
  "logistic": {{"inputs": ["time", "intrinsic_property_1", "intrinsic_property_2", "boundary_growth", "initial_growth"]}},
  "maxwell": {{"inputs": ["time", "space_x", "space_y", "space_z", "intrinsic_property_1", "intrinsic_property_2", "boundary_em_field", "initial_em_field"]}},
  "navier_stokes": {{"inputs": ["time", "space_x", "space_y", "intrinsic_property_1", "intrinsic_property_2", "boundary_velocity", "initial_velocity"]}},
  "orbital": {{"inputs": ["time", "intrinsic_property_1", "intrinsic_property_2", "boundary_position", "initial_position"]}},
  "phase_change": {{"inputs": ["time", "space", "intrinsic_property_1", "intrinsic_property_2", "boundary_state", "initial_state"]}},
  "radiation": {{"inputs": ["time", "space", "intrinsic_property_1", "intrinsic_property_2", "boundary_intensity", "initial_intensity"]}},
  "reaction_diffusion": {{"inputs": ["time", "space_x", "space_y", "intrinsic_property_1", "intrinsic_property_2", "boundary_concentration", "initial_concentration"]}},
  "schrodinger": {{"inputs": ["time", "space", "intrinsic_property_1", "intrinsic_property_2", "boundary_wavefunction", "initial_wavefunction"]}},
  "solid_mechanics": {{"inputs": ["space_x", "space_y", "intrinsic_property_1", "intrinsic_property_2", "boundary_displacement", "initial_displacement"]}},
  "stress": {{"inputs": ["time", "intrinsic_property_1", "intrinsic_property_2", "boundary_stress", "initial_stress"]}}
}}

CRITICAL RULE: You MUST choose ONLY from the EXACT Available Surrogates listed above. 
DO NOT INVENT new domain names (like 'osmotic_stress' or 'growth'). 
If a required biological or physical phenomenon isn't listed, map it to the closest existing surrogate (e.g., use 'stress' for any type of stress, or 'biology' for generic biological growth).
You MUST strictly use the exact input names specified in the schema for the chosen model. DO NOT invent input names.
DYNAMIC COUPLING RULE: If a node's physics depends on the output of a previous node (e.g., biology depends on temperature), you MUST replace `intrinsic_property_1` or `intrinsic_property_2` in its input array with the exact `output_maps_to` string from the previous node. Do not replace time, space, boundary, or initial variables.

Output Format:
You MUST return a JSON with:
- "execution_chain": A list of dictionaries defining the strict execution order. Each dict must have:
    - "model": The name of the surrogate to activate (from the schema above).
    - "inputs": A list of physical parameters this model requires (MUST EXACTLY MATCH the schema inputs for that model).
    - "output_maps_to": The physical parameter name that this model's prediction will overwrite for subsequent models (e.g., "temperature", "biomass", "ph").
    - "target_value": The numerical target extracted from the user's specification (e.g., 45.0 for temperature, 90.0 for maturity).
        If the user specification does not explicitly provide a numerical value, you MUST use your
        scientific knowledge to infer a realistic standard physical baseline rather than outputting
        an arbitrary placeholder. Use these standard references depending on the model:
          - biology / logistic (target_biomass, boundary_growth): 100.0 (full maturity / 100% yield capacity)
          - arrhenius (target_rate): 0.1 if optimization_goal implies decay/stability (minimize),
            or 10.0 if it implies catalysis/speed (maximize)
          - stress (target_stress): 1.0 (the pipeline normalizes maximum critical stress to 1.0)
        If the specification genuinely implies a target of exactly zero (e.g. "minimize stress to
        zero", "no water retention"), use 0.0 — that is a real value, not a missing one.
        You MUST add "target_value" to this node's "assumed_defaults" list whenever you used a
        standard baseline above instead of a number extracted directly from the specification.
    - "initial_value": The baseline or starting state extracted from the user's specification (e.g., 25.0 for standard room temp, 0.0 for zero stress).
        If not explicitly provided, use your scientific knowledge to infer a realistic standard
        baseline. Use these standard references:
          - heat / arrhenius (initial_temperature, boundary_temperature): 25.0 (standard room
            temperature in Celsius)
          - arrhenius (initial_ph): 7.0 (neutral baseline)
          - arrhenius (initial_rate): 1.0 (normalized baseline reaction rate)
          - biology / logistic (initial_biomass, initial_growth): 1.0 (seedling / initial culture —
            do NOT use 0.0, since growth equations often stall at exactly zero)
          - biology disease/infection state (initial_infected): 0.01 (a small initial outbreak)
          - stress / darcy (initial_stress, initial_pressure): 0.0 (no applied external load)
          - darcy boundary (boundary_pressure): 1.0 (standard atmospheric pressure baseline)
        DO NOT output arbitrary placeholders like -999.0 under any circumstance.
        You MUST add "initial_value" to this node's "assumed_defaults" list whenever you used a
        standard baseline above instead of a number extracted directly from the specification.
    - "optimization_goal": Must be "maximize", "minimize", or "target".
    - "assumed_defaults": A list of field names in THIS node (choose only from "target_value",
        "initial_value") that you filled using standard scientific baselines rather than reading
        directly from the user's specification. Use an empty list [] if every field was explicit.
- "weights": A dictionary assigning viability weights to the active models (must sum to 1.0).
- "target_entities": A list of 1-3 specific target subjects extracted from the specification (e.g., ["maize", "corn", "zea mays"]). If not explicitly given, derive from context.
- "semantic_keywords": A list of 5-8 EXACT synonyms, scientific names, and specific taxonomic identifiers for the requested entity. (e.g., if it's about corn, include ["corn", "maize", "zea mays", "sweet corn", "field corn"]). DO NOT use generic category terms like "crop", "grain", "plant", "biology", or "development" as they will pull in completely irrelevant false positives!
- "domain_filters": A list of 1-3 broad data domains to restrict the search to, chosen from exactly these options: ["biology", "chemistry", "environment", "physics", "materials", "economics"].

Example Output:
{{
    "execution_chain": [
        {{"model": "heat", "inputs": ["time", "space", "intrinsic_property_1", "intrinsic_property_2", "boundary_temperature", "initial_temperature"], "output_maps_to": "temperature", "target_value": 45.0, "initial_value": 25.0, "optimization_goal": "target", "assumed_defaults": []}},
        {{"model": "darcy", "inputs": ["space_x", "space_y", "intrinsic_property_1", "intrinsic_property_2", "boundary_pressure", "initial_pressure"], "output_maps_to": "water_retention", "target_value": 0.0, "initial_value": 0.0, "optimization_goal": "minimize", "assumed_defaults": ["initial_value"]}},
        {{"model": "logistic", "inputs": ["time", "intrinsic_property_1", "intrinsic_property_2", "boundary_growth", "initial_growth"], "output_maps_to": "biomass", "target_value": 1000.0, "initial_value": 10.0, "optimization_goal": "maximize", "assumed_defaults": []}}
    ],
    "weights": {{
        "heat": 0.25,
        "darcy": 0.25,
        "logistic": 0.50
    }},
    "target_entities": ["corn", "maize"],
    "semantic_keywords": ["corn", "maize", "zea mays", "sweet corn", "field corn"],
    "domain_filters": ["biology", "environment"]
}}
"""

    def __init__(self, use_mock=True):
        self.use_mock = use_mock

    def build_dag(self, spec_path: str) -> dict:
        """Reads the spec.json and returns the execution DAG."""
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Spec file not found: {spec_path}")
            
        with open(spec_path, 'r', encoding='utf-8') as f:
            spec = json.load(f)
            
        if self.use_mock:
            return self._mock_llm_routing(spec)
        else:
            return self._call_real_llm(spec)
            
    def _mock_llm_routing(self, spec: dict) -> dict:
        """
        Simulates the LLM's response by parsing the spec.json text.
        This allows the pipeline to run locally without API keys.
        """
        print("\n[!] WARNING: Falling back to Local Mock Router as API failed or was bypassed.")
        stresses = " ".join(spec.get("stress_conditions", [])).lower()
        targets = " ".join(spec.get("target_traits", [])).lower()
        combined_text = stresses + " " + targets
        
        chain = []
        weights = {}
        
        # Rule-based routing simulating the LLM's logic
        if "heat" in combined_text or "temperature" in combined_text:
            chain.append({"model": "heat", "inputs": ["time", "space", "intrinsic_property_1", "intrinsic_property_2", "boundary_temperature", "initial_temperature"], "output_maps_to": "temperature", "target_value": 45.0, "initial_value": 25.0, "optimization_goal": "target", "assumed_defaults": ["initial_value"]})
            weights["heat"] = 0.3
            
        if "drought" in combined_text or "flood" in combined_text or "disease" in combined_text:
            chain.append({"model": "darcy", "inputs": ["space_x", "space_y", "intrinsic_property_1", "intrinsic_property_2", "boundary_pressure", "initial_pressure"], "output_maps_to": "water_retention", "target_value": 0.0, "initial_value": 0.0, "optimization_goal": "minimize", "assumed_defaults": ["initial_value"]})
            weights["darcy"] = 0.4
            
        if "yield" in combined_text or "growth" in combined_text:
            chain.append({"model": "logistic", "inputs": ["time", "intrinsic_property_1", "intrinsic_property_2", "boundary_growth", "initial_growth"], "output_maps_to": "biomass", "target_value": 100.0, "initial_value": 1.0, "optimization_goal": "maximize", "assumed_defaults": ["initial_value"]})
            weights["logistic"] = 0.3
            
        if "salinity" in combined_text or "ph" in combined_text or "soil" in combined_text:
            chain.append({"model": "reaction_diffusion", "inputs": ["time", "space_x", "space_y", "intrinsic_property_1", "intrinsic_property_2", "boundary_concentration", "initial_concentration"], "output_maps_to": "chemical_concentration", "target_value": 7.0, "initial_value": 7.0, "optimization_goal": "target", "assumed_defaults": ["initial_value"]})
            weights["reaction_diffusion"] = 0.2
            
        if "finance" in combined_text or "cost" in combined_text or "price" in combined_text or "economy" in combined_text:
            chain.append({"model": "economics", "inputs": ["time", "intrinsic_property_1", "intrinsic_property_2", "boundary_cost", "initial_cost"], "output_maps_to": "cost", "target_value": 0.0, "initial_value": 100.0, "optimization_goal": "minimize", "assumed_defaults": ["initial_value"]})
            weights["economics"] = 0.3
            
        if "fluid" in combined_text or "velocity" in combined_text or "aerodynamics" in combined_text or "flow" in combined_text:
            chain.append({"model": "navier_stokes", "inputs": ["time", "space_x", "space_y", "intrinsic_property_1", "intrinsic_property_2", "boundary_velocity", "initial_velocity"], "output_maps_to": "velocity", "target_value": 10.0, "initial_value": 0.0, "optimization_goal": "target", "assumed_defaults": ["initial_value"]})
            weights["navier_stokes"] = 0.3
            
        if "structure" in combined_text or "elasticity" in combined_text or "deformation" in combined_text or "strain" in combined_text:
            chain.append({"model": "solid_mechanics", "inputs": ["space_x", "space_y", "intrinsic_property_1", "intrinsic_property_2", "boundary_displacement", "initial_displacement"], "output_maps_to": "displacement", "target_value": 0.0, "initial_value": 0.0, "optimization_goal": "minimize", "assumed_defaults": ["initial_value"]})
            weights["solid_mechanics"] = 0.3
            
        if "electric" in combined_text or "magnetic" in combined_text or "em wave" in combined_text or "plasma" in combined_text:
            chain.append({"model": "maxwell", "inputs": ["time", "space_x", "space_y", "space_z", "intrinsic_property_1", "intrinsic_property_2", "boundary_em_field", "initial_em_field"], "output_maps_to": "em_field", "target_value": 1.0, "initial_value": 0.0, "optimization_goal": "target", "assumed_defaults": ["initial_value"]})
            weights["maxwell"] = 0.3
            
        if "orbit" in combined_text or "gravity" in combined_text or "spacecraft" in combined_text:
            chain.append({"model": "orbital", "inputs": ["time", "intrinsic_property_1", "intrinsic_property_2", "boundary_position", "initial_position"], "output_maps_to": "position", "target_value": 100.0, "initial_value": 0.0, "optimization_goal": "target", "assumed_defaults": ["initial_value"]})
            weights["orbital"] = 0.3
            
        if "quantum" in combined_text or "wavefunction" in combined_text or "electron" in combined_text:
            chain.append({"model": "schrodinger", "inputs": ["time", "space", "intrinsic_property_1", "intrinsic_property_2", "boundary_wavefunction", "initial_wavefunction"], "output_maps_to": "wavefunction", "target_value": 1.0, "initial_value": 0.0, "optimization_goal": "target", "assumed_defaults": ["initial_value"]})
            weights["schrodinger"] = 0.3
            
        if "melting" in combined_text or "freezing" in combined_text or "phase change" in combined_text:
            chain.append({"model": "phase_change", "inputs": ["time", "space", "intrinsic_property_1", "intrinsic_property_2", "boundary_state", "initial_state"], "output_maps_to": "state", "target_value": 1.0, "initial_value": 0.0, "optimization_goal": "target", "assumed_defaults": ["initial_value"]})
            weights["phase_change"] = 0.3
            
        if "radiation" in combined_text or "decay" in combined_text or "intensity" in combined_text:
            chain.append({"model": "radiation", "inputs": ["time", "space", "intrinsic_property_1", "intrinsic_property_2", "boundary_intensity", "initial_intensity"], "output_maps_to": "intensity", "target_value": 0.0, "initial_value": 1.0, "optimization_goal": "minimize", "assumed_defaults": ["initial_value"]})
            weights["radiation"] = 0.3
            
        # Fallback if spec is incredibly vague: Build a tough 3-stage chain
        if not chain:
            chain = [
                {"model": "heat", "inputs": ["time", "space", "intrinsic_property_1", "intrinsic_property_2", "boundary_temperature", "initial_temperature"], "output_maps_to": "temperature", "target_value": 45.0, "initial_value": 25.0, "optimization_goal": "target", "assumed_defaults": ["initial_value"]},
                {"model": "darcy", "inputs": ["space_x", "space_y", "temperature", "intrinsic_property_2", "boundary_pressure", "initial_pressure"], "output_maps_to": "pressure", "target_value": 1.0, "initial_value": 0.0, "optimization_goal": "target", "assumed_defaults": ["initial_value"]},
                {"model": "biology", "inputs": ["time", "pressure", "intrinsic_property_2", "boundary_biomass", "initial_biomass"], "output_maps_to": "biomass", "target_value": 100.0, "initial_value": 1.0, "optimization_goal": "maximize", "assumed_defaults": ["initial_value"]}
            ]
            weights = {"heat": 0.2, "darcy": 0.3, "biology": 0.5}
            
        # Normalize weights to exactly 1.0
        total_weight = sum(weights.values())
        normalized_weights = {k: round(v / total_weight, 2) for k, v in weights.items()}
        
        # Ensure exact sum due to rounding
        diff = 1.0 - sum(normalized_weights.values())
        if diff != 0 and normalized_weights:
            first_key = list(normalized_weights.keys())[0]
            normalized_weights[first_key] = round(normalized_weights[first_key] + diff, 2)
            
        # Extract target entities from spec for strict filtering
        target_entities = []
        if "crop" in spec and spec["crop"]:
            target_entities = [spec["crop"], spec["crop"].lower()]
        elif "material" in spec and spec["material"]:
            target_entities = [spec["material"], spec["material"].lower()]
            
        return {
            "execution_chain": chain,
            "weights": normalized_weights,
            "target_entities": list(set(target_entities)),
            "semantic_keywords": ["crop", "plant", "seed", "botany", "agriculture"],
            "context": "Generated via local Mock LLM router based on keyword triggers."
        }
        
    def _call_real_llm(self, spec: dict) -> dict:
        """Actual API call to Gemini using GEMINI_API_KEY from .env."""
        try:
            import google.generativeai as genai
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("WARNING: GEMINI_API_KEY not found in .env. Falling back to Mock Router.")
                return self._mock_llm_routing(spec)
                
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash',
                                          system_instruction=self.LLM_SYSTEM_PROMPT)
            
            prompt = f"Please generate the DAG execution JSON for this specification:\n{json.dumps(spec, indent=2)}\nRemember to return ONLY valid JSON."
            response = model.generate_content(prompt)
            
            # Clean JSON formatting from markdown code blocks
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
                
            dag = json.loads(text.strip())
            
            # Anti-Hallucination Interceptor
            factory = PINNFactory()
            valid_models = list(factory._registry.keys())
            for node in dag.get("execution_chain", []):
                model_name = node.get("model", "")
                if model_name not in valid_models:
                    old_name = model_name
                    if "stress" in model_name.lower() or "drought" in model_name.lower():
                        node["model"] = "stress"
                    elif "biology" in model_name.lower() or "growth" in model_name.lower() or "yield" in model_name.lower():
                        node["model"] = "biology"
                    else:
                        import difflib
                        matches = difflib.get_close_matches(model_name, valid_models, n=1, cutoff=0.5)
                        if matches:
                            node["model"] = matches[0]
                        else:
                            node["model"] = "logistic"
                    
                    new_name = node["model"]
                    print(f"      => [Router Anti-Hallucination] Mapped '{old_name}' -> '{new_name}'")
                    if "weights" in dag and old_name in dag["weights"]:
                        dag["weights"][new_name] = dag["weights"].pop(old_name)

            # Default-Value Safety Net
            # Catches any node where Gemini still emits an old-style sentinel (-999.0,
            # or an unflagged ambiguous 0.0) despite the updated prompt instructions.
            # Values mirror exactly what Aryan specified for each domain.
            INITIAL_DEFAULTS = {
                "temperature": 25.0, "temp": 25.0,
                "ph": 7.0,
                "rate": 1.0,
                "biomass": 1.0, "growth": 1.0,
                "infected": 0.01,
                "stress": 0.0, "pressure": 0.0,
                "cost": 100.0,
                "velocity": 0.0,
                "displacement": 0.0,
                "intensity": 0.0,
                "concentration": 0.0,
                "wavefunction": 0.0,
                "state": 0.0,
                "em_field": 0.0,
                "position": 0.0
            }
            # boundary_pressure specifically defaults to atmospheric baseline (1.0),
            # distinct from initial_stress/initial_pressure (0.0, no applied load).
            BOUNDARY_PRESSURE_DEFAULT = 1.0

            TARGET_DEFAULTS = {
                "biomass": 100.0, "growth": 100.0,
                "stress": 1.0,
            }

            def _pick_initial_default(output_var: str, req_name: str = "") -> float:
                combined = f"{output_var} {req_name}".lower()
                if "pressure" in combined and "boundary" in combined:
                    return BOUNDARY_PRESSURE_DEFAULT
                for key, val in INITIAL_DEFAULTS.items():
                    if key in combined:
                        return val
                return 0.0

            def _pick_target_default(output_var: str, optimization_goal: str) -> float:
                ov = (output_var or "").lower()
                if "rate" in ov:
                    return 0.1 if optimization_goal == "minimize" else 10.0
                for key, val in TARGET_DEFAULTS.items():
                    if key in ov:
                        return val
                return 0.0

            for node in dag.get("execution_chain", []):
                node.setdefault("assumed_defaults", [])
                output_var = node.get("output_maps_to", "")
                goal = node.get("optimization_goal", "target")

                if float(node.get("initial_value", -999.0)) == -999.0:
                    node["initial_value"] = _pick_initial_default(output_var)
                    if "initial_value" not in node["assumed_defaults"]:
                        node["assumed_defaults"].append("initial_value")

                # target_value's old 0.0-means-missing heuristic is retired —
                # 0.0 can be a real, intentional target. We only trust the
                # router's own "assumed_defaults" flag here, never the raw value.
                if "target_value" in node.get("assumed_defaults", []):
                    node["target_value"] = _pick_target_default(output_var, goal)

            dag["context"] = "Generated via Gemini API."
            return dag
            
        except Exception as e:
            print(f"WARNING: Gemini API failed: {e}. Falling back to Mock Router.")
            return self._mock_llm_routing(spec)

if __name__ == "__main__":
    router = DAGRouter(use_mock=True)
    # Test on one of the specific spec files
    test_spec = r"c:\Users\dhanu\Downloads\Updated_pranag\Updated\Model\datasrc\spec_20260604_062219_5ab7e575.json"
    if os.path.exists(test_spec):
        dag = router.build_dag(test_spec)
        print("Generated DAG:")
        print(json.dumps(dag, indent=2))
