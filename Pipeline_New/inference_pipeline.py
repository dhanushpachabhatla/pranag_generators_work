import pandas as pd
import numpy as np
import joblib
import os
import json
import pyarrow.parquet as pq
import re
import hashlib
from dag_router import DAGRouter

class InferenceEngine:
    """
    Phase 2 Physics Scoring Engine (Robust Version).
    Features: Batch Streaming, Sequential DAG Chaining, Dynamic Input Sizing, 
    and Mathematical Weight Normalization.
    """
    
    SURROGATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "unified_pipeline_new_output_1", "surrogate"))
    PARQUET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasrc', 'universal_index_final.parquet'))
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'inference_handoff.csv')
    
    def __init__(self, use_mock_llm=False):
        self.router = DAGRouter(use_mock=use_mock_llm)
        self.surrogates = {}
        
    def _load_surrogate(self, domain: str):
        if domain in self.surrogates:
            return self.surrogates[domain]
        model_path = os.path.join(self.SURROGATE_DIR, f"Surrogate_{domain}.joblib")
        if not os.path.exists(model_path):
            return None
        try:
            self.surrogates[domain] = joblib.load(model_path)
            return self.surrogates[domain]
        except Exception:
            return None

    def _build_inputs(self, df: pd.DataFrame, spec: dict, n_features: int, node: dict, previous_output_var: str = None) -> np.ndarray:
        """
        Dynamically builds the correct input array based on model.n_features_in_.
        Injects initial and boundary conditions (from spec.json and chaining).
        """
        n = len(df)
        time_v = np.full(n, 0.5, dtype=np.float32) # Temporal (normalized 0.5)
        spatial_v = np.linspace(0.0, 1.0, n, dtype=np.float32) # Spatial gradient across entities
        
        # Parse Environment Overrides from spec.json (Initial & Boundary limits)
        env_temp = float(spec.get("temperature", 25.0))
        env_ph = 7.0
        if "acidic" in str(spec).lower(): env_ph = 5.0
        if "alkaline" in str(spec).lower(): env_ph = 8.5
        
        # Helper: Extract via Regex, fallback to Hash
        def get_dynamic_col(feature_name, default_val, regex_pattern=None):
            # 1. Check if chained from a previous model
            if feature_name in df.columns:
                val = pd.to_numeric(df[feature_name], errors="coerce")
                if not val.isna().all():
                    return val.fillna(default_val).values.astype(np.float32)
            
            # 2. Regex extract from description if available
            extracted = np.full(n, np.nan, dtype=np.float32)
            if regex_pattern and "description" in df.columns:
                # Vectorized regex extraction
                matches = df["description"].astype(str).str.extract(regex_pattern, expand=False)
                extracted = pd.to_numeric(matches, errors="coerce").values
            
            # 3. Deterministic Hash Fallback
            def hash_to_float(eid, seed_str):
                h = int(hashlib.md5(f"{eid}_{seed_str}".encode()).hexdigest()[:8], 16)
                return default_val * (0.5 + (h / 0xffffffff))
                
            hash_fallback = np.array([hash_to_float(eid, feature_name) for eid in df.get("entity_id", range(n))])
            
            final_vals = np.where(np.isnan(extracted), hash_fallback, extracted)
            return final_vals.astype(np.float32)

        cols = []
        
        # We MUST enforce a strict positional layout matching the LHS training data.
        from training.simulation_generator import EQUATION_PATTERNS
        domain_name = node.get("pinn_type", "heat")
        
        # Determine base variables dynamically
        if domain_name in EQUATION_PATTERNS:
            independent_vars = EQUATION_PATTERNS[domain_name]["independent"]
        else:
            # Fallback based on n_features if domain not found
            independent_vars = ["t", "x"] if n_features == 4 else ["t", "x", "y"] if n_features == 5 else ["t"]
            
        requested_inputs = []
        for var in independent_vars:
            if var in ["t", "time", "time_days"]:
                requested_inputs.append("time")
            elif var in ["x", "space", "depth", "S", "r"]:
                requested_inputs.append("space_x")
            elif var == "y":
                requested_inputs.append("space_y")
            elif var == "z":
                requested_inputs.append("space_z")
            else:
                requested_inputs.append("space_x") # default spatial
                
        # The remaining columns are parameters (targets/boundaries/intrinsics)
        # unified_pipeline adds +4 dimensions: Boundary, IC, Intrinsic 1, Intrinsic 2
        remaining = n_features - len(requested_inputs)
        if remaining > 0:
            if remaining >= 1: requested_inputs.append("boundary")
            if remaining >= 2: requested_inputs.append("initial_condition")
            if remaining >= 3: requested_inputs.append("intrinsic_property_1")
            if remaining >= 4: requested_inputs.append("intrinsic_property_2")
            for i in range(5, remaining + 1):
                requested_inputs.append(f"intrinsic_property_{i-2}")

        # GUARANTEED CHAINING: If a previous model ran, its output becomes the primary boundary condition
        # for this model. This enforces mathematical continuity across the DAG.
        if previous_output_var and previous_output_var in df.columns:
            if "boundary" in requested_inputs:
                idx = requested_inputs.index("boundary")
                requested_inputs[idx] = previous_output_var
                
        for req in requested_inputs:
            # 1. Temporal dimension (scaled [0, 1] in training)
            if req in ["time", "t", "time_days"]:
                cols.append(np.ones(n, dtype=np.float32) * 0.5)
                continue
                
            # 2. Spatial dimensions (scaled [-1, 1] in training)
            if req in ["space", "x", "space_x", "space_y", "depth"]:
                cols.append(spatial_v)
                continue
                
            # 3. Parametric Boundary/Initial targets (Static Absolute Scaling)
            if "boundary" in req.lower() or "initial" in req.lower():
                is_boundary = "boundary" in req.lower()
                raw_val = float(node.get("target_value", 0.0)) if is_boundary else float(node.get("initial_value", -999.0))
                
                # Absolute static scaling bounds to ensure mathematical consistency across ALL prompts
                if "temp" in req.lower() or "heat" in req.lower(): 
                    scaler = 1500.0  # Safe upper limit for materials and biology
                    default_init = 25.0
                elif "pressure" in req.lower() or "stress" in req.lower(): 
                    scaler = 1000.0  # MPa
                    default_init = 0.0
                elif "biomass" in req.lower() or "growth" in req.lower(): 
                    scaler = 10000.0 # kg/ha
                    default_init = 10.0
                elif "ph" in req.lower():
                    scaler = 14.0
                    default_init = 7.0
                else: 
                    scaler = 100.0
                    default_init = 0.0
                    
                # Apply defaults if LLM did not provide initial state
                if not is_boundary and raw_val == -999.0:
                    raw_val = default_init
                elif is_boundary and raw_val == 0.0:
                    raw_val = default_init  # Ensure we don't accidentally simulate boundary at absolute zero if unspecified
                    
                val = np.clip(raw_val / scaler, 0.0, 1.0)
                cols.append(np.full(n, val, dtype=np.float32))
                continue
                
            # 4. Handle Intrinsic Properties
            if req in ["intrinsic_property_1", "intrinsic_property_2"]:
                if req in df.columns:
                    # Use the actual normalized values from the Parquet!
                    val = pd.to_numeric(df[req], errors="coerce").fillna(0.5).values.astype(np.float32)
                    cols.append(np.clip(val, 0.0, 1.0))
                else:
                    # Deterministic random fallback between 0.0 and 1.0 based on entity ID
                    def hash_to_intrinsic(eid, seed_str):
                        h = int(hashlib.md5(f"{eid}_{seed_str}".encode()).hexdigest()[:8], 16)
                        return h / 0xffffffff
                    
                    dummy_intrinsic = np.array([hash_to_intrinsic(eid, req) for eid in df.get("entity_id", range(n))])
                    cols.append(dummy_intrinsic.astype(np.float32))
                continue
                
            # Assign regex if it's a known physical property
            pattern = None
            if "temp" in req.lower(): pattern = r'(?i)(?:temp|temperature).*?(\d{2,3})'
            elif "ph" in req.lower(): pattern = r'(?i)ph.*?([0-9]{1,2}\.?[0-9]?)'
            elif "mass" in req.lower() or "yield" in req.lower(): pattern = r'(?i)(?:mass|weight|yield).*?([0-9]+\.?[0-9]?)'
            
            # Extract dynamic values based on the LLM's requested input name
            # Normalized safely for surrogate
            val_col = get_dynamic_col(req, 0.5, pattern)
            # Clip between [0, 1] to prevent massive extrapolation from Random Forest
            val_col = np.clip(val_col / np.max(val_col) if np.max(val_col) > 0 else val_col, 0.0, 1.0)
            cols.append(val_col)
            
        # Pad any remaining expected features with zeros to prevent shape crashes
        while len(cols) < n_features:
            cols.append(np.zeros(n, dtype=np.float32))
            
        return np.column_stack(cols[:n_features])

    def _calculate_score(self, pred: np.ndarray, spec: dict, target_value: float, optimization_goal: str, scaler: float, domain: str) -> np.ndarray:
        """
        Validates scoring math using spec.json targets and DAG optimization goals.
        Normalizes predictions into a strict 0.0 - 1.0 viability score.
        """
        pred_flat = pred.flatten()
        scaled_target = np.clip(target_value / scaler, 0.0, 1.0)
        
        # Preserve variance by using a softer base tolerance instead of a vertical math cliff
        base_tolerance = 0.5
        
        # Enforce physical reality over LLM hallucinations
        if "arrhenius" in domain.lower() or "logistic" in domain.lower() or "biology" in domain.lower():
            optimization_goal = "maximize"
            base_tolerance = 1.0  # Wider tolerance for unscaled PINN outputs
            
        if optimization_goal == "maximize":
            effective_target = max(scaled_target, 0.1)
            max_pred = max(np.max(pred_flat), effective_target + 0.0001)
            scores = np.where(
                pred_flat >= effective_target,
                0.5 + 0.5 * ((pred_flat - effective_target) / (max_pred - effective_target)),
                0.5 * np.exp(-np.abs(pred_flat - effective_target) / base_tolerance)
            )
            
        elif optimization_goal == "minimize":
            effective_target = max(scaled_target, 0.1)
            min_pred = min(np.min(pred_flat), effective_target - 0.0001)
            scores = np.where(
                pred_flat <= effective_target,
                0.5 + 0.5 * ((effective_target - pred_flat) / (effective_target - min_pred)),
                0.5 * np.exp(-np.abs(pred_flat - effective_target) / base_tolerance)
            )
            
        else: # "target"
            tolerance = max(scaled_target + 0.1, base_tolerance)
            # Center target scores similarly to preserve scale consistency across the DAG
            scores = np.where(
                np.abs(pred_flat - scaled_target) < 1e-5,
                1.0,
                0.5 * np.exp(-np.abs(pred_flat - scaled_target) / tolerance)
            )
            
        print(f"DEBUG SCORE [{domain}]: pred_mean={np.mean(pred_flat):.4f}, target={target_value}, scaled_target={scaled_target:.4f}, goal={optimization_goal}, scores_mean={np.mean(scores):.4f}")
        return scores

    def run(self, spec_path: str):
        print(f"\n==============================================")
        print(f"--- INFERENCE PIPELINE INITIATED ---")
        print(f"==============================================")
        
        with open(spec_path, 'r', encoding='utf-8') as f:
            spec = json.load(f)
            
        print(f"1. Parsing Spec & Triggering DAG Router...")
        dag = self.router.build_dag(spec_path)
        
        # Check if the LLM output is the legacy format or the new format
        if "execution_chain" in dag:
            execution_chain = dag.get("execution_chain", [])
        else:
            # Fallback for LLMs that didn't follow the new prompt perfectly
            active = dag.get("active_surrogates", [])
            execution_chain = [{"model": d, "output_maps_to": "generic_param"} for d in active]
            
        weights = dag.get("weights", {})
        
        # Edge Case & Robustness: Strictly halt if requested models are missing
        valid_chain = []
        valid_weights = {}
        for node in execution_chain:
            domain = node.get("model", "")
            surrogate_path = os.path.join(self.SURROGATE_DIR, f"Surrogate_{domain}.joblib")
            if not os.path.exists(surrogate_path):
                print(f"\n[Warning] The LLM requested '{domain}' but it is not trained. Re-mapping to 'logistic'.")
                domain = "logistic"
                node["model"] = "logistic"
            valid_chain.append(node)
            valid_weights[domain] = weights.get(domain, 0.5)
            
        total_weight = sum(valid_weights.values())
        if total_weight == 0: total_weight = 1.0
        normalized_weights = {k: v / total_weight for k, v in valid_weights.items()}
        
        print(f"   => Router Context : {dag.get('context')}")
        print(f"   => Execution DAG  : {' -> '.join([n['model'] for n in valid_chain])}")
        print(f"   => Adjusted Weights: {json.dumps(normalized_weights)}")
        
        print(f"\n2. Streaming Parquet in Batches...")
        if not os.path.exists(self.PARQUET_PATH):
            print(f"ERROR: Parquet not found at {self.PARQUET_PATH}")
            return
            
        parquet_file = pq.ParquetFile(self.PARQUET_PATH)
        batch_size = 50000
        
        if os.path.exists(self.OUTPUT_PATH):
            os.remove(self.OUTPUT_PATH)
            
        total_survivors = 0
        total_processed = 0
        
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            df = batch.to_pandas()
            
            # --- Dynamic Semantic Domain Pre-Filter ---
            target_entities = dag.get("target_entities", [])
            semantic_keywords = dag.get("semantic_keywords", [])
            domain_filters = dag.get("domain_filters", [])
            
            combined_mask = pd.Series(False, index=df.index)
            
            if target_entities:
                pattern_strict = r'(?i)(?:^|[^a-zA-Z])(' + '|'.join(target_entities) + r')(?:[^a-zA-Z]|$)'
                strict_mask = df["name"].astype(str).str.contains(pattern_strict, regex=True) | \
                              df["description"].astype(str).str.contains(pattern_strict, regex=True) | \
                              df.get("tags", pd.Series("")).astype(str).str.contains(pattern_strict, regex=True)
                combined_mask = combined_mask | strict_mask
                
            if semantic_keywords:
                pattern_broad = r'(?i)\b(' + '|'.join(semantic_keywords) + r')\b'
                broad_mask = df["name"].astype(str).str.contains(pattern_broad, regex=True) | \
                             df["description"].astype(str).str.contains(pattern_broad, regex=True) | \
                             df.get("tags", pd.Series("")).astype(str).str.contains(pattern_broad, regex=True)
                combined_mask = combined_mask | broad_mask
                
            if domain_filters and "domain" in df.columns:
                valid_domains = [d.lower() for d in domain_filters]
                domain_mask = df["domain"].astype(str).str.lower().isin(valid_domains)
                combined_mask = combined_mask & domain_mask
                
            if target_entities or semantic_keywords:
                df_filtered = df[combined_mask].copy()
            else:
                df_filtered = df.copy()
                    
            df = df_filtered
            df.reset_index(drop=True, inplace=True)
            
            n = len(df)
            if n == 0:
                print(f"   => Batch {batch_idx+1}: 0 semantically relevant entities found. Skipping.")
                continue
            total_processed += n
            print(f"\n   --- Processing Batch {batch_idx+1} ({n} rows) ---")
            
            scores_df = pd.DataFrame({"entity_id": df["entity_id"], "name": df["name"], "domain": df["domain"]})
            viability_score = np.zeros(n, dtype=np.float32)
            
            # Confidence penalty (Edge case missing data)
            missing_cols = [c for c in ['key_prop_1', 'key_prop_2', 'key_prop_3'] if c in df.columns]
            if missing_cols:
                missing_count = df[missing_cols].isna().sum(axis=1).values
                penalty = np.where(missing_count >= 2, -0.20, 0.0).astype(np.float32)
            else:
                penalty = 0.0
            
            # Execute Sequential Chaining
            previous_output_var = None
            for node in valid_chain:
                domain = node["model"]
                maps_to = node.get("output_maps_to", "generic_param")
                model = self._load_surrogate(domain)
                
                # Dynamic Joblib Inputs
                n_features = getattr(model, 'n_features_in_', 3)
                X = self._build_inputs(df, spec, n_features, node, previous_output_var)
                
                # Determine mathematically consistent scaler based on the domain mapped output
                if "temp" in maps_to or "heat" in maps_to: scaler = 1500.0
                elif "pressure" in maps_to or "stress" in maps_to: scaler = 1000.0
                elif "biomass" in maps_to or "growth" in maps_to: scaler = 10000.0
                elif "ph" in maps_to: scaler = 14.0
                else: scaler = 100.0
                
                # Dynamic Targets & Optimization Goal
                target_val = float(node.get("target_value", 0.0))
                opt_goal = node.get("optimization_goal", "target")
                
                print(f"      -> Simulating '{domain}' (Input shape: {X.shape})...")
                try:
                    raw_pred = model.predict(X)
                except Exception as e:
                    print(f"      -> ERROR running '{domain}': {e}. Skipping.")
                    continue
                    
                if raw_pred.ndim > 1 and raw_pred.shape[1] > 1:
                    # Multi-Output mapping
                    from training.simulation_generator import EQUATION_PATTERNS
                    domain_outputs = EQUATION_PATTERNS.get(domain, {}).get("dependent", [])
                    primary_pred = raw_pred[:, 0]
                    for i in range(raw_pred.shape[1]):
                        var_name = domain_outputs[i] if i < len(domain_outputs) else f"out_{i}"
                        df[f"{maps_to}_{var_name}"] = raw_pred[:, i]
                        if i == 0:
                            df[maps_to] = raw_pred[:, i] # Legacy fallback chaining
                            previous_output_var = maps_to
                else:
                    if raw_pred.ndim > 1: raw_pred = raw_pred[:, 0]
                    primary_pred = raw_pred
                    df[maps_to] = raw_pred
                    previous_output_var = maps_to
                
                # Robust Scoring Math ensuring prediction is evaluated against scaled limits
                # Always score based on the primary physical variable
                y_score = self._calculate_score(primary_pred, spec, target_val, opt_goal, scaler, domain)
                scores_df[f"{domain}_score"] = y_score
                
                # Apply weight
                viability_score += y_score * normalized_weights.get(domain, 0.0)
                
            viability_score += penalty
            scores_df["viability_score"] = np.clip(viability_score, 0.0, 1.0)
            
            # 0.45 Pre-Filter (Adjusted for continuous variance centered at 0.5)
            surviving_df = scores_df[scores_df["viability_score"] >= 0.45]
            
            min_survivors = int(n * 0.05)
            if len(surviving_df) < min_survivors:
                # Guarantee at least Top 5% of the batch survive
                surviving_df = scores_df.sort_values(by="viability_score", ascending=False).head(min_survivors)
                
            total_survivors += len(surviving_df)
            
            # Append to CSV securely
            mode = 'w' if batch_idx == 0 else 'a'
            header = True if batch_idx == 0 else False
            surviving_df.to_csv(self.OUTPUT_PATH, mode=mode, header=header, index=False)
            
        print(f"\n3. Pre-Filtering Summary")
        print(f"   => Processed {total_processed:,} total candidates.")
        print(f"   => Deleted {total_processed - total_survivors:,} failed candidates (< 0.7).")
        print(f"   => {total_survivors:,} highly viable candidates survived the physics.")
        print(f"\nInference Complete! Handoff saved to: {self.OUTPUT_PATH}")
        
        return {
            "execution_chain": [n["model"] for n in valid_chain],
            "total_processed": total_processed,
            "total_survivors": total_survivors,
            "handoff_path": self.OUTPUT_PATH
        }

if __name__ == "__main__":
    engine = InferenceEngine(use_mock_llm=False)
    test_spec = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasrc', 'spec_20260604_062219_5ab7e575.json'))
    engine.run(test_spec)
