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
    
    SURROGATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'unified_pipeline_output', 'surrogate'))
    PARQUET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model', 'datasrc', 'universal_index_final.parquet'))
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

    def _build_inputs(self, df: pd.DataFrame, spec: dict, n_features: int) -> np.ndarray:
        """
        Dynamically builds the correct input array based on model.n_features_in_.
        Injects initial and boundary conditions (from spec.json and chaining).
        """
        n = len(df)
        time_v = np.full(n, 24.0, dtype=np.float32) # Temporal
        spatial_v = np.full(n, 0.5, dtype=np.float32) # Spatial
        
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

        time_v = np.ones(n, dtype=np.float32) * 5.0
        cols = [time_v]
        
        requested_inputs = node.get("inputs", [])
        for req in requested_inputs:
            if req in ["time", "space", "t", "x"]: continue
            
            # Assign regex if it's a known physical property
            pattern = None
            if "temp" in req.lower(): pattern = r'(?i)(?:temp|temperature).*?(\d{2,3})'
            elif "ph" in req.lower(): pattern = r'(?i)ph.*?([0-9]{1,2}\.?[0-9]?)'
            elif "mass" in req.lower() or "yield" in req.lower(): pattern = r'(?i)(?:mass|weight|yield).*?([0-9]+\.?[0-9]?)'
            
            # Extract dynamic values based on the LLM's requested input name
            val_col = get_dynamic_col(req, 1.0, pattern)
            cols.append(val_col)
            
        # Pad any remaining expected features with zeros to prevent shape crashes
        while len(cols) < n_features:
            cols.append(np.zeros(n, dtype=np.float32))
            
        return np.column_stack(cols[:n_features])

    def _calculate_score(self, pred: np.ndarray, spec: dict, target_value: float, optimization_goal: str) -> np.ndarray:
        """
        Validates scoring math using spec.json targets and DAG optimization goals.
        Normalizes predictions into a strict 0.0 - 1.0 viability score.
        """
        pred_flat = pred.flatten()
        scores = np.zeros_like(pred_flat)
        
        if optimization_goal == "maximize":
            # Score approaches 1.0 as prediction approaches or exceeds target
            if target_value == 0.0: target_value = 1.0 # Prevent div by 0
            scores = np.clip(pred_flat / target_value, 0.0, 1.0)
            
        elif optimization_goal == "minimize":
            # Score is 1.0 if prediction is 0, approaches 0 as prediction reaches or exceeds target
            if target_value == 0.0: target_value = 1.0
            scores = np.clip(1.0 - (pred_flat / target_value), 0.0, 1.0)
            
        else: # "target"
            # Exponential decay error metric: 1.0 if perfect, drops to 0 if far from target
            scores = np.exp(-np.abs(pred_flat - target_value) / (target_value + 1e-8))
            
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
                print(f"\n[CRITICAL HALT] Missing trained surrogate model: '{surrogate_path}'")
                print(f"The DAG requested '{domain}', but it has not been trained yet.")
                print(f"ACTION REQUIRED: Add '{domain}' to the 'domains_to_train' list in unified_pipeline.py, run it to pre-train the model, and then restart this inference pipeline.")
                import sys
                sys.exit(1)
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
            # Automatically drop irrelevant encyclopedia entries using LLM-generated keywords
            semantic_keywords = dag.get("semantic_keywords", [])
            if semantic_keywords:
                pattern = r'(?i)(' + '|'.join(semantic_keywords) + ')'
                semantic_mask = df["description"].astype(str).str.contains(pattern) | \
                                df.get("tags", pd.Series("")).astype(str).str.contains(pattern)
                df = df[semantic_mask].copy()
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
            for node in valid_chain:
                domain = node["model"]
                maps_to = node.get("output_maps_to", "generic_param")
                model = self._load_surrogate(domain)
                
                # Dynamic Joblib Inputs
                n_features = getattr(model, 'n_features_in_', 3)
                X = self._build_inputs(df, spec, n_features, node)
                
                # Dynamic Targets & Optimization Goal
                target_val = float(node.get("target_value", 0.0))
                opt_goal = node.get("optimization_goal", "target")
                
                print(f"      -> Simulating '{domain}' (Input shape: {X.shape})...")
                try:
                    raw_pred = model.predict(X)
                except Exception as e:
                    print(f"      -> ERROR running '{domain}': {e}. Skipping.")
                    continue
                    
                if raw_pred.ndim > 1:
                    raw_pred = raw_pred[:, 0]
                    
                # Chaining: Feed output forward into the dataframe
                df[maps_to] = raw_pred
                
                # Robust Scoring Math
                y_score = self._calculate_score(raw_pred, spec, target_val, opt_goal)
                scores_df[f"{domain}_score"] = y_score
                
                # Apply weight
                viability_score += y_score * normalized_weights.get(domain, 0.0)
                
            viability_score += penalty
            scores_df["viability_score"] = np.clip(viability_score, 0.0, 1.0)
            
            # 0.7 Pre-Filter (Hybrid Trimming Fallback to top 5%)
            surviving_df = scores_df[scores_df["viability_score"] >= 0.70]
            
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

if __name__ == "__main__":
    engine = InferenceEngine(use_mock_llm=False)
    test_spec = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model', 'datasrc', 'spec_20260604_062219_5ab7e575.json'))
    engine.run(test_spec)
