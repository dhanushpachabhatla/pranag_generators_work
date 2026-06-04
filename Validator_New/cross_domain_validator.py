import pandas as pd
import json
import os

class CrossDomainValidator:
    """
    Phase 2 Safety Filter.
    Reads the inference_handoff.csv, applies cross-domain safety checks,
    and returns the Top 100 designs.
    """
    
    def __init__(self, inference_path: str, output_path: str):
        self.inference_path = inference_path
        self.output_path = output_path
        
    def validate(self):
        print(f"\n==============================================")
        print(f"--- CROSS-DOMAIN VALIDATION INITIATED ---")
        print(f"==============================================")
        
        if not os.path.exists(self.inference_path):
            print(f"ERROR: Inference handoff file not found at {self.inference_path}")
            return
            
        print(f"1. Loading Pre-Filtered Handoff Data...")
        df = pd.read_csv(self.inference_path)
        print(f"   => Loaded {len(df):,} highly-viable candidates.")
        
        print(f"\n2. Applying Cross-Domain Safety Checks...")
        # Cross domain safety thresholds. 
        # Even if a domain wasn't the primary focus, if it was scored and its score is terribly low, it's a hazard.
        # If the pipeline didn't score it, the column won't exist or will be NaN, which is safe to ignore here.
        
        failed_reasons = []
        safety_mask = pd.Series(True, index=df.index)
        
        # Cross-Domain Hazard Checks with Reason Tracking
        if "chemistry_score" in df.columns:
            chem_fail = df["chemistry_score"] < 0.2
            for idx in df[chem_fail].index:
                failed_reasons.append({
                    "entity_id": str(df.loc[idx, "entity_id"]),
                    "name": str(df.loc[idx, "name"]),
                    "reason": f"Critical chemistry hazard (score {df.loc[idx, 'chemistry_score']:.2f} < 0.2)"
                })
            safety_mask = safety_mask & ~chem_fail
            
        if "stress_score" in df.columns:
            stress_fail = df["stress_score"] < 0.2
            # Only append if not already failed by chemistry
            stress_only_fail = stress_fail & safety_mask
            for idx in df[stress_only_fail].index:
                failed_reasons.append({
                    "entity_id": str(df.loc[idx, "entity_id"]),
                    "name": str(df.loc[idx, "name"]),
                    "reason": f"Critical stress/structural failure (score {df.loc[idx, 'stress_score']:.2f} < 0.2)"
                })
            safety_mask = safety_mask & ~stress_fail
            
        df_safe = df[safety_mask]
        print(f"   => {len(failed_reasons):,} candidates rejected due to Cross-Domain Hazards.")
        
        # Save failed reasons log
        failed_log_path = os.path.join(os.path.dirname(self.output_path), 'Failed_Designs_Log.json')
        with open(failed_log_path, 'w', encoding='utf-8') as f:
            json.dump({"total_failed": len(failed_reasons), "failed_designs": failed_reasons}, f, indent=4)
        print(f"   => Failure reasons saved to: {failed_log_path}")
        
        print(f"\n3. Sorting and Extracting Top 100 Designs...")
        df_sorted = df_safe.sort_values(by="viability_score", ascending=False)
        top_100 = df_sorted.head(100)
        
        results = []
        for _, row in top_100.iterrows():
            rec = {
                "entity_id": str(row["entity_id"]),
                "name": str(row["name"]),
                "domain": str(row["domain"]),
                "viability_score": round(float(row["viability_score"]), 4),
                "domain_scores": {}
            }
            # Add whatever domain scores exist
            for col in df.columns:
                if col.endswith("_score") and col != "viability_score" and pd.notnull(row[col]):
                    rec["domain_scores"][col] = round(float(row[col]), 4)
            results.append(rec)
            
        output_data = {
            "total_evaluated": len(df),
            "total_safe": len(df_safe),
            "top_100": results
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
            
        print(f"   => Extraction complete. Safest Top {len(results)} designs identified.")
        print(f"\nValidation Complete! Final output saved to: {self.output_path}")

if __name__ == "__main__":
    inference_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pipeline_New', 'inference_handoff.csv'))
    output_json = os.path.join(os.path.dirname(__file__), 'Top_100_Validated_Designs.json')
    validator = CrossDomainValidator(inference_csv, output_json)
    validator.validate()
