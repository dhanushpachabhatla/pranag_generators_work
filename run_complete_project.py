import os
import sys
import time

# Ensure custom modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Pipeline_New')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Validator_New')))

from inference_pipeline import InferenceEngine
from cross_domain_validator import CrossDomainValidator
from dashboard_generator import generate_dashboard

def run_project(spec_path: str):
    """
    Master Orchestrator for Phase 2: Downstream Inference & Validation.
    """
    print(f"\n=================================================================")
    print(f"             PRANA-G DOWNSTREAM INFERENCE PIPELINE               ")
    print(f"=================================================================")
    print(f"Target Specification: {os.path.basename(spec_path)}")
    
    start_time = time.time()
    
    # 1. Run Inference Pipeline
    print(f"\n[Flow] Executing Phase 1: Heavy AI Inference from Parquet...")
    inf_start = time.time()
    engine = InferenceEngine(use_mock_llm=False)  # Tries Gemini API, falls back to Mock
    metrics = engine.run(spec_path)
    inf_end = time.time()
    print(f"[Flow] Phase 1 Complete in {inf_end - inf_start:.2f} seconds. Found {metrics['total_processed']} relevant entities.")
    
    # Check if handoff was created (if a surrogate was missing, it halts)
    if not os.path.exists(engine.OUTPUT_PATH) or not metrics:
        print(f"\n[!] Pipeline halted gracefully before validation.")
        return
        
    # 2. Run Cross-Domain Validator
    print(f"\n[Flow] Executing Phase 2: Cross-Domain Safety Validation...")
    val_start = time.time()
    output_json = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Validator_New', 'Top_100_Validated_Designs.json'))
    validator = CrossDomainValidator(engine.OUTPUT_PATH, output_json)
    validator.validate()
    val_end = time.time()
    print(f"[Flow] Phase 2 Complete in {val_end - val_start:.2f} seconds.")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Add execution stats to metrics
    metrics["total_time_seconds"] = round(total_time, 2)
    metrics["top_designs_json"] = output_json
    metrics["failed_designs_json"] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Validator_New', 'Failed_Designs_Log.json'))
    
    # 3. Generate HTML Dashboard
    dashboard_path = generate_dashboard(metrics)
    
    print(f"\n=================================================================")
    print(f"                      PIPELINE COMPLETE                          ")
    print(f"=================================================================")
    print(f" Total Execution Time : {metrics['total_time_seconds']} seconds")
    print(f" Total Entities Scanned: {metrics['total_processed']:,}")
    print(f" Physics Survivors     : {metrics['total_survivors']:,}")
    print(f" Final Dashboard       : {dashboard_path}")
    print(f"=================================================================\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        spec_file = sys.argv[1]
    else:
        # Default testing spec
        # spec_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasrc', 'spec_20260604_062219_5ab7e575.json'))
        spec_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasrc', 'spec3.json'))
        
    if not os.path.exists(spec_file):
        print(f"ERROR: Specification file not found: {spec_file}")
        sys.exit(1)
        
    run_project(spec_file)
