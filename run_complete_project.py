import os
import sys

# Ensure custom modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Pipeline_New')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Validator_New')))

from inference_pipeline import InferenceEngine
from cross_domain_validator import CrossDomainValidator

def run_project(spec_path: str):
    """
    Master Orchestrator for Phase 2: Downstream Inference & Validation.
    """
    print(f"\n=================================================================")
    print(f"             PRANA-G DOWNSTREAM INFERENCE PIPELINE               ")
    print(f"=================================================================")
    print(f"Target Specification: {os.path.basename(spec_path)}")
    
    # 1. Run Inference Pipeline
    engine = InferenceEngine(use_mock_llm=False)  # Tries Gemini API, falls back to Mock
    engine.run(spec_path)
    
    # Check if handoff was created (if a surrogate was missing, it halts)
    if not os.path.exists(engine.OUTPUT_PATH):
        print(f"\n[!] Pipeline halted gracefully before validation.")
        return
        
    # 2. Run Cross-Domain Validator
    output_json = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Validator_New', 'Top_100_Validated_Designs.json'))
    validator = CrossDomainValidator(engine.OUTPUT_PATH, output_json)
    validator.validate()
    
    print(f"\n=================================================================")
    print(f"                      PIPELINE COMPLETE                          ")
    print(f"=================================================================")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        spec_file = sys.argv[1]
    else:
        # Default testing spec
        spec_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Model', 'datasrc', 'spec_20260604_062219_5ab7e575.json'))
        
    if not os.path.exists(spec_file):
        print(f"ERROR: Specification file not found: {spec_file}")
        sys.exit(1)
        
    run_project(spec_file)
