import sys
import os
from unified_pipeline import PranagPipeline

# Combined list of all domains
all_domains = ["maxwell"]

print("========================================================")
print("=== FAST SURROGATE UPGRADE SCRIPT ===")
print("========================================================")
print("This script will skip PyTorch PINN training (Phase 1) and instantly")
print("jump to Data Generation (Phase 2) and Random Forest Training (Phase 3)!\n")

for d in all_domains:
    # Only upgrade if the PINN actually exists
    pinn_path = f"unified_pipeline_new_output_1/pinn/Parametric_{d.capitalize()}PINN.pt"
    if os.path.exists(pinn_path):
        print(f"\n================ UPGRADING SURROGATE FOR {d} ================")
        pipeline = PranagPipeline(domain=d, test_mode=False)
        # We pass use_existing_pinn=True to bypass PyTorch training!
        pipeline.run_end_to_end(use_existing_pinn=True)
    else:
        print(f"\n[SKIP] No existing PINN found for '{d}'. Run train_all_domains.py first.")
