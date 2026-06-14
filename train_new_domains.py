import sys
from unified_pipeline import PranagPipeline

new_domains = ["navier_stokes", "maxwell", "schrodinger", "orbital", "radiation", "economics"]

for d in new_domains:
    print(f"\n================ STARTING TRAINING FOR {d} ================")
    pipeline = PranagPipeline(domain=d, test_mode=False)
    pipeline.run_end_to_end()
