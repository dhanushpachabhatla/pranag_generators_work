import sys
from unified_pipeline import PranagPipeline

new_domains = ["radiation"]

for d in new_domains:
    print(f"\n================ STARTING TRAINING FOR {d} ================")
    pipeline = PranagPipeline(domain=d, test_mode=False)
    pipeline.run_end_to_end()
