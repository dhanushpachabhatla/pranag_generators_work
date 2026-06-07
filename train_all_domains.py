import sys
from unified_pipeline import PranagPipeline

domains = ["heat", "darcy", "stress", "arrhenius", "biology", "logistic", "reaction_diffusion"]

for d in domains:
    print(f"\n================ STARTING TRAINING FOR {d} ================")
    pipeline = PranagPipeline(domain=d, test_mode=False)
    pipeline.run_end_to_end()
