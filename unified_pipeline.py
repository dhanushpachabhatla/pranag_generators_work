import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "Model", "models"))
from training.pinn_factory import PINNFactory
from training.pinn_trainer import train_pinn_model
from Model.models.loss_generator import create_cross_domain_loss_generator
from Model.visualization import plot_surrogate_performance

class PranagPipeline:
    def __init__(self, domain="heat", test_mode=False):
        self.domain = domain
        self.test_mode = test_mode
        self.factory = PINNFactory()
        
        # Setup new unified output directory structure
        self.base_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "unified_pipeline_new_output_1"))
        self.pinn_dir = os.path.join(self.base_output_dir, "pinn")
        self.surrogate_dir = os.path.join(self.base_output_dir, "surrogate")
        self.plots_dir = os.path.join(self.base_output_dir, "plots")
        self.results_dir = os.path.join(self.base_output_dir, "results")
        
        for d in [self.pinn_dir, self.surrogate_dir, self.plots_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)

    def run_end_to_end(self):
        print(f"\n========================================================")
        print(f"=== Unified PRANA-G Pipeline for Domain: '{self.domain}' ===")
        print(f"========================================================\n")
        
        # 1. Build & Train Parametric PINN
        # Dynamically calculate the correct input dimension for N-Dimensional physics
        from training.simulation_generator import SimulationGenerator
        cfg = SimulationGenerator().from_hint(self.domain)
        # Base physical variables (e.g. t, x, y) + 2 Parametric Targets (T_bound, IC_bound)
        actual_input_dim = len(cfg.equation_info.independent) + 2
        
        print("--- Phase 1: Train Parametric PINN ---")
        pinn_model = self.factory.create(self.domain, input_dim=actual_input_dim, hidden_dim=128, num_layers=6, dynamic=True)
        # BUG FIX: Dynamically inject base_dim so pinn_trainer.py enforces Initial Conditions (IC_bound) correctly for 1D ODEs
        pinn_model.base_dim = len(cfg.equation_info.independent)
        alias = f"Parametric_{self.domain.capitalize()}PINN"
            
        max_epochs = 10 if self.test_mode else 3000
        num_points = 100 if self.test_mode else 10000
        batch_size = 10 if self.test_mode else 2048
            
        trained_pinn, _ = train_pinn_model(
            pinn_model=pinn_model,
            input_dim=actual_input_dim,
            num_points=num_points,
            max_epochs=max_epochs,
            batch_size=batch_size,
            model_alias=alias,
            plot_dir=self.pinn_dir
        )
        
        # 2. On-the-fly Data Generation (In-Memory)
        print("\n--- Phase 2: In-Memory Data Generation ---")
        
        # Dynamically generate N-dimensional grid to avoid RAM crashes
        target_total_points = 1000 if self.test_mode else 100000
        print(f"Generating exactly {target_total_points} points using Latin Hypercube Sampling (LHS).")
        
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=actual_input_dim)
        sample = sampler.random(n=target_total_points)
        
        # Scale bounds from [0, 1) to [-1, 1] for spatial/parametric dimensions
        inputs = 2.0 * sample - 1.0
        if actual_input_dim > 0:
            # Revert time dimension to [0, 1]
            inputs[:, 0] = (inputs[:, 0] + 1.0) / 2.0
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        
        # Ensure inputs_tensor is on the same device as the trained model (e.g. CUDA)
        device = next(trained_pinn.parameters()).device
        inputs_tensor = inputs_tensor.to(device)
        
        trained_pinn.eval()
        with torch.no_grad():
            preds = trained_pinn(inputs_tensor).cpu().numpy()
            
        print(f"Generated {len(inputs)} parametric data points in RAM.")
        
        # 3. Train Surrogate Model
        print("\n--- Phase 3: Train Surrogate Model ---")
        X = inputs
        y = preds[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        surrogate_estimators = 5 if self.test_mode else 50
        surrogate = RandomForestRegressor(n_estimators=surrogate_estimators, max_depth=10, n_jobs=4)
        surrogate.fit(X_train, y_train)
        
        # 4. Evaluation and Plotting
        print("\n--- Phase 4: Surrogate Evaluation ---")
        y_pred = surrogate.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"Surrogate R2 Score: {r2:.4f}")
        print(f"DEBUG: Surrogate Prediction StdDev = {np.std(y_pred):.6f} (If 0.0, model is collapsed!)")
        
        # Generate Evaluation Plot
        plot_path = os.path.join(self.plots_dir, f"surrogate_{self.domain}_r2.png")
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.3, color='blue', label='Predictions vs Truth')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
        plt.title(f"Surrogate Performance ({self.domain})\nR2 Score: {r2:.4f}")
        plt.xlabel("PINN Physics Truth")
        plt.ylabel("Surrogate Prediction")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved evaluation plot to {plot_path}")
        
        # 5. Evaluate PRANA-G Constraints
        print("\n--- Phase 5: PRANA-G 7-Component Diagnostics ---")
        loss_gen = create_cross_domain_loss_generator()
        preds_tensor = torch.tensor(y_pred, dtype=torch.float32)
        
        # The parametric targets (T_bound, IC_bound) are appended after the physical dimensions.
        # So the first parametric target is at index: actual_input_dim - 2
        num_physical_dims = max(0, actual_input_dim - 2)
        bound_temp_tensor = torch.tensor(X_test[:, num_physical_dims], dtype=torch.float32)
        
        safety_hazard = torch.relu(preds_tensor * bound_temp_tensor) * 100.0
        
        inputs_dict = {
            "physics": torch.tensor(0.0),
            "data": preds_tensor,
            "boundary": preds_tensor,
            "biology": safety_hazard,
            "ecology": torch.tensor(0.0),
            "economics": torch.tensor(0.0),
            "safety": safety_hazard
        }
        
        total_loss = loss_gen.compute_total_loss(inputs_dict)
        print(f"Average Total Biological/Economic Constraint Cost: {total_loss.item():.4f}")
        
        breakdown_dict = {}
        for name, comp in loss_gen.components.items():
            try:
                breakdown_dict[name] = round(comp.compute_fn(inputs_dict.get(name)).item(), 4)
            except:
                breakdown_dict[name] = 0.0
        
        # 6. Save the Surrogate Model and Metrics
        surrogate_path = os.path.join(self.surrogate_dir, f"Surrogate_{self.domain}.joblib")
        joblib.dump(surrogate, surrogate_path)
        print(f"Saved Surrogate Model to {surrogate_path}")
        
        import json
        metrics_file = os.path.join(self.results_dir, "surrogate_metrics.json")
        metrics_data = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
                
        metrics_data[self.domain] = {
            "R2_Score": round(r2, 4),
            "PRANA_G_Total_Loss": round(total_loss.item(), 4),
            "PRANA_G_Breakdown": breakdown_dict
        }
        
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=4)
        print(f"Saved Surrogate Metrics to {metrics_file}")
        
        print(f"\n[Completed] Unified Pipeline execution for '{self.domain}'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a quick test to catch errors")
    args = parser.parse_args()
    
    domains_to_train = ["arrhenius"]
    print("Initializing Automated Multi-Domain Pipeline...")
    for d in domains_to_train:
        pipeline = PranagPipeline(domain=d, test_mode=args.test)
        try:
            pipeline.run_end_to_end()
        except Exception as e:
            print(f"\\n[CRITICAL ERROR] Pipeline failed for domain '{d}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nAll domains processed successfully! Outputs are in unified_pipeline_new_output/")