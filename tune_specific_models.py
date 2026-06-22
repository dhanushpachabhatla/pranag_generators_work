import os
import sys
import torch
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import qmc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Model', 'models')))
from training.pinn_factory import PINNFactory

factory = PINNFactory()

out_dir = "unified_pipeline_new_output_1/experiment_surrogates"
os.makedirs(out_dir, exist_ok=True)

configs = {
    "navier_stokes": {"points": 200000, "depth": 22, "estimators": 60},
    "maxwell": {"points": 300000, "depth": 25, "estimators": 60}
}

for d, cfg in configs.items():
    print(f"\n================ TUNING SURROGATE FOR {d} ================")
    pinn_path = f"unified_pipeline_new_output_1/pinn/Parametric_{d.capitalize()}PINN.pt"
    
    if not os.path.exists(pinn_path):
        print(f"Skipping {d}, PINN not found.")
        continue
        
    from Pipeline_New.data_generators import SimulationGenerator
    cfg = SimulationGenerator().from_hint(d)
    actual_input_dim = len(cfg.equation_info.independent) + 4
    
    pinn_model = factory.create(d, input_dim=actual_input_dim, hidden_dim=128, num_layers=6, dynamic=True)
    pinn_model.base_dim = len(cfg.equation_info.independent)
    try:
        pinn_model.load_state_dict(torch.load(pinn_path, map_location=torch.device('cpu'), weights_only=True))
    except TypeError:
        pinn_model.load_state_dict(torch.load(pinn_path, map_location=torch.device('cpu')))
    pinn_model.eval()
    
    actual_input_dim = getattr(pinn_model, "input_dim", 3) # fallback
    
    if hasattr(pinn_model, 'fc1'):
        actual_input_dim = pinn_model.fc1.in_features
    
    print(f"Generating {cfg['points']} LHS points (input_dim={actual_input_dim})...")
    sampler = qmc.LatinHypercube(d=actual_input_dim)
    sample = sampler.random(n=cfg['points'])
    inputs = 2.0 * sample - 1.0
    if actual_input_dim > 0:
        inputs[:, 0] = (inputs[:, 0] + 1.0) / 2.0
    inputs_tensor = torch.tensor(inputs, dtype=float).float() # robust
    
    with torch.no_grad():
        preds = pinn_model(inputs_tensor).cpu().numpy()
        
    # Standardize shape for scikit learn
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
        
    X_train, X_test, y_train, y_test = train_test_split(inputs, preds, test_size=0.2, random_state=42)
    
    print(f"Training RF: depth={cfg['depth']}, estimators={cfg['estimators']} ...")
    surrogate = RandomForestRegressor(n_estimators=cfg['estimators'], max_depth=cfg['depth'], n_jobs=4)
    surrogate.fit(X_train, y_train)
    
    y_pred = surrogate.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    save_path = os.path.join(out_dir, f"Surrogate_{d}.joblib")
    joblib.dump(surrogate, save_path)
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[{d}] R2 Score: {r2:.4f} | Size: {size_mb:.2f} MB")
