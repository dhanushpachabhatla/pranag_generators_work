import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Model', 'models')))
import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
from Model.models.pinn_factory import PINNFactory
from Model.models.pinn_trainer import PINNLightningModule

def analytical_heat_solution(t, x, alpha=0.05, terms=50):
    """
    Computes the exact analytical Ground Truth using the Fourier Series.
    Domain: x in [-1, 1]
    Boundary Conditions: u(t, -1) = 0, u(t, 1) = 0
    Initial Condition: u(0, x) = 1.0
    
    The Fourier series for this specific square wave initial condition is:
    u(t, x) = SUM_{n=0}^{inf} (-1)^n * (4 / ((2n+1)*pi)) * cos((2n+1)*pi*x / 2) * exp(-alpha * ((2n+1)*pi/2)^2 * t)
    """
    u_exact = np.zeros_like(x)
    for n in range(terms):
        lam = (2 * n + 1) * np.pi / 2.0
        coef = ((-1)**n) * (4.0 / ((2 * n + 1) * np.pi))
        spatial = np.cos(lam * x)
        temporal = np.exp(-alpha * (lam**2) * t)
        u_exact += coef * spatial * temporal
    return u_exact

def run_validation():
    print("--- PRANA-G Analytical Validation Suite ---")
    
    # 1. Load the trained PINN Architecture
    factory = PINNFactory()
    # Note: Parametric Heat PINN uses 4 inputs: (t, x, T_bound, IC_bound)
    # The checkpoint was trained with hidden_dim=128 and num_layers=6
    pinn_model = factory.create("heat", input_dim=4, num_layers=6, hidden_dim=128, dynamic=True)
    
    ckpt_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        "unified_pipeline_output", "pinn", "Parametric_HeatPINN-v1.ckpt"
    ))
    
    if not os.path.exists(ckpt_path):
        # Fallback to the non-v1 checkpoint if v1 doesn't exist
        ckpt_path = ckpt_path.replace("-v1", "")
        
    print(f"Loading checkpoint: {ckpt_path}")
    import torch
    checkpoint = torch.load(ckpt_path, weights_only=False)
    lightning_model = PINNLightningModule(pinn_model=pinn_model)
    lightning_model.load_state_dict(checkpoint['state_dict'])
    lightning_model.eval()
    trained_pinn = lightning_model.pinn
    
    # 2. Generate Validation Grid
    # We want to test at T_bound = 0.0 (walls are at 0 degrees)
    # This matches our Analytical Fourier constraints.
    t_vals = np.linspace(0.01, 1.0, 50)  # Avoid exact t=0 for Fourier convergence stability
    x_vals = np.linspace(-0.99, 0.99, 50)
    T, X = np.meshgrid(t_vals, x_vals, indexing='ij')
    
    T_flat = T.ravel()
    X_flat = X.ravel()
    # T_bound is set to 0.0 to match analytical BCs
    # IC_bound is set to 1.0 to match analytical initial condition of u(0, x) = 1
    B_flat = np.zeros_like(T_flat)
    IC_flat = np.ones_like(T_flat)
    
    inputs_np = np.column_stack([T_flat, X_flat, B_flat, IC_flat])
    inputs_tensor = torch.tensor(inputs_np, dtype=torch.float32)
    
    # 3. Predict using PINN
    with torch.no_grad():
        preds = trained_pinn(inputs_tensor).numpy().ravel()
        
    # 4. Calculate Ground Truth using Fourier
    # The default alpha in HeatPINN is 0.05
    alpha = trained_pinn.alpha if hasattr(trained_pinn, 'alpha') else 0.05
    truth = analytical_heat_solution(T_flat, X_flat, alpha=alpha)
    
    # 5. Compute L2 Error Norm (RMSE)
    mse = np.mean((preds - truth)**2)
    rmse = np.sqrt(mse)
    
    print(f"Validation complete.")
    print(f"PINN vs Analytical Ground Truth (Fourier) L2 RMSE: {rmse:.6f}")
    
    # 6. Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(truth, preds, alpha=0.5, color='blue', s=2)
    plt.plot([truth.min(), truth.max()], [truth.min(), truth.max()], 'r--', lw=2)
    plt.title(f"PINN vs Fourier Ground Truth\nRMSE = {rmse:.6f}")
    plt.xlabel("True Analytical Solution (u)")
    plt.ylabel("PINN Prediction (u)")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    time_idx = 25 # halfway through time
    plt.plot(x_vals, truth.reshape(50, 50)[time_idx, :], label='Fourier Truth', color='black', lw=2)
    plt.plot(x_vals, preds.reshape(50, 50)[time_idx, :], label='PINN Prediction', color='red', linestyle='dashed', lw=2)
    plt.title(f"Cross-section at t={t_vals[time_idx]:.2f}")
    plt.xlabel("Spatial coordinate (x)")
    plt.ylabel("Temperature (u)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "heat_analytical_validation.png"))
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved validation plot to {out_path}")
    
if __name__ == "__main__":
    run_validation()
