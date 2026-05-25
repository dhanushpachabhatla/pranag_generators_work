import torch
import deepxde as dde
import sys
import os

# Ensure the project root is in the path so we can import Model modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.models.sympy_loss_generator import SymPyLossGenerator

def main():
    print("Initializing SymPyLossGenerator...")
    gen = SymPyLossGenerator()

    # Define a 1D Heat Equation: u_t - 0.5 * u_xx = 0
    pde_str = "u_t - 0.5 * u_xx"
    input_vars = ["x", "t"]  # index 0 is x, index 1 is t
    
    print(f"\nCompiling PDE: {pde_str}")
    pde_fn = gen.compile_pde(pde_str, input_vars=input_vars, output_var="u")
    print("SUCCESS: PDE successfully compiled into a DeepXDE function.")

    print("\nCreating DeepXDE Geometry and PDE Data...")
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    
    data = gen.create_pde_data(
        pde_string=pde_str,
        input_vars=input_vars,
        geom=geom,
        time_domain=timedomain,
        num_domain=10,
        num_boundary=2,
        num_initial=2
    )
    print("SUCCESS: DeepXDE TimePDE data object created successfully.")
    
    # We can do a quick manual check of the compiled function
    # x is a tensor of shape (N, 2), y is a tensor of shape (N, 1)
    print("\nTesting the generated pde_fn with dummy tensors (requires gradient tracking)...")
    
    # Create dummy inputs requiring gradients
    x = torch.rand((5, 2), requires_grad=True)
    # Dummy network output 
    y = x[:, 0:1]**2 + x[:, 1:2]  # y = x^2 + t
    
    # Evaluate the PDE residual: u_t - 0.5 * u_xx
    # dy/dt = 1
    # dy/dx = 2x, d^2y/dx^2 = 2
    # So residual should be 1 - 0.5 * 2 = 0
    residual = pde_fn(x, y)
    
    print(f"Computed PDE Residuals:\n{residual.detach()}")
    
    # Expecting residuals near zero for this specific dummy y
    expected = torch.zeros_like(residual)
    if torch.allclose(residual, expected, atol=1e-5):
        print("\nSUCCESS: The compiled PDE function calculated the correct symbolic derivatives via DeepXDE!")
    else:
        print("\nWARNING: The residuals did not match expected values.")

if __name__ == "__main__":
    main()
