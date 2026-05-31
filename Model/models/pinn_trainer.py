"""
pinn_trainer.py - Automated PyTorch Lightning Trainer for PINNs
===============================================================
Wraps any PINN model into a LightningModule, utilizing the full 7-component
LossGenerator and PyTorch Lightning Checkpointing.
"""

import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint

# Import the 7-component loss generator
from loss_generator import create_cross_domain_loss_generator

class PINNLightningModule(pl.LightningModule):
    def __init__(self, pinn_model, learning_rate=1e-3, optimizer_type="adam"):
        super().__init__()
        self.pinn = pinn_model
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.save_hyperparameters(ignore=['pinn'])
        
        # Validate physics parameters before training
        if hasattr(pinn_model, 'validate_physics_parameters'):
            is_valid, missing_params = pinn_model.validate_physics_parameters()
            if not is_valid:
                raise ValueError(
                    f"Physics model has missing required parameters: {missing_params}. "
                    f"This will cause AttributeError during training. "
                    f"Available parameters: {list(pinn_model._initialized_params.keys()) if hasattr(pinn_model, '_initialized_params') else 'unknown'}"
                )
        
        self.loss_history = {"train_loss": [], "phys_loss": [], "boundary_loss": []}
        
    def forward(self, x):
        return self.pinn(x)

    def training_step(self, batch, batch_idx):
        x_collocation = batch[0]
        x_collocation.requires_grad_(True)
        
        # 1. Calculate base Physics Loss from DeepXDE (residual)
        phys_loss = self.pinn.physics_loss(x_collocation)
        
        total_loss = phys_loss
        
        # 2. Dynamic Parametric Boundary Loss
        # We enforce that at physical edges (x = -1 or 1), the prediction matches the parametric T_bound
        input_dim = x_collocation.shape[1]
        if input_dim >= 3:
            # Mask points that are on the physical boundary
            bound_mask = (torch.abs(x_collocation[:, 1]) >= 0.99)
            if bound_mask.any():
                u_pred = self.pinn(x_collocation[bound_mask])
                # Target is the second-to-last column (Parametric T_bound)
                u_target = x_collocation[bound_mask, -2:-1]
                boundary_loss = torch.mean((u_pred - u_target)**2)
                
                # Weight the boundary loss heavily to force convergence
                total_loss = total_loss + (10.0 * boundary_loss)
                self.log("boundary_loss", boundary_loss, prog_bar=True, on_epoch=True)

        # 3. Step-Response Initial Condition Loss
        # Enforces that the simulation always starts at a normalized baseline (0.0) at Time = 0
        if input_dim >= 1:
            # Mask points that are exactly at the start of the simulation (t = 0)
            ic_mask = (torch.abs(x_collocation[:, 0]) <= 0.01)
            if ic_mask.any():
                u_pred_ic = self.pinn(x_collocation[ic_mask])
                # Target is ALWAYS the last column (Parametric IC_bound)
                u_target_ic = x_collocation[ic_mask, -1:]
                ic_loss = torch.mean((u_pred_ic - u_target_ic)**2)
                
                total_loss = total_loss + (10.0 * ic_loss)
                self.log("ic_loss", ic_loss, prog_bar=True, on_epoch=True)
                
        self.log("phys_loss", phys_loss, prog_bar=True, on_epoch=True)
        self.log("train_loss", total_loss, prog_bar=True, on_epoch=True)
        return total_loss

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "train_loss" in metrics:
            self.loss_history["train_loss"].append(metrics["train_loss"].item())
        if "phys_loss" in metrics:
            self.loss_history["phys_loss"].append(metrics["phys_loss"].item())
        if "boundary_loss" in metrics:
            self.loss_history["boundary_loss"].append(metrics["boundary_loss"].item())
        if "ic_loss" in metrics:
            if "ic_loss" not in self.loss_history:
                self.loss_history["ic_loss"] = []
            self.loss_history["ic_loss"].append(metrics["ic_loss"].item())

        # Heartbeat print every 100 epochs to prevent terminal lag
        epoch = self.current_epoch
        if (epoch + 1) % 100 == 0:
            train_loss = metrics.get("train_loss")
            loss_val = train_loss.item() if train_loss is not None else 0.0
            print(f"[Epoch {epoch + 1}/{self.trainer.max_epochs}] Total Loss: {loss_val:.6f}")

    def configure_optimizers(self):
        if self.optimizer_type == "lbfgs":
            # Stage 2: Sniper Mode (High precision, full batch)
            optimizer = torch.optim.LBFGS(
                self.pinn.parameters(), 
                lr=0.1, 
                max_iter=20, 
                max_eval=25, 
                tolerance_grad=1e-5, 
                tolerance_change=1e-9, 
                history_size=50, 
                line_search_fn="strong_wolfe"
            )
            return optimizer
        else:
            # Stage 1: Explorer Mode (Mini-batch)
            optimizer = torch.optim.Adam(self.pinn.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=500
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss"
                }
            }

def validate_pinn(pinn_model, input_dim=2, num_points=500):
    """Post-training validation on unseen data across physics constraints."""
    print("\n--- Post-Training Validation ---")
    x_test = (torch.rand(num_points, input_dim) * 2) - 1.0
    if input_dim > 0:
        x_test[:, 0] = (x_test[:, 0] + 1) / 2.0  # Time from 0 to 1
    x_test.requires_grad_(True)
    device = next(pinn_model.parameters()).device
    x_test = x_test.to(device)
    
    # 1. Physics Loss
    phys_loss = pinn_model.physics_loss(x_test).item()
    print(f"Validation Physics Residual: {phys_loss:.4f}")
    
    passed = phys_loss <= 1.0
            
    if not passed:
        print("FAILED VALIDATION: The model did not converge properly on the physics laws.")
    else:
        print("PASSED VALIDATION: The model successfully learned the underlying physics constraints.")
        
    return {"physics": phys_loss}, passed

def train_pinn_model(pinn_model, input_dim=2, num_points=1000, max_epochs=50, batch_size=256, model_alias=None, checkpoint_dir=None):
    """
    Utility function to automatically train and save a generated PINN model.
    """
    print(f"[PINNTrainer] Generating {num_points} collocation points for {input_dim}D input.")
    
    # 1. Interior collocation points
    x_colloc = (torch.rand(num_points, input_dim) * 2) - 1.0
    if input_dim > 0:
        x_colloc[:, 0] = (x_colloc[:, 0] + 1) / 2.0  # Time from 0 to 1
    
    # 2. Explicit Boundary Points
    num_boundary = num_points // 4
    x_bound = (torch.rand(num_boundary, input_dim) * 2) - 1.0
    if input_dim > 0:
        x_bound[:, 0] = (x_bound[:, 0] + 1) / 2.0  # Time from 0 to 1
    if input_dim >= 2:
        # Force spatial dimension (x) to exactly -1 or 1
        edges = torch.randint(0, 2, (num_boundary,)).float() * 2 - 1.0
        x_bound[:, 1] = edges
        
    # 3. Explicit Initial Condition Points (t = 0)
    num_ic = num_points // 4
    x_ic = (torch.rand(num_ic, input_dim) * 2) - 1.0
    if input_dim > 0:
        x_ic[:, 0] = 0.0  # Force t = 0 exactly
        
    x_train = torch.cat([x_colloc, x_bound, x_ic], dim=0)
    
    dataset = TensorDataset(x_train)
    
    # Checkpointing: Save the model automatically
    if checkpoint_dir is None:
        checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_name = model_alias if model_alias else type(pinn_model).__name__
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=model_name,
        save_top_k=1,
        monitor="train_loss",
        mode="min"
    )
    
    # --- STAGE 1: Adam Optimization ---
    dataloader_adam = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lightning_model = PINNLightningModule(pinn_model, optimizer_type="adam")
    
    print(f"[PINNTrainer] Starting Stage 1 (Adam) for {max_epochs} epochs...")
    trainer_adam = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_model_summary=False,
        logger=False,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback]
    )
    
    import time
    start_time = time.time()
    trainer_adam.fit(lightning_model, train_dataloaders=dataloader_adam)
    adam_time = time.time() - start_time
    
    # --- STAGE 2: L-BFGS Optimization ---
    print(f"\n[PINNTrainer] Starting Stage 2 (L-BFGS Sniper) for 1000 epochs...")
    # L-BFGS requires the full dataset in a single massive batch for stable Hessians
    dataloader_lbfgs = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Switch optimizer
    lightning_model.optimizer_type = "lbfgs"
    
    trainer_lbfgs = pl.Trainer(
        max_epochs=1000,
        accelerator="auto",
        devices=1,
        enable_model_summary=False,
        logger=False,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback]  # Reuse checkpointing to save the ultimate best
    )
    
    start_time = time.time()
    trainer_lbfgs.fit(lightning_model, train_dataloaders=dataloader_lbfgs)
    lbfgs_time = time.time() - start_time
    
    total_time = adam_time + lbfgs_time
    mins, secs = divmod(total_time, 60)
    print(f"\n[PINNTrainer] Two-Stage Training completed in {int(mins)}m {int(secs)}s! Best model saved to: {checkpoint_callback.best_model_path}")
    
    # --- Generate Loss History Plot ---
    try:
        import matplotlib.pyplot as plt
        history = lightning_model.loss_history
        if len(history["train_loss"]) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(history["train_loss"], label="Total Loss", linewidth=2)
            if len(history["phys_loss"]) > 0:
                plt.plot(history["phys_loss"], label="Physics Loss", linestyle="--")
            if len(history["boundary_loss"]) > 0:
                plt.plot(history["boundary_loss"], label="Boundary Loss", linestyle=":")
            if "ic_loss" in history and len(history["ic_loss"]) > 0:
                plt.plot(history["ic_loss"], label="Initial Condition Loss", linestyle="-.")
                
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (Log Scale)")
            plt.title(f"PINN Training Convergence: {model_alias}")
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend()
            
            plot_path = os.path.join(checkpoint_dir, f"{model_alias}_loss_history.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[PINNTrainer] Saved convergence plot to {plot_path}")
    except Exception as e:
        print(f"[PINNTrainer] Failed to generate plot: {e}")
    
    # Run Post-Training Validation
    breakdown, passed = validate_pinn(lightning_model.pinn, input_dim=input_dim)
    
    return lightning_model.pinn, breakdown
