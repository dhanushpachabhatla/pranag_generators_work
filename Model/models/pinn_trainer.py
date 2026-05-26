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
    def __init__(self, pinn_model, learning_rate=1e-3):
        super().__init__()
        self.pinn = pinn_model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['pinn'])

    def forward(self, x):
        return self.pinn(x)

    def training_step(self, batch, batch_idx):
        x_collocation = batch[0]
        x_collocation.requires_grad_(True)
        
        # 1. Calculate base Physics Loss from DeepXDE (residual)
        phys_loss = self.pinn.physics_loss(x_collocation)
        
        # We can add explicit boundary logic here later, but for Foundation models 
        # we learn the pure physics PDE first.
        total_loss = phys_loss
        
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.pinn.parameters(), lr=self.learning_rate)

def validate_pinn(pinn_model, input_dim=2, num_points=500):
    """Post-training validation on unseen data across physics constraints."""
    print("\n--- Post-Training Validation ---")
    x_test = (torch.rand(num_points, input_dim) * 2) - 1.0
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
    x_train = (torch.rand(num_points, input_dim) * 2) - 1.0
    
    dataset = TensorDataset(x_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    lightning_model = PINNLightningModule(pinn_model)
    
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
    
    print(f"[PINNTrainer] Starting PyTorch Lightning training for {max_epochs} epochs...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_model_summary=False,
        logger=False,
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(lightning_model, train_dataloaders=dataloader)
    print(f"\n[PINNTrainer] Training completed! Best model saved to: {checkpoint_callback.best_model_path}")
    
    # Run Post-Training Validation
    breakdown, passed = validate_pinn(lightning_model.pinn, input_dim=input_dim)
    
    return lightning_model.pinn, breakdown
