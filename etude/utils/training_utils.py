# etude/utils/training_utils.py

import torch
import random
import numpy as np
import shutil
from pathlib import Path

def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def save_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    is_epoch_end: bool = False
):
    """Saves a training checkpoint."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'global_step': global_step,
        'model_config': model.config.to_dict()
    }
    latest_path = run_dir / "latest.pth"
    torch.save(payload, latest_path)
    
    if is_epoch_end:
        epoch_path = run_dir / f"epoch_{(epoch + 1):04d}.pth"
        shutil.copyfile(latest_path, epoch_path)
        print(f"[INFO] Epoch-specific checkpoint saved to {epoch_path}")

def load_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str
) -> tuple[int, int]:
    """Loads a training checkpoint."""
    start_epoch, global_step = 0, 0
    checkpoint_path = run_dir / "latest.pth"
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in payload:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in payload:
            scheduler.load_state_dict(payload['scheduler_state_dict'])
        
        global_step = payload.get('global_step', 0)
        start_epoch = payload.get('epoch', -1) + 1 # Resume from the next epoch
        print(f"[RESUME] Checkpoint loaded. Resuming training from Epoch {start_epoch}")
        return start_epoch, global_step
    
    print("[INFO] No checkpoint found. Starting training from scratch.")
    return start_epoch, global_step