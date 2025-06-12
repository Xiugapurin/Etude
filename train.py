# train.py

import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from pathlib import Path
import json
import random
import numpy as np
from tqdm import tqdm
import os
import time
import math
import sys
import traceback
import shutil
from datetime import datetime
from typing import Dict, Optional

from corpus import Vocab, EtudeDataset
from models import EtudeDecoder, EtudeDecoderConfig


# --- Configuration ---
PREPROCESSED_DIR = Path("./dataset/tokenized/")
VOCAB_PATH = PREPROCESSED_DIR / "vocab.json"
CHECKPOINT_DIR = Path("./checkpoint/decoder/")
BASE_CHECKPOINT_DIR = Path("./checkpoint/decoder/")
MODEL_CONFIG_SAVE_PATH = PREPROCESSED_DIR / "etude_decoder_config.json"

PAD_TOKEN = "<PAD>"
ATTRIBUTE_PAD_ID = 0
PAD_CLASS_ID = 0
SEED = 1234
LEARNING_RATE = 2e-4
ADAM_B1 = 0.9
ADAM_B2 = 0.98
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 10
# [MODIFIED] Update number of epochs
NUM_EPOCHS = 800
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
CLIP_GRAD_NORM = 1.0

# Dataset related configs
AVG_NOTE_OVERLAP_THRESHOLD = 0.5
# [MODIFIED] Update context window to 6 bars
CONTEXT_NUM_PAST_XY_PAIRS = 6
# [MODIFIED] Set bin counts for different attributes
NUM_ATTRIBUTE_BINS_DEFAULT = 5  # For most attributes
NUM_BINS_AVG_NOTE_OVERLAP = 3     # Specific for the new attribute

MODEL_PARAMS = {
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "intermediate_size": 512 * 4,
    "hidden_act": "gelu",
    "max_position_embeddings": 1024,
    "dropout_prob": 0.1,
    "num_classes": 3,
    "pad_class_id": PAD_CLASS_ID,
    "rotary_pct": 0.25,
    "rotary_emb_base": 10000,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-5,
    "use_cache": True,
    "bos_token_id": None,
    "eos_token_id": None,
    "tie_word_embeddings": False,

    # [MODIFIED] Add new attribute and update others to use new variables
    "num_avg_note_overlap_bins": NUM_BINS_AVG_NOTE_OVERLAP,
    "avg_note_overlap_emb_dim": 64,
    "num_pitch_coverage_bins": NUM_ATTRIBUTE_BINS_DEFAULT,
    "pitch_coverage_emb_dim": 64,
    "num_note_per_pos_bins": NUM_ATTRIBUTE_BINS_DEFAULT,
    "note_per_pos_emb_dim": 64,
    "num_pitch_class_entropy_bins": NUM_ATTRIBUTE_BINS_DEFAULT,
    "pitch_class_entropy_emb_dim": 64,

    "attribute_pad_id": ATTRIBUTE_PAD_ID,
    "context_num_past_xy_pairs": CONTEXT_NUM_PAST_XY_PAIRS
}

# Runtime Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_SAVE_FORMAT = 'npy'
DATASET_MAX_SEQ_LEN = MODEL_PARAMS["max_position_embeddings"]
NUM_WORKERS_DATALOADER = 1
# [MODIFIED] Set checkpoint save frequency
CHECKPOINT_SAVE_EPOCHS = 25
LOGGING_STEPS = 10000

# --- Utility Functions ---
def set_seed(seed_value):
    random.seed(seed_value); np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)


def write_log(log_file_path: Path, log_data: Dict):
    """Appends a log entry (dictionary) to a JSON Lines file."""
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data) + '\n')
    except Exception as e:
        print(f"Error writing to log file {log_file_path}: {e}")


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, run_checkpoint_dir: Path, base_checkpoint_dir: Path, is_epoch_end: bool = False):
    """Saves model, optimizer, scheduler state, epoch, and step."""
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_in_run_path = run_checkpoint_dir / "latest.pth"
    temp_latest_run_path = run_checkpoint_dir / "latest.pth.tmp"
    
    latest_in_base_path = base_checkpoint_dir / "latest.pth"
    temp_latest_base_path = base_checkpoint_dir / "latest.pth.tmp"

    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'global_step': global_step,
        'model_config': model.config.to_dict(),
    }

    try:
        torch.save(payload, temp_latest_run_path)
        if latest_in_run_path.exists(): os.remove(latest_in_run_path)
        os.replace(temp_latest_run_path, latest_in_run_path)
        
        # [MODIFIED] is_epoch_end now controls if a numbered checkpoint is saved
        if is_epoch_end:
            epoch_specific_save_path = run_checkpoint_dir / f"epoch_{(epoch + 1):04d}_gs{global_step}.pth"
            shutil.copyfile(latest_in_run_path, epoch_specific_save_path)
            print(f"Epoch-specific checkpoint saved to {epoch_specific_save_path}")

        torch.save(payload, temp_latest_base_path)
        if latest_in_base_path.exists(): os.remove(latest_in_base_path)
        os.replace(temp_latest_base_path, latest_in_base_path)

    except Exception as e: 
        print(f"Error saving checkpoint: {e}")
        traceback.print_exc(file=sys.stderr)
    finally:
        if temp_latest_run_path.exists(): os.remove(temp_latest_run_path)
        if temp_latest_base_path.exists(): os.remove(temp_latest_base_path)


def load_checkpoint(model, optimizer, scheduler, specific_run_dir: Optional[Path], base_checkpoint_dir: Path, device: str):
    start_epoch, global_step = 0, 0
    checkpoint_to_load = None
    if specific_run_dir and (specific_run_dir / "latest.pth").exists():
        checkpoint_to_load = specific_run_dir / "latest.pth"
    elif (base_checkpoint_dir / "latest.pth").exists():
        checkpoint_to_load = base_checkpoint_dir / "latest.pth"
    
    if checkpoint_to_load and checkpoint_to_load.exists():
        print(f"Loading checkpoint from {checkpoint_to_load}")
        try:
            payload = torch.load(checkpoint_to_load, map_location=device)
            
            model_state = payload.get('model_state_dict', payload.get('model', payload))
            if isinstance(model_state, dict):
                new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in model_state.items()}
                model_state = new_state_dict
            
            load_result = model.load_state_dict(model_state, strict=False)
            print(f"  Model load result: Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")

            if optimizer and 'optimizer_state_dict' in payload:
                try: optimizer.load_state_dict(payload['optimizer_state_dict']); print("  Optimizer state loaded.")
                except: print("  Warning: Could not load optimizer state. Optimizer reinitialized.")
            if scheduler and 'scheduler_state_dict' in payload and payload['scheduler_state_dict']:
                try: scheduler.load_state_dict(payload['scheduler_state_dict']); print("  Scheduler state loaded.")
                except: print("  Warning: Could not load scheduler state. Scheduler reinitialized.")
            
            saved_global_step = payload.get('global_step', 0)
            saved_epoch = payload.get('epoch', -1)

            if saved_global_step > 0 :
                global_step = saved_global_step
                start_epoch = saved_epoch + 1

            print(f"  Checkpoint loaded. Resuming/Starting from Epoch {start_epoch}, Global Step {global_step}")

        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_to_load}: {e}. Training from scratch.", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            start_epoch = 0; global_step = 0
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    return start_epoch, global_step


# --- Main Training Function ---
def train(run_id: Optional[str] = None):
    set_seed(SEED)

    run_id_to_use = run_id if run_id else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    current_run_checkpoint_dir = BASE_CHECKPOINT_DIR / run_id_to_use
    current_run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Current Run ID: {run_id_to_use}")
    print(f"Run-specific checkpoints & logs: {current_run_checkpoint_dir.resolve()}")

    log_file_path = current_run_checkpoint_dir / "train_log.jsonl"
    print(f"Training log: {log_file_path}")
    
    current_run_model_params = MODEL_PARAMS.copy()

    print(f"Using device: {DEVICE}")

    try:
        vocab = Vocab.load(VOCAB_PATH)
        current_run_model_params['vocab_size'] = len(vocab)
        current_run_model_params['pad_token_id'] = vocab.get_pad_id()
        if current_run_model_params['pad_token_id'] == -1:
            raise ValueError(f"{PAD_TOKEN} not found in vocabulary.")
        print(f"Vocabulary loaded: {len(vocab)} tokens, PAD ID: {vocab.get_pad_id()}")
    except Exception as e:
        print(f"Error loading vocab: {e}", file=sys.stderr); sys.exit(1)

    try:
        model_config_for_run = EtudeDecoderConfig(**current_run_model_params)
        run_specific_model_config_path = current_run_checkpoint_dir / f"{model_config_for_run.model_type}_config_used.json"
        with open(run_specific_model_config_path, 'w') as f:
            json.dump(model_config_for_run.to_dict(), f, indent=2)
        print(f"Model configuration for this run saved to {run_specific_model_config_path}")
    except Exception as e:
        print(f"Error creating/saving model config: {e}", file=sys.stderr); sys.exit(1)

    try:
        dataset = EtudeDataset(
            dataset_dir=PREPROCESSED_DIR, 
            vocab=vocab, 
            max_seq_len=model_config_for_run.max_seq_len,
            data_format=DATA_SAVE_FORMAT,
            num_attribute_bins=NUM_ATTRIBUTE_BINS_DEFAULT,
            context_num_past_xy_pairs=CONTEXT_NUM_PAST_XY_PAIRS,
            verbose_stats=False
        )
        if len(dataset) == 0: print("Error: Dataset is empty."); sys.exit(1)
        print(f"Dataset loaded: {len(dataset)} training samples.")
        dataloader = dataset.get_dataloader(BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DATALOADER)
        print(f"DataLoader created. Batches per epoch: {len(dataloader)}")
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr); sys.exit(1)

    model = EtudeDecoder(model_config_for_run).to(DEVICE)
    print(f"Model initialized: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params.")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(ADAM_B1, ADAM_B2), weight_decay=WEIGHT_DECAY)
    
    if len(dataloader) == 0:
        print("Error: DataLoader is empty.", file=sys.stderr); sys.exit(1)
        
    num_update_steps_per_epoch = math.ceil(len(dataloader) / GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = num_update_steps_per_epoch * NUM_EPOCHS
    warmup_steps = WARMUP_EPOCHS * num_update_steps_per_epoch
    
    if total_training_steps == 0:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_training_steps
        )
    print(f"Scheduler: Total steps={total_training_steps}, Warmup steps={warmup_steps}")

    start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, current_run_checkpoint_dir, BASE_CHECKPOINT_DIR, DEVICE)
    
    model.train()
    total_loss_for_log_period = 0.0 
    num_optimizer_steps_in_log_period = 0
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    print("\n--- Starting Training ---")
    training_start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        num_optimizer_steps_this_epoch = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch", dynamic_ncols=True)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if not batch or batch['input_ids'].numel() == 0:
                continue

            try:
                # [MODIFIED] Load all four attributes from the batch
                input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
                attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
                class_ids = batch['class_ids'].to(DEVICE, non_blocking=True)
                labels = batch['labels'].to(DEVICE, non_blocking=True)
                
                avg_note_overlap_bin_ids = batch['note_overlap_bin_ids'].to(DEVICE, non_blocking=True) # Key from dataset.py
                pitch_coverage_bin_ids = batch['pitch_coverage_bin_ids'].to(DEVICE, non_blocking=True)
                note_per_pos_bin_ids = batch['note_per_pos_bin_ids'].to(DEVICE, non_blocking=True)
                pitch_class_entropy_bin_ids = batch['pitch_class_entropy_bin_ids'].to(DEVICE, non_blocking=True)

            except KeyError as ke:
                print(f"KeyError: {ke}. Available keys: {batch.keys()}", file=sys.stderr)
                continue
            except Exception as e_batch:
                print(f"Error processing batch: {e_batch}", file=sys.stderr); continue

            with torch.amp.autocast(device_type=DEVICE.split(':')[0], enabled=(DEVICE.startswith("cuda"))):
                # [MODIFIED] Pass all four attributes to the model
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    class_ids=class_ids,
                    avg_note_overlap_bin_ids=avg_note_overlap_bin_ids,
                    pitch_coverage_bin_ids=pitch_coverage_bin_ids,
                    note_per_pos_bin_ids=note_per_pos_bin_ids,
                    pitch_class_entropy_bin_ids=pitch_class_entropy_bin_ids,
                    labels=labels, 
                    return_dict=True
                )
                loss = outputs.loss

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at epoch {epoch}, batch {batch_idx}. Skipping step.", file=sys.stderr)
                optimizer.zero_grad(set_to_none=True)
                continue 
            
            loss_value = loss.item()
            epoch_loss_sum += loss_value
            total_loss_for_log_period += loss_value
            
            scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                global_step += 1
                num_optimizer_steps_this_epoch +=1
                num_optimizer_steps_in_log_period +=1

                if global_step > 0 and global_step % LOGGING_STEPS == 0:
                    if num_optimizer_steps_in_log_period > 0:
                         avg_loss_for_log = total_loss_for_log_period / num_optimizer_steps_in_log_period
                         current_lr = optimizer.param_groups[0]['lr']
                         pbar.set_postfix({"Loss": f"{avg_loss_for_log:.4f}", "LR": f"{current_lr:.3e}"})
                         write_log(log_file_path, {"type": "step_log", "global_step": global_step, "epoch": epoch, "avg_loss_period": avg_loss_for_log, "learning_rate": current_lr})
                    total_loss_for_log_period = 0.0
                    num_optimizer_steps_in_log_period = 0
        
        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss_sum / len(dataloader) if len(dataloader) > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time
        current_lr_end_epoch = optimizer.param_groups[0]['lr']
        
        pbar.close()
        print(f"End of Epoch {epoch+1}/{NUM_EPOCHS}. Avg Batch Loss: {avg_epoch_loss:.4f}. LR: {current_lr_end_epoch:.3e}. Duration: {epoch_duration:.2f}s")
        
        write_log(log_file_path, {"type": "epoch_log", "epoch": epoch, "global_step": global_step, "avg_epoch_loss": avg_epoch_loss, "learning_rate_end_epoch": current_lr_end_epoch})
        
        # [MODIFIED] Save checkpoint based on epoch frequency
        is_save_epoch = ((epoch + 1) % CHECKPOINT_SAVE_EPOCHS == 0) or ((epoch + 1) == NUM_EPOCHS)
        # Always update latest.pth, but only save numbered checkpoint on save epochs
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, current_run_checkpoint_dir, BASE_CHECKPOINT_DIR, is_epoch_end=is_save_epoch)

    # --- Training Finished ---
    print("\n--- Training Finished ---")
    total_duration_sec = time.time() - training_start_time
    print(f"Total training time: {total_duration_sec//3600:.0f}h {(total_duration_sec%3600)//60:.0f}m {total_duration_sec%60:.0f}s")


if __name__ == "__main__":
    train()