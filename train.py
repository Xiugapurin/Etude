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
NUM_EPOCHS = 400 # 總訓練 epoch 數
BATCH_SIZE = 8    # 每個 GPU 的 batch size (如果單 GPU 則是總 batch size)
GRADIENT_ACCUMULATION_STEPS = 4
CLIP_GRAD_NORM = 1.0

# Dataset 相關配置 (與 EtudeDataset 的 __init__ 參數對應)
AVG_NOTE_OVERLAP_THRESHOLD = 0.5
CONTEXT_NUM_PAST_XY_PAIRS = 2 # EtudeDataset 中上下文回看的 (X,Y) 對數量
NUM_ATTRIBUTE_BINS = 5        # 所有屬性統一為 5 個 bin

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

    # 更新後的屬性 (num_bins 設為 5)
    "num_pitch_coverage_bins": NUM_ATTRIBUTE_BINS,
    "pitch_coverage_emb_dim": 64,
    "num_note_per_pos_bins": NUM_ATTRIBUTE_BINS,
    "note_per_pos_emb_dim": 64,
    "num_pitch_class_entropy_bins": NUM_ATTRIBUTE_BINS,
    "pitch_class_entropy_emb_dim": 64,

    "attribute_pad_id": ATTRIBUTE_PAD_ID,
    "context_num_past_xy_pairs": CONTEXT_NUM_PAST_XY_PAIRS
}

# Runtime Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_SAVE_FORMAT = 'npy'
DATASET_MAX_SEQ_LEN = MODEL_PARAMS["max_position_embeddings"]
NUM_WORKERS_DATALOADER = 1
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
        'epoch': epoch, # 保存的是當前已完成的 epoch (0-indexed)
        'global_step': global_step,
        'model_config': model.config.to_dict(),
    }

    try:
        # 1. 總是保存/更新 latest.pth 到運行專用目錄
        torch.save(payload, temp_latest_run_path)
        if latest_in_run_path.exists(): os.remove(latest_in_run_path)
        os.replace(temp_latest_run_path, latest_in_run_path)

        # 2. 如果是在 epoch 結束，則保存一個 epoch 特定編號的 checkpoint
        if is_epoch_end:
            epoch_specific_save_path = run_checkpoint_dir / f"epoch_{(epoch + 1):04d}_gs{global_step}.pth"
            shutil.copyfile(latest_in_run_path, epoch_specific_save_path)
            print(f"Epoch-specific checkpoint saved to {epoch_specific_save_path}")

        # 3. 總是更新主 checkpoint 目錄下的 latest.pth 副本
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
            payload = torch.load(checkpoint_to_load, map_location=device) # map_location
            
            # 模型配置比較 (可選但推薦)
            if 'model_config' in payload and payload['model_config'] is not None:
                loaded_config_dict = payload['model_config']
                current_config_dict = model.config.to_dict()
                # ... (您的配置比較邏輯) ...
                print("  Loaded config from checkpoint. Current model config will be used.")

            # 載入模型狀態字典 (使用 strict=False 以處理可能的結構差異)
            model_state = payload.get('model_state_dict', payload.get('model', payload.get('state_dict', payload)))
            if isinstance(model_state, dict): # 處理 DDP 可能添加的 'module.' 前綴
                new_state_dict = {}
                for k, v in model_state.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model_state = new_state_dict
            
            load_result = model.load_state_dict(model_state, strict=False)
            print(f"  Model load result: Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
            if load_result.missing_keys and not load_result.unexpected_keys:
                 print("  Note: Missing keys usually okay if new layers were added (they'll be initialized).")
            if load_result.unexpected_keys:
                 print("  Warning: Unexpected keys found, check if model structure changed significantly from checkpoint.")


            if optimizer and 'optimizer_state_dict' in payload:
                try: optimizer.load_state_dict(payload['optimizer_state_dict']); print("  Optimizer state loaded.")
                except: print("  Warning: Could not load optimizer state. Optimizer reinitialized.")
            if scheduler and 'scheduler_state_dict' in payload and payload['scheduler_state_dict']:
                try: scheduler.load_state_dict(payload['scheduler_state_dict']); print("  Scheduler state loaded.")
                except: print("  Warning: Could not load scheduler state. Scheduler reinitialized.")
            
            # 恢復 epoch 和 global_step
            # 如果 global_step > 0, 表示至少完成了一個 step，我們從該 epoch 繼續，或下一個 epoch
            saved_global_step = payload.get('global_step', 0)
            saved_epoch = payload.get('epoch', -1) # -1 表示從0開始

            if saved_global_step > 0 : # Resuming a run
                global_step = saved_global_step
                start_epoch = saved_epoch + 1 if payload.get('batch',-1) < 0 or payload.get('batch_is_end_of_epoch', True) else saved_epoch

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

    run_id_to_use = run_id
    if run_id_to_use is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id_to_use = f"run_{timestamp}"
    
    current_run_checkpoint_dir = BASE_CHECKPOINT_DIR / run_id_to_use # 用於保存此 run 的 checkpoints 和 logs
    current_run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Current Run ID: {run_id_to_use}")
    print(f"Run-specific checkpoints & logs: {current_run_checkpoint_dir.resolve()}")

    log_file_path = current_run_checkpoint_dir / "train_log.jsonl"
    print(f"Training log: {log_file_path}")
    
    # 將 MODEL_PARAMS 和其他訓練超參複製一份用於保存，避免修改全域 MODEL_PARAMS
    current_run_model_params = MODEL_PARAMS.copy()

    print(f"Using device: {DEVICE}")

    # --- Load Vocabulary & Update Model Params ---
    try:
        vocab = Vocab.load(VOCAB_PATH)
        current_run_model_params['vocab_size'] = len(vocab)
        current_run_model_params['pad_token_id'] = vocab.get_pad_id()

        if current_run_model_params['pad_token_id'] == -1:
            raise ValueError(f"{PAD_TOKEN} not found in vocabulary loaded from {VOCAB_PATH}.")
        print(f"Vocabulary loaded: {current_run_model_params['vocab_size']} tokens, PAD ID: {current_run_model_params['pad_token_id']}")
    except Exception as e:
        print(f"Error loading vocab from {VOCAB_PATH}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr); sys.exit(1)

    # --- Prepare Model Configuration & Save It (associated with this run) ---
    model_config_for_run = None
    try:
        model_config_for_run = EtudeDecoderConfig(**current_run_model_params)
        run_specific_model_config_path = current_run_checkpoint_dir / f"{model_config_for_run.model_type}_config_used.json"
        
        with open(run_specific_model_config_path, 'w') as f:
            json.dump(model_config_for_run.to_dict(), f, indent=2)
        print(f"Model configuration for this run saved to {run_specific_model_config_path}")
    except Exception as e:
        print(f"Error creating/saving model config: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr); sys.exit(1)

    try:
        dataset = EtudeDataset(
            dataset_dir=PREPROCESSED_DIR, 
            vocab=vocab, 
            max_seq_len=model_config_for_run.max_seq_len,
            cond_suffix=f'_cond.{DATA_SAVE_FORMAT}', 
            tgt_suffix=f'_tgt.{DATA_SAVE_FORMAT}',
            data_format=DATA_SAVE_FORMAT,
            num_attribute_bins=NUM_ATTRIBUTE_BINS,
            avg_note_overlap_threshold=AVG_NOTE_OVERLAP_THRESHOLD,
            context_num_past_xy_pairs=CONTEXT_NUM_PAST_XY_PAIRS,
            verbose_stats=False
        )
        if len(dataset) == 0: print("Error: Dataset is empty after initialization."); sys.exit(1)
        print(f"Dataset loaded: {len(dataset)} training samples (chunks).")
        dataloader = dataset.get_dataloader(BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DATALOADER)
        print(f"DataLoader created. Batches per epoch: {len(dataloader)}")
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr); sys.exit(1)

    # --- Initialize Model, Optimizer, Scheduler ---
    model = EtudeDecoder(model_config_for_run).to(DEVICE) # 使用 run specific config
    print(f"Model initialized: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params.")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(ADAM_B1, ADAM_B2), weight_decay=WEIGHT_DECAY)
    
    # 計算總更新步數和 warmup 步數
    # 如果 len(dataloader) 為0 (例如 dataset 為空), num_update_steps_per_epoch 會是0
    if len(dataloader) == 0 :
        print("Error: DataLoader is empty, cannot proceed with training.", file=sys.stderr)
        sys.exit(1)
        
    num_update_steps_per_epoch = math.ceil(len(dataloader) / GRADIENT_ACCUMULATION_STEPS)
    if num_update_steps_per_epoch == 0: num_update_steps_per_epoch = 1 # 避免 total_training_steps 為0

    total_training_steps = num_update_steps_per_epoch * NUM_EPOCHS
    warmup_steps = WARMUP_EPOCHS * num_update_steps_per_epoch
    
    # 確保 warmup_steps 不超過 total_training_steps
    if warmup_steps >= total_training_steps and total_training_steps > 0 :
        print(f"Warning: Warmup steps ({warmup_steps}) >= Total training steps ({total_training_steps}). Adjusting warmup.")
        warmup_steps = max(1, int(total_training_steps * 0.1)) # 例如，設為總步數的10%
    elif warmup_steps == 0 and total_training_steps > 0 : # 至少1步warmup
        warmup_steps = 1


    if total_training_steps == 0:
        print("Warning: Total training steps is zero. Scheduler might not work as expected.")
        # 創建一個不執行任何操作的 dummy scheduler
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_training_steps
        )
    print(f"Scheduler: Total effective steps={total_training_steps}, Warmup effective steps={warmup_steps}")

    # --- Load Checkpoint ---
    start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, current_run_checkpoint_dir, BASE_CHECKPOINT_DIR, DEVICE)

    if start_epoch > 0 and scheduler is not None and hasattr(scheduler, 'last_epoch'):
        print(f"Scheduler last_epoch might be implicitly set by loaded state or needs {global_step} steps.")


    # --- Training Loop ---
    model.train()
    total_loss_for_log_period = 0.0 
    num_optimizer_steps_in_log_period = 0
    
    # scaler 用於混合精度訓練 (如果使用 CUDA)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    print("\n--- Starting Training ---")
    training_start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        num_optimizer_steps_this_epoch = 0 # 計算本 epoch 實際執行的 optimizer.step() 次數

        # DistributedSampler set_epoch (如果使用 DDP)
        # if ca.num_gpus > 1 and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        #     dataloader.sampler.set_epoch(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch", dynamic_ncols=True)
        optimizer.zero_grad() # 確保在 epoch 開始時梯度是清零的

        for batch_idx, batch in enumerate(pbar):
            if not batch or batch['input_ids'].numel() == 0: 
                print(f"Warning: Skipping empty batch at epoch {epoch}, batch_idx {batch_idx}")
                continue

            # 將數據移動到指定設備
            try:
                input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
                attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
                class_ids = batch['class_ids'].to(DEVICE, non_blocking=True)
                labels = batch['labels'].to(DEVICE, non_blocking=True)
                
                # 從 batch 中獲取更新後的屬性 bin ID
                pitch_coverage_bin_ids = batch['pitch_coverage_bin_ids'].to(DEVICE, non_blocking=True)
                note_per_pos_bin_ids = batch['note_per_pos_bin_ids'].to(DEVICE, non_blocking=True)
                pitch_class_entropy_bin_ids = batch['pitch_class_entropy_bin_ids'].to(DEVICE, non_blocking=True)

            except KeyError as ke:
                print(f"KeyError accessing batch data: {ke}. Available keys: {batch.keys()}", file=sys.stderr)
                print("This might be due to a mismatch between EtudeDataset __getitem__ and training loop.", file=sys.stderr)
                continue # Skip this batch
            except Exception as e_batch:
                print(f"Error processing batch: {e_batch}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                continue


            # 使用自動混合精度 (AMP)
            with torch.amp.autocast(device_type=DEVICE.split(':')[0], enabled=(DEVICE.startswith("cuda"))):
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    class_ids=class_ids,
                    pitch_coverage_bin_ids=pitch_coverage_bin_ids,
                    note_per_pos_bin_ids=note_per_pos_bin_ids,       # 更新
                    pitch_class_entropy_bin_ids=pitch_class_entropy_bin_ids, # 新增
                    labels=labels, 
                    return_dict=True
                )
                loss = outputs.loss

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss (None, NaN, or Inf) at epoch {epoch}, batch {batch_idx}. Skipping step.", file=sys.stderr)
                if loss is not None: print(f"Loss value: {loss.item()}", file=sys.stderr)
                # 清除可能存在的壞梯度
                optimizer.zero_grad(set_to_none=True)
                continue 
            
            loss_value = loss.item()
            epoch_loss_sum += loss_value
            total_loss_for_log_period += loss_value
            
            loss_scaled = scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS)
            loss_scaled.backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer) # 在 clip_grad_norm_ 之前 unscale
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer) # optimizer.step()
                scaler.update()        # 更新 scaler
                scheduler.step()       # 更新學習率 (每個 optimizer step 更新一次)
                optimizer.zero_grad(set_to_none=True) # 清零梯度
                
                global_step += 1
                num_optimizer_steps_this_epoch +=1
                num_optimizer_steps_in_log_period +=1

                if global_step > 0 and global_step % LOGGING_STEPS == 0:
                    if num_optimizer_steps_in_log_period > 0:
                         avg_loss_for_log = total_loss_for_log_period / num_optimizer_steps_in_log_period
                         current_lr = optimizer.param_groups[0]['lr']
                         pbar.set_postfix({"Loss": f"{avg_loss_for_log:.4f}", 
                                          "LR": f"{current_lr:.3e}", 
                                          "Step": global_step})
                         # 記錄到日誌文件 (更頻繁的 step 級日誌)
                         step_log_data = {
                             "type": "step_log", "global_step": global_step, "epoch": epoch,
                             "batch_idx": batch_idx, "avg_loss_period": avg_loss_for_log,
                             "learning_rate": current_lr, "timestamp": datetime.now().isoformat()
                         }
                         write_log(log_file_path, step_log_data)
                    total_loss_for_log_period = 0.0
                    num_optimizer_steps_in_log_period = 0
                
                # 檢查點（基於 global_step）
                if global_step > 0 and global_step % (LOGGING_STEPS * 5) == 0: # 例如，每 5 次 log 就保存一次
                    print(f"\nSaving step-based checkpoint at Global Step {global_step}...")
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, 
                                    current_run_checkpoint_dir, BASE_CHECKPOINT_DIR, is_epoch_end=False)


        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss_sum / num_optimizer_steps_this_epoch if num_optimizer_steps_this_epoch > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time
        current_lr_end_epoch = optimizer.param_groups[0]['lr'] # scheduler.get_last_lr()[0]
        
        pbar.close() # 確保 epoch 的 pbar 在打印 epoch 總結前關閉
        print(f"End of Epoch {epoch+1}/{NUM_EPOCHS}. Avg Batch Loss: {avg_epoch_loss:.4f}. LR: {current_lr_end_epoch:.3e}. Duration: {epoch_duration:.2f}s")
        
        epoch_log_data = {
            "type": "epoch_log", "epoch": epoch, "global_step": global_step,
            "avg_epoch_loss": avg_epoch_loss, "learning_rate_end_epoch": current_lr_end_epoch,
            "epoch_duration_seconds": epoch_duration, "timestamp": datetime.now().isoformat()
        }
        write_log(log_file_path, epoch_log_data)

        save_checkpoint(model, optimizer, scheduler, epoch, global_step, current_run_checkpoint_dir, BASE_CHECKPOINT_DIR, is_epoch_end=True)

    # --- Training Finished ---
    print("\n--- Training Finished ---")
    total_duration_sec = time.time() - training_start_time
    print(f"Total training time: {total_duration_sec//3600:.0f}h {(total_duration_sec%3600)//60:.0f}m {total_duration_sec%60:.0f}s")
    print(f"Final checkpoints and logs are in: {current_run_checkpoint_dir.resolve()}")

    print("Saving final model state...")
    save_checkpoint(model, optimizer, scheduler, NUM_EPOCHS -1, global_step, current_run_checkpoint_dir, BASE_CHECKPOINT_DIR, is_epoch_end=True)


if __name__ == "__main__":
    train()