# train.py

import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from pathlib import Path
import json
import random
import numpy as np
from tqdm import tqdm
import math
import sys
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

CONTEXT_NUM_PAST_XY_PAIRS = 4
NUM_ATTRIBUTE_BINS = 3

MODEL_PARAMS = {
    "hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8,
    "intermediate_size": 2048, "hidden_act": "gelu", "max_position_embeddings": 1024,
    "dropout_prob": 0.1, "num_classes": 3, "pad_class_id": PAD_CLASS_ID,
    "rotary_pct": 0.25, "rotary_emb_base": 10000, "initializer_range": 0.02,
    "layer_norm_eps": 1e-5, "use_cache": True, "bos_token_id": None,
    "eos_token_id": None, "tie_word_embeddings": False,

    # [MODIFIED] 所有屬性均使用 3 個 bins 和 32 維嵌入
    "num_avg_note_overlap_bins": NUM_ATTRIBUTE_BINS,
    "avg_note_overlap_emb_dim": 32,
    "num_pitch_coverage_bins": NUM_ATTRIBUTE_BINS,
    "pitch_coverage_emb_dim": 32,
    "num_note_per_pos_bins": NUM_ATTRIBUTE_BINS,
    "note_per_pos_emb_dim": 32,
    "num_pitch_class_entropy_bins": NUM_ATTRIBUTE_BINS,
    "pitch_class_entropy_emb_dim": 32,

    "attribute_pad_id": ATTRIBUTE_PAD_ID,
    "context_num_past_xy_pairs": CONTEXT_NUM_PAST_XY_PAIRS
}

# Runtime Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_SAVE_FORMAT = 'npy'
DATASET_MAX_SEQ_LEN = MODEL_PARAMS["max_position_embeddings"]

NUM_WORKERS_DATALOADER = 4 if sys.platform != 'win32' else 0
CHECKPOINT_SAVE_EPOCHS = 25
LOGGING_STEPS = 10000


def set_seed(seed_value):
    random.seed(seed_value); np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)


def write_log(log_file_path, log_data):
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data) + '\n')
    except Exception as e:
        print(f"Error writing to log file {log_file_path}: {e}")


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, run_checkpoint_dir, is_epoch_end=False):
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'epoch': epoch,'global_step': global_step,'model_config': model.config.to_dict()}
    latest_path = run_checkpoint_dir / "latest.pth"
    torch.save(payload, latest_path)
    if is_epoch_end:
        epoch_path = run_checkpoint_dir / f"epoch_{(epoch + 1):04d}.pth"
        shutil.copyfile(latest_path, epoch_path)
        print(f"Epoch-specific checkpoint saved to {epoch_path}")


def load_checkpoint(model, optimizer, scheduler, specific_run_dir, device):
    start_epoch, global_step = 0, 0
    checkpoint_to_load = specific_run_dir / "latest.pth"
    if checkpoint_to_load.exists():
        payload = torch.load(checkpoint_to_load, map_location=device)
        model.load_state_dict({k.replace('module.',''):v for k,v in payload.get('model_state_dict', payload).items()}, strict=False)
        if optimizer and 'optimizer_state_dict' in payload: optimizer.load_state_dict(payload['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in payload: scheduler.load_state_dict(payload['scheduler_state_dict'])
        global_step = payload.get('global_step', 0)
        start_epoch = payload.get('epoch', -1) + 1
        print(f"Checkpoint loaded. Resuming from Epoch {start_epoch}")
    return start_epoch, global_step

# --- Main Training Function ---
def train(run_id: Optional[str] = None):
    set_seed(SEED)
    run_id_to_use = run_id if run_id else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    current_run_checkpoint_dir = BASE_CHECKPOINT_DIR / run_id_to_use
    current_run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = current_run_checkpoint_dir / "train_log.jsonl"
    print(f"Current Run ID: {run_id_to_use}\nCheckpoints & logs: {current_run_checkpoint_dir.resolve()}")

    vocab = Vocab.load(VOCAB_PATH)
    MODEL_PARAMS['vocab_size'] = len(vocab)
    MODEL_PARAMS['pad_token_id'] = vocab.get_pad_id()

    model_config_for_run = EtudeDecoderConfig(**MODEL_PARAMS)
    
    dataset = EtudeDataset(
        dataset_dir=PREPROCESSED_DIR, 
        vocab=vocab, 
        max_seq_len=model_config_for_run.max_seq_len,
        data_format=DATA_SAVE_FORMAT,
        num_attribute_bins=NUM_ATTRIBUTE_BINS,
        context_num_past_xy_pairs=CONTEXT_NUM_PAST_XY_PAIRS,
        verbose_stats=True
    )
    dataloader = dataset.get_dataloader(BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DATALOADER)

    model = EtudeDecoder(model_config_for_run).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(ADAM_B1, ADAM_B2), weight_decay=WEIGHT_DECAY)
    
    num_update_steps_per_epoch = math.ceil(len(dataloader) / GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = num_update_steps_per_epoch * NUM_EPOCHS
    warmup_steps = WARMUP_EPOCHS * num_update_steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, current_run_checkpoint_dir, DEVICE)
    
    model.train()
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
    print("\n--- Starting Training ---")

    for epoch in range(start_epoch, NUM_EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if not batch: continue
            try:
                input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
                attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
                class_ids = batch['class_ids'].to(DEVICE, non_blocking=True)
                labels = batch['labels'].to(DEVICE, non_blocking=True)
                avg_note_overlap_bin_ids = batch['note_overlap_bin_ids'].to(DEVICE, non_blocking=True)
                pitch_coverage_bin_ids = batch['pitch_coverage_bin_ids'].to(DEVICE, non_blocking=True)
                note_per_pos_bin_ids = batch['note_per_pos_bin_ids'].to(DEVICE, non_blocking=True)
                pitch_class_entropy_bin_ids = batch['pitch_class_entropy_bin_ids'].to(DEVICE, non_blocking=True)
            except KeyError as ke: continue

            with torch.amp.autocast(device_type=DEVICE.split(':')[0], enabled=(DEVICE == "cuda")):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, class_ids=class_ids,
                    avg_note_overlap_bin_ids=avg_note_overlap_bin_ids,
                    pitch_coverage_bin_ids=pitch_coverage_bin_ids,
                    note_per_pos_bin_ids=note_per_pos_bin_ids,
                    pitch_class_entropy_bin_ids=pitch_class_entropy_bin_ids,
                    labels=labels, return_dict=True
                )
                loss = outputs.loss

            if loss is None or torch.isnan(loss): continue
            
            scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.3e}"})
        
        is_save_epoch = ((epoch + 1) % CHECKPOINT_SAVE_EPOCHS == 0) or ((epoch + 1) == NUM_EPOCHS)
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, current_run_checkpoint_dir, is_epoch_end=is_save_epoch)
        write_log(log_file_path, {"type": "epoch_log", "epoch": epoch, "loss": loss.item() if loss else None})


if __name__ == "__main__":
    train()