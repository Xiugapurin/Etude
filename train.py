# train.py

import argparse
import math
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from etude.data.dataset import EtudeDataset
from etude.models.etude_decoder import EtudeDecoder, EtudeDecoderConfig
from etude.decode.vocab import Vocab
from etude.utils.training_utils import set_seed, save_checkpoint, load_checkpoint

class Trainer:
    """Encapsulates the entire training process."""
    def __init__(self, config: Dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- Setup Environment and Paths ---
        set_seed(self.config['environment']['seed'])
        run_id = self.config['environment']['run_id'] or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.output_dir = Path(self.config['checkpoint']['output_dir'])
        self.run_dir = self.output_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SETUP] Run ID: {run_id}")
        print(f"[SETUP] Device: {self.device}")
        print(f"[SETUP] Checkpoints and logs will be saved to: {self.run_dir.resolve()}")

        # --- Load Data ---
        print("[SETUP] Loading vocabulary and dataset...")
        vocab = Vocab.load(self.config['data']['vocab_path'])
        self.model_config = self._create_model_config(vocab)

        model_config_save_path = self.run_dir / "etude_decoder_config.json"
        with open(model_config_save_path, 'w') as f:
            json.dump(self.model_config.to_dict(), f, indent=2)
        print(f"[SETUP] Final model configuration saved to: {model_config_save_path}")
        
        dataset = EtudeDataset(
            dataset_dir=self.config['data']['dataset_dir'],
            vocab=vocab,
            max_seq_len=self.model_config.max_position_embeddings,
            data_format=self.config['data']['data_format'],
            num_attribute_bins=self.model_config.num_attribute_bins,
            context_num_past_xy_pairs=self.model_config.context_num_past_xy_pairs,
        )
        self.dataloader = dataset.get_dataloader(
            self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        
        # --- Initialize Model, Optimizer, Scheduler ---
        print("[SETUP] Initializing model, optimizer, and scheduler...")
        self.model = EtudeDecoder(self.model_config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(self.config['training']['adam_beta1'], self.config['training']['adam_beta2']),
            weight_decay=self.config['training']['weight_decay']
        )
        self.scheduler = self._create_scheduler()
        
        # --- Resume from Checkpoint if specified ---
        self.start_epoch, self.global_step = 0, 0
        resume_run_id = self.config['checkpoint'].get('resume_from_checkpoint')
        if resume_run_id:
            resume_dir = self.output_dir / resume_run_id
            self.start_epoch, self.global_step = load_checkpoint(
                resume_dir, self.model, self.optimizer, self.scheduler, self.device
            )

    def _create_model_config(self, vocab: Vocab) -> EtudeDecoderConfig:
        """Creates the model configuration from the YAML config."""
        model_cfg_dict = self.config['model'].copy()
        model_cfg_dict['vocab_size'] = len(vocab)
        model_cfg_dict['pad_token_id'] = vocab.get_pad_id()
        
        return EtudeDecoderConfig(**model_cfg_dict)

    def _create_scheduler(self):
        """Creates the learning rate scheduler."""
        cfg = self.config['training']
        num_epochs = cfg['num_epochs']
        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / cfg['gradient_accumulation_steps'])
        total_training_steps = num_update_steps_per_epoch * num_epochs
        warmup_steps = cfg['warmup_epochs'] * num_update_steps_per_epoch
        
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )

    def train(self):
        """Runs the main training loop."""
        print("\n[START] Beginning training loop...")
        scaler = torch.amp.GradScaler(enabled=(self.device == "cuda"))
        self.model.train()
        
        for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}", unit="batch")
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(pbar):
                if not batch: continue
                
                # Move batch to device
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                with torch.amp.autocast(device_type=self.device.split(':')[0], enabled=(self.device == "cuda")):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        class_ids=batch['class_ids'],
                        labels=batch['labels'],
                        polyphony_bin_ids=batch['polyphony_bin_ids'],
                        rhythm_intensity_bin_ids=batch['rhythm_intensity_bin_ids'],
                        note_sustain_bin_ids=batch['sustain_bin_ids'],
                        pitch_overlap_bin_ids=batch['pitch_overlap_bin_ids'],
                        return_dict=True
                    )
                    loss = outputs.loss

                if loss is None or torch.isnan(loss): continue
                
                scaler.scale(loss / self.config['training']['gradient_accumulation_steps']).backward()
                
                if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['clip_grad_norm'])
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{self.scheduler.get_last_lr()[0]:.3e}"})

            # --- End of Epoch ---
            is_save_epoch = ((epoch + 1) % self.config['checkpoint']['save_every_n_epochs'] == 0) or \
                            ((epoch + 1) == self.config['training']['num_epochs'])
            
            save_checkpoint(
                self.run_dir, self.model, self.optimizer, self.scheduler,
                epoch, self.global_step, is_epoch_end=is_save_epoch
            )

        print("\n[SUCCESS] Training finished.")

def main():
    parser = argparse.ArgumentParser(description="Train the EtudeDecoder model.")
    parser.add_argument(
        "--config", type=str, default="configs/training_config.yaml",
        help="Path to the training configuration YAML file."
    )
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()