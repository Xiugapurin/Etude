# train.py

import argparse
import math
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from etude.config import load_config, EtudeConfig
from etude.data.dataset import EtudeDataset
from etude.models.etude_decoder import EtudeDecoder, EtudeDecoderConfig
from etude.data.vocab import Vocab
from etude.utils.training_utils import set_seed, save_checkpoint, load_checkpoint
from etude.utils.logger import logger


class Trainer:
    """Encapsulates the entire training process."""

    def __init__(self, config: EtudeConfig):
        self.config = config
        self.train_cfg = config.train
        self.decoder_cfg = config.decoder

        # Resolve device
        self.device = config.env.device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # --- Setup Environment and Paths ---
        set_seed(config.env.seed)
        run_id = self.train_cfg.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.output_dir = config.paths.train_output_dir
        self.run_dir = self.output_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.run_dir.resolve()}")

        # --- Load Data ---
        logger.step("Loading vocabulary and dataset")
        logger.substep("Loading vocabulary...")
        vocab = Vocab.load(config.paths.dataset_vocab)
        self.model_config = self._create_model_config(vocab)

        model_config_save_path = self.run_dir / "etude_decoder_config.json"
        with open(model_config_save_path, 'w') as f:
            json.dump(self.model_config.to_dict(), f, indent=2)

        logger.substep("Loading dataset...")
        dataset = EtudeDataset(
            dataset_dir=config.paths.tokenized_dir,
            vocab=vocab,
            max_seq_len=self.model_config.max_position_embeddings,
            data_format=self.train_cfg.data_format,
            num_attribute_bins=self.model_config.num_attribute_bins,
            context_num_past_xy_pairs=self.model_config.context_num_past_xy_pairs
        )
        self.dataloader = dataset.get_dataloader(
            self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers
        )
        logger.info(f"Loaded {len(dataset)} training samples.")

        # --- Initialize Model, Optimizer, Scheduler ---
        logger.step("Initializing model and optimizer")
        logger.substep("Building model...")
        self.model = EtudeDecoder(self.model_config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate,
            betas=(self.train_cfg.adam_beta1, self.train_cfg.adam_beta2),
            weight_decay=self.train_cfg.weight_decay
        )
        self.scheduler = self._create_scheduler()
        logger.info("Model and optimizer initialized.")

        # --- Resume from Checkpoint if specified ---
        self.start_epoch, self.global_step = 0, 0
        resume_checkpoint = self.train_cfg.resume_from_checkpoint
        if resume_checkpoint:
            logger.step("Resuming from checkpoint")
            resume_dir = self.output_dir / resume_checkpoint
            self.start_epoch, self.global_step = load_checkpoint(
                resume_dir, self.model, self.optimizer, self.scheduler, self.device
            )

    def _create_model_config(self, vocab: Vocab) -> EtudeDecoderConfig:
        """Creates the model configuration from the config."""
        return EtudeDecoderConfig(
            vocab_size=len(vocab),
            pad_token_id=vocab.get_pad_id(),
            hidden_size=self.decoder_cfg.hidden_size,
            num_hidden_layers=self.decoder_cfg.num_hidden_layers,
            num_attention_heads=self.decoder_cfg.num_attention_heads,
            intermediate_size=self.decoder_cfg.intermediate_size,
            max_position_embeddings=self.decoder_cfg.max_position_embeddings,
            num_classes=self.decoder_cfg.num_classes,
            num_attribute_bins=self.decoder_cfg.num_attribute_bins,
            attribute_emb_dim=self.decoder_cfg.attribute_emb_dim,
            pad_class_id=self.decoder_cfg.pad_class_id,
            attribute_pad_id=self.decoder_cfg.attribute_pad_id,
            context_num_past_xy_pairs=self.decoder_cfg.context_num_past_xy_pairs,
        )

    def _create_scheduler(self):
        """Creates the learning rate scheduler."""
        num_epochs = self.train_cfg.num_epochs
        num_update_steps_per_epoch = math.ceil(
            len(self.dataloader) / self.train_cfg.gradient_accumulation_steps
        )
        total_training_steps = num_update_steps_per_epoch * num_epochs
        warmup_steps = self.train_cfg.warmup_epochs * num_update_steps_per_epoch

        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )

    def train(self):
        """Runs the main training loop."""
        logger.step("Starting training")
        scaler = torch.amp.GradScaler(enabled=(self.device == "cuda"))
        self.model.train()

        num_epochs = self.train_cfg.num_epochs
        grad_accum_steps = self.train_cfg.gradient_accumulation_steps
        save_every_n = self.train_cfg.save_every_n_epochs
        clip_grad_norm = self.train_cfg.clip_grad_norm

        for epoch in range(self.start_epoch, num_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(pbar):
                if not batch:
                    continue

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

                if loss is None or torch.isnan(loss):
                    continue

                scaler.scale(loss / grad_accum_steps).backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "LR": f"{self.scheduler.get_last_lr()[0]:.3e}"
                })

            # --- End of Epoch ---
            is_save_epoch = ((epoch + 1) % save_every_n == 0) or ((epoch + 1) == num_epochs)

            save_checkpoint(
                self.run_dir, self.model, self.optimizer, self.scheduler,
                epoch, self.global_step, is_epoch_end=is_save_epoch
            )

        logger.success("Training finished.")


def main():
    parser = argparse.ArgumentParser(description="Train the EtudeDecoder model.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the configuration file."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
