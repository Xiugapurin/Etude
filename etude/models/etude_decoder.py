# etude/models/etude_decoder.py

"""
Defines the core EtudeDecoder model for the 'decode' stage.

This module contains the model configuration (EtudeDecoderConfig) and the main
PyTorch model class (EtudeDecoder), which is built upon the GPT-NeoX architecture
from the Hugging Face Transformers library.
"""

import sys
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPTNeoXConfig, GPTNeoXModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from tqdm import tqdm

from ..data.dataset import SRC_CLASS_ID, TGT_CLASS_ID


class EtudeDecoderConfig(PretrainedConfig):
    """
    Configuration class for the EtudeDecoder model.
    This class is designed to be perfectly compatible with the original training config.
    """
    model_type = "etude_decoder"

    def __init__(
        self,
        # Standard GPT-NeoX parameters
        vocab_size: int = 3000,
        pad_token_id: int = 0,
        hidden_size: int = 512, 
        num_hidden_layers: int = 8, 
        num_attention_heads: int = 8,
        intermediate_size: int = 2048, 
        max_position_embeddings: int = 1024,
        
        # Custom parameters for EtudeDecoder
        num_classes: int = 3, 
        pad_class_id: int = 0,
        attribute_pad_id: int = 0,
        context_num_past_xy_pairs: int = 4,

        num_attribute_bins: int = 3,
        attribute_emb_dim: int = 64,
        
        initializer_range: float = 0.02,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size, 
            pad_token_id=pad_token_id, 
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers, 
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size, 
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range, 
            **kwargs
        )
        
        # Store all parameters as instance attributes
        self.num_classes = num_classes
        self.pad_class_id = pad_class_id
        self.attribute_pad_id = attribute_pad_id
        self.context_num_past_xy_pairs = context_num_past_xy_pairs
        self.num_attribute_bins = num_attribute_bins
        self.attribute_emb_dim = attribute_emb_dim
        self.num_pitch_overlap_ratio_bins = num_attribute_bins
        self.pitch_overlap_ratio_emb_dim = attribute_emb_dim
        self.num_relative_polyphony_bins = num_attribute_bins
        self.relative_polyphony_emb_dim = attribute_emb_dim
        self.num_relative_rhythmic_intensity_bins = num_attribute_bins
        self.relative_rhythmic_intensity_emb_dim = attribute_emb_dim
        self.num_relative_note_sustain_bins = num_attribute_bins
        self.relative_note_sustain_emb_dim = attribute_emb_dim


class EtudeDecoder(PreTrainedModel):
    """
    The EtudeDecoder model, a custom Causal LM for music generation.

    This model combines token, class, and four distinct musical attribute embeddings,
    projects them into a unified hidden space, and feeds them into a GPT-NeoX
    transformer backbone to predict the next token in a musical sequence.
    """
    config_class = EtudeDecoderConfig

    def __init__(self, config: EtudeDecoderConfig):
        super().__init__(config)
        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.class_embeddings = nn.Embedding(config.num_classes, config.hidden_size, padding_idx=config.pad_class_id)

        self.pitch_overlap_embeddings = nn.Embedding(
            config.num_pitch_overlap_ratio_bins, config.pitch_overlap_ratio_emb_dim, padding_idx=config.attribute_pad_id
        )
        self.polyphony_embeddings = nn.Embedding(
            config.num_relative_polyphony_bins, config.relative_polyphony_emb_dim, padding_idx=config.attribute_pad_id
        )
        self.note_sustain_embeddings = nn.Embedding(
            config.num_relative_note_sustain_bins, config.relative_note_sustain_emb_dim, padding_idx=config.attribute_pad_id
        )
        self.rhythm_intensity_embeddings = nn.Embedding(
            config.num_relative_rhythmic_intensity_bins, config.relative_rhythmic_intensity_emb_dim, padding_idx=config.attribute_pad_id
        )
        
        total_attribute_dim = (
            config.pitch_overlap_ratio_emb_dim + config.relative_polyphony_emb_dim +
            config.relative_note_sustain_emb_dim + config.relative_rhythmic_intensity_emb_dim
        )
        self.attribute_projection = nn.Linear(total_attribute_dim, config.hidden_size)
        
        gpt_neox_config = GPTNeoXConfig(**config.to_dict())
        self.transformer = GPTNeoXModel(gpt_neox_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
             if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                 module.weight.data[module.padding_idx].zero_()
             if isinstance(module, nn.Linear) and module.bias is not None:
                 module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self): 
        return self.word_embeddings
    
    def set_input_embeddings(self, new_embeddings): 
        self.word_embeddings = new_embeddings

    def get_output_embeddings(self): 
        return self.lm_head
    
    def set_output_embeddings(self, new_lm_head): 
        self.lm_head = new_lm_head

    def forward(
        self,
        input_ids: torch.LongTensor,
        class_ids: torch.LongTensor,
        polyphony_bin_ids: torch.LongTensor,
        rhythm_intensity_bin_ids: torch.LongTensor,
        note_sustain_bin_ids: torch.LongTensor,
        pitch_overlap_bin_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Get individual embeddings for tokens, classes, and attributes.
            word_embeds = self.word_embeddings(input_ids)
            class_embeds = self.class_embeddings(class_ids)
            
            attr_embeds = torch.cat([
                self.pitch_overlap_embeddings(pitch_overlap_bin_ids),
                self.polyphony_embeddings(polyphony_bin_ids),
                self.note_sustain_embeddings(note_sustain_bin_ids),
                self.rhythm_intensity_embeddings(rhythm_intensity_bin_ids)
            ], dim=-1)
            
            projected_attrs = self.attribute_projection(attr_embeds)
            inputs_embeds = word_embeds + class_embeds + projected_attrs
        
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits, 
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states, 
            attentions=transformer_outputs.attentions
        )


    @torch.no_grad()
    def generate(
        self,
        vocab,
        all_x_bars: List[List[int]],
        target_attributes_per_bar: List[Dict[str, int]],
        max_output_tokens: int = 25600,
        max_bar_token_limit: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        context_overlap_ratio: float = 0.5,
    ) -> List:
        
        device = next(self.parameters()).device
        self.eval()

        try:
            bar_bos_id, bar_eos_id = vocab.get_bar_bos_id(), vocab.get_bar_eos_id()
            if bar_bos_id == -1 or bar_eos_id == -1: 
                raise ValueError("Bar tokens not in vocab.")
            num_past_xy_pairs_for_context = self.config.context_num_past_xy_pairs
        except Exception as e:
            print(f"[ERROR] accessing vocab/config: {e}", file=sys.stderr); 
            return []

        if not all_x_bars or len(all_x_bars) != len(target_attributes_per_bar):
            print("[ERROR] Condition bars mismatch with target attributes.", file=sys.stderr); 
            return []

        attr_key_map = {
            "polyphony_bin": "polyphony_bin_ids",
            "rhythm_intensity_bin": "rhythm_intensity_bin_ids",
            "sustain_bin": "note_sustain_bin_ids",
            "pitch_overlap_bin": "pitch_overlap_bin_ids"
        }
        user_attr_keys = sorted(target_attributes_per_bar[0].keys())

        generated_events_final, total_generated_target_tokens = [], 0
        history_bar_pairs: List[Tuple[List[int], List[int], Dict[str, int]]] = []
        
        empty_bar_ids = [bar_bos_id, bar_eos_id]
        neutral_attributes = {key: 1 for key in user_attr_keys} 
        
        pbar = tqdm(range(len(all_x_bars)), desc="Generating Bars", unit="bar")
        for i in pbar:
            current_xi_ids = all_x_bars[i]
            current_yi_attrs = target_attributes_per_bar[i]
            
            prompt_tokens, p_classes, p_attr_lists = [], [], defaultdict(list)

            history_to_use = history_bar_pairs[-num_past_xy_pairs_for_context:]
            padding_needed = num_past_xy_pairs_for_context - len(history_to_use)

            # A. Fill with empty history
            for _ in range(padding_needed):
                for class_id in [SRC_CLASS_ID, TGT_CLASS_ID]:
                    prompt_tokens.extend(empty_bar_ids)
                    p_classes.extend([class_id] * len(empty_bar_ids))
                    for key in user_attr_keys: 
                        p_attr_lists[key].extend([neutral_attributes[key]] * len(empty_bar_ids))

            # B. Add actual history
            for x_ids, y_ids, attrs in history_to_use:
                for item_ids, class_id in [(x_ids, SRC_CLASS_ID), (y_ids, TGT_CLASS_ID)]:
                    prompt_tokens.extend(item_ids) 
                    p_classes.extend([class_id] * len(item_ids))
                    for key in user_attr_keys: 
                        p_attr_lists[key].extend([attrs[key]] * len(item_ids))

            # C. Add current Xi (the condition bar)
            prompt_tokens.extend(current_xi_ids)
            p_classes.extend([SRC_CLASS_ID] * len(current_xi_ids))
            for key in user_attr_keys: 
                p_attr_lists[key].extend([current_yi_attrs[key]] * len(current_xi_ids))
            
            # Truncate context if it exceeds model's max position embeddings
            if len(prompt_tokens) > self.config.max_position_embeddings - max_bar_token_limit:
                keep_len = int(self.config.max_position_embeddings * context_overlap_ratio)
                prompt_tokens, p_classes = prompt_tokens[-keep_len:], p_classes[-keep_len:]
                for key in user_attr_keys: 
                    p_attr_lists[key] = p_attr_lists[key][-keep_len:]

            tokens_this_bar, kv_cache = [], None
            
            # Prepare initial input tensors for this bar's generation
            input_ids = torch.tensor([prompt_tokens + [bar_bos_id]], device=device)
            class_ids = torch.tensor([p_classes + [TGT_CLASS_ID]], device=device)
            attr_ids = {key: torch.tensor([p_attr_lists[key] + [current_yi_attrs[key]]], device=device) for key in user_attr_keys}
            attention_mask = torch.ones_like(input_ids)
            
            # Auto-regressive generation loop for the current bar
            for _ in range(max_bar_token_limit):
                if total_generated_target_tokens >= max_output_tokens: break

                forward_kwargs = {
                    "input_ids": input_ids, 
                    "attention_mask": attention_mask, 
                    "class_ids": class_ids, 
                    "past_key_values": kv_cache, 
                    "use_cache": True,
                }
                
                for user_key, model_key in attr_key_map.items():
                    if user_key in attr_ids:
                        forward_kwargs[model_key] = attr_ids[user_key]
                
                outputs = self(**forward_kwargs)

                next_logits = outputs.logits[:, -1, :]
                kv_cache = outputs.past_key_values
                
                # Sampling (temperature and top-p)
                if temperature > 0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    if 0 < top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cum_probs = torch.cumsum(sorted_probs, dim=-1)
                        indices_to_remove = cum_probs > top_p
                        indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
                        indices_to_remove[..., 0] = 0
                        probs[0, sorted_indices[0, indices_to_remove[0]]] = 0
                        probs = probs / probs.sum()
                    next_token_id = torch.multinomial(probs, 1).squeeze().item()
                else: 
                    next_token_id = torch.argmax(next_logits, dim=-1).item()
                
                tokens_this_bar.append(next_token_id)
                total_generated_target_tokens += 1
                if next_token_id == bar_eos_id: break

                # Prepare inputs for the next token
                input_ids = torch.tensor([[next_token_id]], device=device)
                class_ids = torch.tensor([[TGT_CLASS_ID]], device=device)
                attr_ids = {key: torch.tensor([[current_yi_attrs[key]]], device=device) for key in user_attr_keys}
                attention_mask = torch.cat([attention_mask, torch.ones((1,1), device=device)], dim=1)
            
            # Update history and final results
            history_bar_pairs.append((current_xi_ids, [bar_bos_id] + tokens_this_bar, current_yi_attrs))
            if len(history_bar_pairs) > num_past_xy_pairs_for_context: 
                history_bar_pairs.pop(0)

            generated_events_final.extend(vocab.decode_sequence_to_events([bar_bos_id] + tokens_this_bar))
            pbar.set_postfix({"Generated Tokens": total_generated_target_tokens})
            if total_generated_target_tokens >= max_output_tokens: break

        return generated_events_final