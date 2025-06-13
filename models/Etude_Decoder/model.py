# model.py

import sys, pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from transformers import GPTNeoXConfig, GPTNeoXModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List, Dict
from tqdm import tqdm
import traceback
from corpus.tokenizer import Vocab, Event, COND_CLASS_ID, TGT_CLASS_ID


def _split_into_bars_for_generate(id_sequence: List[int], bar_bos_id: int, bar_eos_id: int) -> List[List[int]]:
    """Helper function to split a token ID sequence into bars for the generate method."""
    bars = []; current_bar = []; in_bar = False
    if bar_bos_id < 0 or bar_eos_id < 0:
        print("Warning (_split_into_bars_for_generate): Invalid Bar BOS/EOS IDs, returning single bar.", file=sys.stderr)
        return [id_sequence] if id_sequence else [] # Fallback
        
    for token_id in id_sequence:
        if token_id == bar_bos_id:
            if in_bar and current_bar: # Save previous bar
                if current_bar[-1] != bar_eos_id: current_bar.append(bar_eos_id) # Ensure ended
                bars.append(current_bar)
            current_bar = [token_id]; in_bar = True
        elif token_id == bar_eos_id:
            if in_bar: 
                current_bar.append(token_id); bars.append(current_bar)
                current_bar = []; in_bar = False
        elif in_bar: 
            current_bar.append(token_id)
    if current_bar and in_bar: # Handle last bar if sequence didn't end with EOS token
        if current_bar[-1] != bar_eos_id: current_bar.append(bar_eos_id)
        if current_bar[0] == bar_bos_id : bars.append(current_bar)
            
    return [b for b in bars if len(b) > 1 and b[0] == bar_bos_id and b[-1] == bar_eos_id]


class EtudeDecoderConfig(PretrainedConfig):
    model_type = "etude_decoder"
    def __init__(
        self,
        vocab_size: int = 3000,
        pad_token_id: int = 0,
        num_classes: int = 3, 
        pad_class_id: int = 0,
        hidden_size: int = 512, 
        num_hidden_layers: int = 8, 
        num_attention_heads: int = 8,
        intermediate_size: int = 2048, 
        hidden_act: str = "gelu",
        rotary_pct: float = 0.25, 
        rotary_emb_base: int = 10000,
        max_position_embeddings: int = 1024, 
        max_seq_len: Optional[int] = None,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5, 
        use_cache: bool = True,
        bos_token_id: Optional[int] = None, 
        eos_token_id: Optional[int] = None,
        tie_word_embeddings: bool = False, 
        dropout_prob: float = 0.1,

        # [MODIFIED] Align all bin counts to 3 and embedding dimensions to 32
        num_avg_note_overlap_bins: int = 3,
        avg_note_overlap_emb_dim: int = 32,
        num_pitch_coverage_bins: int = 3,
        pitch_coverage_emb_dim: int = 32,
        num_note_per_pos_bins: int = 3,
        note_per_pos_emb_dim: int = 32,
        num_pitch_class_entropy_bins: int = 3,
        pitch_class_entropy_emb_dim: int = 32,
        
        attribute_pad_id: int = 0,
        # [MODIFIED] Align context window to 4 bars
        context_num_past_xy_pairs: int = 4,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size, pad_token_id=pad_token_id, hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size, hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings, initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps, use_cache=use_cache, bos_token_id=bos_token_id,
            eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs )
        self.num_classes, self.pad_class_id = num_classes, pad_class_id
        self.rotary_pct, self.rotary_emb_base = rotary_pct, rotary_emb_base
        self.dropout_prob = dropout_prob
        self.max_seq_len = max_seq_len if max_seq_len is not None else max_position_embeddings

        self.num_avg_note_overlap_bins, self.avg_note_overlap_emb_dim = \
            num_avg_note_overlap_bins, avg_note_overlap_emb_dim
        self.num_pitch_coverage_bins, self.pitch_coverage_emb_dim = \
            num_pitch_coverage_bins, pitch_coverage_emb_dim
        self.num_note_per_pos_bins, self.note_per_pos_emb_dim = \
            num_note_per_pos_bins, note_per_pos_emb_dim
        self.num_pitch_class_entropy_bins, self.pitch_class_entropy_emb_dim = \
            num_pitch_class_entropy_bins, pitch_class_entropy_emb_dim
        
        self.attribute_pad_id = attribute_pad_id
        self.context_num_past_xy_pairs = context_num_past_xy_pairs


class EtudeDecoder(PreTrainedModel):
    config_class = EtudeDecoderConfig
    def __init__(self, config: EtudeDecoderConfig):
        super().__init__(config)
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.class_embeddings = nn.Embedding(config.num_classes, config.hidden_size, padding_idx=config.pad_class_id)

        self.avg_note_overlap_embeddings = nn.Embedding(
            config.num_avg_note_overlap_bins, config.avg_note_overlap_emb_dim, padding_idx=config.attribute_pad_id)
        self.pitch_coverage_embeddings = nn.Embedding(
            config.num_pitch_coverage_bins, config.pitch_coverage_emb_dim, padding_idx=config.attribute_pad_id)
        self.note_per_pos_embeddings = nn.Embedding(
            config.num_note_per_pos_bins, config.note_per_pos_emb_dim, padding_idx=config.attribute_pad_id)
        self.pitch_class_entropy_embeddings = nn.Embedding(
            config.num_pitch_class_entropy_bins, config.pitch_class_entropy_emb_dim, padding_idx=config.attribute_pad_id)
        
        total_attribute_concat_dim = (
            config.avg_note_overlap_emb_dim + config.pitch_coverage_emb_dim +
            config.note_per_pos_emb_dim + config.pitch_class_entropy_emb_dim
        )
        self.attribute_projection_layer = nn.Linear(total_attribute_concat_dim, config.hidden_size)

        gpt_neox_config = GPTNeoXConfig(
            vocab_size=config.vocab_size, hidden_size=config.hidden_size, 
            num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads, 
            intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, 
            rotary_pct=config.rotary_pct, rotary_emb_base=config.rotary_emb_base,
            max_position_embeddings=config.max_position_embeddings,
            initializer_range=config.initializer_range, layer_norm_eps=config.layer_norm_eps, 
            use_cache=config.use_cache, bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id, tie_word_embeddings=config.tie_word_embeddings,
        )
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

    def get_input_embeddings(self): return self.word_embeddings
    def set_input_embeddings(self, new_embeddings): self.word_embeddings = new_embeddings
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_lm_head): self.lm_head = new_lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        class_ids: Optional[torch.LongTensor] = None,
        avg_note_overlap_bin_ids: Optional[torch.LongTensor] = None,
        pitch_coverage_bin_ids: Optional[torch.LongTensor] = None,
        note_per_pos_bin_ids: Optional[torch.LongTensor] = None,
        pitch_class_entropy_bin_ids: Optional[torch.LongTensor] = None, 
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            if input_ids is None or class_ids is None or avg_note_overlap_bin_ids is None or \
               pitch_coverage_bin_ids is None or note_per_pos_bin_ids is None or pitch_class_entropy_bin_ids is None:
                raise ValueError("`input_ids`, `class_ids`, and all four attribute bin IDs must be provided.")

            word_embeds = self.word_embeddings(input_ids)
            cls_embeds = self.class_embeddings(class_ids)
            
            attr_embs = torch.cat([
                self.avg_note_overlap_embeddings(avg_note_overlap_bin_ids),
                self.pitch_coverage_embeddings(pitch_coverage_bin_ids),
                self.note_per_pos_embeddings(note_per_pos_bin_ids),
                self.pitch_class_entropy_embeddings(pitch_class_entropy_bin_ids)
            ], dim=-1)
            
            inputs_embeds = word_embeds + cls_embeds + self.attribute_projection_layer(attr_embs)
        
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=past_key_values,
            use_cache=use_cache, output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            return ((loss,) + (logits,) + transformer_outputs[1:]) if loss is not None else ((logits,) + transformer_outputs[1:])
        
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=transformer_outputs.past_key_values,
                                      hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)
    

    @torch.no_grad()
    def generate(
        self,
        vocab: Vocab,
        initial_condition_token_ids: List[int],
        target_attributes_per_bar: List[Dict[str, int]],
        # [MODIFIED] Update default token limit
        max_output_tokens: int = 25600,
        max_bar_token_limit: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        context_overlap_ratio: float = 0.5,
    ) -> List[Event]:
        
        device = next(self.parameters()).device
        self.eval()

        try:
            bar_bos_id, bar_eos_id = vocab.get_bar_bos_id(), vocab.get_bar_eos_id()
            if bar_bos_id == -1 or bar_eos_id == -1: raise ValueError("Bar tokens not in vocab.")
            num_past_xy_pairs_for_context = self.config.context_num_past_xy_pairs
        except Exception as e:
            print(f"Error accessing vocab/config: {e}", file=sys.stderr); return []

        all_x_bars = _split_into_bars_for_generate(initial_condition_token_ids, bar_bos_id, bar_eos_id)
        if not all_x_bars or len(all_x_bars) != len(target_attributes_per_bar):
            print("Error: Condition bars mismatch with target attributes.", file=sys.stderr); return []

        generated_events_final, total_generated_target_tokens = [], 0
        history_bar_pairs: List[Tuple[List[int], List[int], Dict[str, int], Dict[str, int]]] = []
        
        empty_bar_ids = [bar_bos_id, bar_eos_id]
        # [MODIFIED] Neutral attribute is always middle bin 1, since all attributes have 3 bins.
        neutral_attributes = {
            "avg_note_overlap_bin": 1, "pitch_coverage_bin": 1,
            "note_per_pos_bin": 1, "pitch_class_entropy_bin": 1
        }
        
        # [MODIFIED] Add tqdm wrapper here
        pbar = tqdm(range(len(all_x_bars)), desc="Generating Bars", unit="bar")
        for i in pbar:
            current_xi_ids = all_x_bars[i]
            current_yi_attrs = target_attributes_per_bar[i]
            
            prompt_tokens, p_classes, p_ano, p_pcov, p_npp, p_pce = [], [], [], [], [], []
            
            # Simplified history building
            history_to_use = history_bar_pairs[-num_past_xy_pairs_for_context:]
            padding_needed = num_past_xy_pairs_for_context - len(history_to_use)

            for _ in range(padding_needed):
                for _ in range(2): # X_empty, Y_empty
                    prompt_tokens.extend(empty_bar_ids); p_classes.extend([COND_CLASS_ID]*len(empty_bar_ids))
                    p_ano.extend([neutral_attributes["avg_note_overlap_bin"]]*len(empty_bar_ids))
                    p_pcov.extend([neutral_attributes["pitch_coverage_bin"]]*len(empty_bar_ids))
                    p_npp.extend([neutral_attributes["note_per_pos_bin"]]*len(empty_bar_ids))
                    p_pce.extend([neutral_attributes["pitch_class_entropy_bin"]]*len(empty_bar_ids))

            for x_ids, y_ids, x_attrs, y_attrs in history_to_use:
                prompt_tokens.extend(x_ids); p_classes.extend([COND_CLASS_ID]*len(x_ids))
                p_ano.extend([x_attrs["avg_note_overlap_bin"]]*len(x_ids)); p_pcov.extend([x_attrs["pitch_coverage_bin"]]*len(x_ids)); p_npp.extend([x_attrs["note_per_pos_bin"]]*len(x_ids)); p_pce.extend([x_attrs["pitch_class_entropy_bin"]]*len(x_ids))
                prompt_tokens.extend(y_ids); p_classes.extend([TGT_CLASS_ID]*len(y_ids))
                p_ano.extend([y_attrs["avg_note_overlap_bin"]]*len(y_ids)); p_pcov.extend([y_attrs["pitch_coverage_bin"]]*len(y_ids)); p_npp.extend([y_attrs["note_per_pos_bin"]]*len(y_ids)); p_pce.extend([y_attrs["pitch_class_entropy_bin"]]*len(y_ids))

            prompt_tokens.extend(current_xi_ids); p_classes.extend([COND_CLASS_ID]*len(current_xi_ids))
            p_ano.extend([current_yi_attrs["avg_note_overlap_bin"]]*len(current_xi_ids)); p_pcov.extend([current_yi_attrs["pitch_coverage_bin"]]*len(current_xi_ids)); p_npp.extend([current_yi_attrs["note_per_pos_bin"]]*len(current_xi_ids)); p_pce.extend([current_yi_attrs["pitch_class_entropy_bin"]]*len(current_xi_ids))

            # Context Truncation
            if len(prompt_tokens) > self.config.max_position_embeddings - max_bar_token_limit:
                keep_len = int(self.config.max_position_embeddings * context_overlap_ratio)
                prompt_tokens, p_classes, p_ano, p_pcov, p_npp, p_pce = (l[-keep_len:] for l in [prompt_tokens, p_classes, p_ano, p_pcov, p_npp, p_pce])

            # Prepare for generation loop
            tokens_this_bar, kv_cache = [], None
            
            # Initial prompt processing
            prompt_tensors = [torch.tensor([l + [bar_bos_id]], device=device) for l in [prompt_tokens, p_classes, p_ano, p_pcov, p_npp, p_pce]]
            attention_mask = torch.ones_like(prompt_tensors[0])

            while len(tokens_this_bar) < max_bar_token_limit and total_generated_target_tokens < max_output_tokens:
                if kv_cache: # Subsequent steps
                    last_token = torch.tensor([[tokens_this_bar[-1]]], device=device)
                    attn_mask_len = kv_cache[0][0].shape[-2] + 1
                    attention_mask = torch.ones((1, attn_mask_len), device=device)
                    step_inputs = [last_token, torch.tensor([[TGT_CLASS_ID]], device=device)] + \
                                  [torch.tensor([[current_yi_attrs[k]]], device=device) for k in sorted(current_yi_attrs)]
                else: # First step
                    step_inputs = prompt_tensors
                
                try:
                    outputs = self(
                        input_ids=step_inputs[0], class_ids=step_inputs[1], attention_mask=attention_mask,
                        avg_note_overlap_bin_ids=step_inputs[2], pitch_coverage_bin_ids=step_inputs[3],
                        note_per_pos_bin_ids=step_inputs[4], pitch_class_entropy_bin_ids=step_inputs[5],
                        past_key_values=kv_cache, use_cache=True, return_dict=True)
                    kv_cache = outputs.past_key_values
                    
                    next_logits = outputs.logits[:, -1, :]
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
                    else: next_token_id = torch.argmax(next_logits, dim=-1).item()
                    
                    tokens_this_bar.append(next_token_id)
                    total_generated_target_tokens += 1
                    if next_token_id == bar_eos_id: break
                except Exception as e:
                    print(f"Error during generation step: {e}", file=sys.stderr); break

            # Post-generation
            history_bar_pairs.append((current_xi_ids, [bar_bos_id] + tokens_this_bar, current_yi_attrs, current_yi_attrs))
            generated_events_final.extend(vocab.decode_sequence_to_events([bar_bos_id] + tokens_this_bar))
            pbar.set_postfix({"Generated Tokens": total_generated_target_tokens}) # Update progress bar
            if total_generated_target_tokens >= max_output_tokens: break

        print(f"\nGeneration finished. Total target tokens: {total_generated_target_tokens}")
        return generated_events_final


def load_model(config_path: str, checkpoint_path: Optional[str] = None, device: str = "cpu") -> EtudeDecoder:
    try:
        with open(config_path, 'r') as f: config_dict = json.load(f)
        config = EtudeDecoderConfig.from_dict(config_dict)
        print("Loaded configuration using from_dict.")
    except Exception as e_outer:
        print(f"Error loading configuration with from_dict: {e_outer}. Falling back to direct init.")
        try:
            with open(config_path, 'r') as f: config_dict_direct = json.load(f)
            config = EtudeDecoderConfig(**config_dict_direct)
            print("Loaded configuration using direct initialization.")
        except Exception as e_inner: print(f"Error loading configuration directly: {e_inner}"); raise

    model = EtudeDecoder(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    if checkpoint_path:
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model_state = state_dict.get('model_state_dict', state_dict.get('model', state_dict))
            load_result = model.load_state_dict(model_state, strict=True)
            print(f"Loaded checkpoint from: {checkpoint_path}, Result: {load_result}")
        except Exception as e: print(f"Error loading checkpoint: {e}. Random weights.")
    model.to(device)
    return model