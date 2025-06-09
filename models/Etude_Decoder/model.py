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

        num_pitch_coverage_bins: int = 5,
        pitch_coverage_emb_dim: int = 64,
        num_note_per_pos_bins: int = 5,
        note_per_pos_emb_dim: int = 64,
        num_pitch_class_entropy_bins: int = 5,
        pitch_class_entropy_emb_dim: int = 64,
        
        attribute_pad_id: int = 0,
        context_num_past_xy_pairs: int = 2,
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

        # 更新屬性定義
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

        self.pitch_coverage_embeddings = nn.Embedding(
            config.num_pitch_coverage_bins, config.pitch_coverage_emb_dim, padding_idx=config.attribute_pad_id)
        self.note_per_pos_embeddings = nn.Embedding(
            config.num_note_per_pos_bins, config.note_per_pos_emb_dim, padding_idx=config.attribute_pad_id)
        self.pitch_class_entropy_embeddings = nn.Embedding(
            config.num_pitch_class_entropy_bins, config.pitch_class_entropy_emb_dim, padding_idx=config.attribute_pad_id)
        
        total_attribute_concat_dim = (
            config.pitch_coverage_emb_dim +
            config.note_per_pos_emb_dim +
            config.pitch_class_entropy_emb_dim
        )
        self.attribute_projection_layer = nn.Linear(total_attribute_concat_dim, config.hidden_size)

        gpt_neox_config = GPTNeoXConfig(
            vocab_size=config.vocab_size, # 雖然 GPTNeoXModel 不直接用，但配置中通常會保留
            hidden_size=config.hidden_size, 
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads, 
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act, 
            rotary_pct=config.rotary_pct, 
            rotary_emb_base=config.rotary_emb_base,
            max_position_embeddings=config.max_position_embeddings,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps, 
            use_cache=config.use_cache, 
            bos_token_id=config.bos_token_id, # GPTNeoX 不需要這些，但保持配置一致性
            eos_token_id=config.eos_token_id, 
            tie_word_embeddings=config.tie_word_embeddings,
        )
        self.transformer = GPTNeoXModel(gpt_neox_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
             if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
             if module.bias is not None: module.bias.data.zero_()
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            if input_ids is None: raise ValueError("`input_ids` or `inputs_embeds` must be provided.")
            if class_ids is None: raise ValueError("`class_ids` (COND/TGT) must be provided.")
            if (pitch_coverage_bin_ids is None or
                note_per_pos_bin_ids is None or 
                pitch_class_entropy_bin_ids is None):
                raise ValueError("All three attribute bin IDs (pitch_coverage, note_per_pos, pitch_class_entropy) must be provided.")

            word_embeds = self.word_embeddings(input_ids)
            cls_embeds = self.class_embeddings(class_ids)

            pitch_coverage_embs = self.pitch_coverage_embeddings(pitch_coverage_bin_ids)
            note_per_pos_embs = self.note_per_pos_embeddings(note_per_pos_bin_ids)
            pitch_class_entropy_embs = self.pitch_class_entropy_embeddings(pitch_class_entropy_bin_ids)
            
            combined_attribute_embs = torch.cat(
                [pitch_coverage_embs, note_per_pos_embs, pitch_class_entropy_embs],
                dim=-1
            )
            projected_attribute_embs = self.attribute_projection_layer(combined_attribute_embs)
            inputs_embeds = word_embeds + cls_embeds + projected_attribute_embs
        
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            position_ids=None, 
            past_key_values=past_key_values,
            use_cache=use_cache, 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss() 
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)
    
    @torch.no_grad()
    def generate(
        self,
        vocab: Vocab,
        initial_condition_token_ids: List[int],
        target_attributes_per_bar: List[Dict[str, int]],
        max_output_tokens: int = 10000,
        max_bar_token_limit: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        # num_past_bars_for_context: int, # 現在從 self.config 讀取
        context_overlap_ratio: float = 0.5,
    ) -> List[Event]:
        
        device = next(self.parameters()).device
        self.eval()

        try:
            bar_bos_id = vocab.get_bar_bos_id()
            bar_eos_id = vocab.get_bar_eos_id()
            pad_id = vocab.get_pad_id()
            model_max_seq_len = self.config.max_position_embeddings
            # 從 config 獲取要回看的 bar pair 數量
            num_past_xy_pairs_for_context = self.config.context_num_past_xy_pairs 

            if bar_bos_id == -1 or bar_eos_id == -1:
                 raise ValueError("Special tokens (Bar_BOS, Bar_EOS) not in vocab for generation.")
        except AttributeError as ae:
            print(f"Error: Missing attribute in config (e.g., context_num_past_xy_pairs). Config: {self.config}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr); return []
        except Exception as e:
            print(f"Error accessing vocab/config: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr); return []

        all_x_bars_token_ids = _split_into_bars_for_generate(initial_condition_token_ids, bar_bos_id, bar_eos_id)
        if not all_x_bars_token_ids:
            print("Error: Initial condition IDs could not be split into bars.", file=sys.stderr)
            return []
        
        num_x_bars = len(all_x_bars_token_ids)
        if num_x_bars != len(target_attributes_per_bar):
            print(f"Error: Number of condition bars ({num_x_bars}) "
                  f"does not match number of target attribute sets ({len(target_attributes_per_bar)}).", file=sys.stderr)
            return []
        # print(f"Condition split into {num_x_bars} X bars. Will generate {num_x_bars} Y bars.")

        generated_events_final: List[Event] = []
        total_generated_target_tokens = 0
        
        # history_bar_pairs 儲存元組 (x_ids, y_ids, x_attrs_dict, y_attrs_dict)
        # 其中 attrs_dict 是 {"pitch_coverage_bin": val, ...}
        history_bar_pairs: List[Tuple[List[int], List[int], Dict[str, int], Dict[str, int]]] = []
        
        kv_cache = None 

        empty_bar_ids = [bar_bos_id, bar_eos_id]
        # 假設所有屬性都使用相同的 bin 數量 self.config.num_pitch_coverage_bins
        # 如果不同，需要從 config 中分別獲取每個屬性的 num_bins
        # 為了簡化，我們假設有一個通用的 num_attribute_bins (例如5)
        num_bins_for_neutral = getattr(self.config, 'num_pitch_coverage_bins', 5) # 示例
        neutral_attr_bin_id = num_bins_for_neutral // 2
        
        neutral_attributes = {
            "pitch_coverage_bin": neutral_attr_bin_id,
            "note_per_pos_bin": neutral_attr_bin_id,
            "pitch_class_entropy_bin": neutral_attr_bin_id
        }

        for i in tqdm(range(num_x_bars), desc="Generating Bars (Xi -> Yi)"):
            current_xi_token_ids = all_x_bars_token_ids[i]
            current_target_yi_attributes = target_attributes_per_bar[i] # 這是用於生成 Yi 的目標屬性
            
            # --- 1. 構建 prompt 前綴，包含 num_past_xy_pairs_for_context 個 XY對，不足則填充 ---
            prompt_prefix_token_ids: List[int] = []
            prompt_prefix_class_ids: List[int] = []
            prompt_prefix_pcov_ids: List[int] = []
            prompt_prefix_npp_ids: List[int] = []
            prompt_prefix_pce_ids: List[int] = []

            num_actual_history_pairs = len(history_bar_pairs)
            num_padding_pairs_needed = max(0, num_past_xy_pairs_for_context - num_actual_history_pairs)

            # A. 填充空歷史
            for _ in range(num_padding_pairs_needed):
                # Pad X_empty
                prompt_prefix_token_ids.extend(empty_bar_ids)
                prompt_prefix_class_ids.extend([COND_CLASS_ID] * len(empty_bar_ids))
                prompt_prefix_pcov_ids.extend([neutral_attributes["pitch_coverage_bin"]] * len(empty_bar_ids))
                prompt_prefix_npp_ids.extend([neutral_attributes["note_per_pos_bin"]] * len(empty_bar_ids))
                prompt_prefix_pce_ids.extend([neutral_attributes["pitch_class_entropy_bin"]] * len(empty_bar_ids))
                # Pad Y_empty
                prompt_prefix_token_ids.extend(empty_bar_ids)
                prompt_prefix_class_ids.extend([TGT_CLASS_ID] * len(empty_bar_ids))
                prompt_prefix_pcov_ids.extend([neutral_attributes["pitch_coverage_bin"]] * len(empty_bar_ids))
                prompt_prefix_npp_ids.extend([neutral_attributes["note_per_pos_bin"]] * len(empty_bar_ids))
                prompt_prefix_pce_ids.extend([neutral_attributes["pitch_class_entropy_bin"]] * len(empty_bar_ids))

            # B. 添加實際歷史 (從 history_bar_pairs 的尾部取)
            start_idx_for_actual_history = max(0, num_actual_history_pairs - (num_past_xy_pairs_for_context - num_padding_pairs_needed))
            for hist_idx in range(start_idx_for_actual_history, num_actual_history_pairs):
                hist_x_ids, hist_y_ids, hist_x_attrs, hist_y_attrs = history_bar_pairs[hist_idx]
                # Add X_hist
                prompt_prefix_token_ids.extend(hist_x_ids)
                prompt_prefix_class_ids.extend([COND_CLASS_ID] * len(hist_x_ids))
                prompt_prefix_pcov_ids.extend([hist_x_attrs["pitch_coverage_bin"]] * len(hist_x_ids))
                prompt_prefix_npp_ids.extend([hist_x_attrs["note_per_pos_bin"]] * len(hist_x_ids))
                prompt_prefix_pce_ids.extend([hist_x_attrs["pitch_class_entropy_bin"]] * len(hist_x_ids))
                # Add Y_hist
                prompt_prefix_token_ids.extend(hist_y_ids)
                prompt_prefix_class_ids.extend([TGT_CLASS_ID] * len(hist_y_ids))
                prompt_prefix_pcov_ids.extend([hist_y_attrs["pitch_coverage_bin"]] * len(hist_y_ids))
                prompt_prefix_npp_ids.extend([hist_y_attrs["note_per_pos_bin"]] * len(hist_y_ids))
                prompt_prefix_pce_ids.extend([hist_y_attrs["pitch_class_entropy_bin"]] * len(hist_y_ids))

            # C. 添加當前的 Xi
            # Xi 的屬性使用其對應的目標 Yi 的屬性 (current_target_yi_attributes)
            # 這是一個簡化，假設訓練時 Xi 的屬性也是這樣處理的
            prompt_prefix_token_ids.extend(current_xi_token_ids)
            prompt_prefix_class_ids.extend([COND_CLASS_ID] * len(current_xi_token_ids))
            prompt_prefix_pcov_ids.extend([current_target_yi_attributes["pitch_coverage_bin"]] * len(current_xi_token_ids))
            prompt_prefix_npp_ids.extend([current_target_yi_attributes["note_per_pos_bin"]] * len(current_xi_token_ids))
            prompt_prefix_pce_ids.extend([current_target_yi_attributes["pitch_class_entropy_bin"]] * len(current_xi_token_ids))
            
            # --- 2. 上下文截斷 (作用於 prompt_prefix_xxx_ids) ---
            max_safe_prefix_len = model_max_seq_len - (max_bar_token_limit + 2) 
            if max_safe_prefix_len <= 0: max_safe_prefix_len = model_max_seq_len // 2

            if len(prompt_prefix_token_ids) > max_safe_prefix_len:
                keep_len = min(int(model_max_seq_len * context_overlap_ratio), max_safe_prefix_len)
                if keep_len <=0 and len(prompt_prefix_token_ids) > 0: keep_len = 1
                
                if keep_len > 0 and len(prompt_prefix_token_ids) > keep_len:
                    prompt_prefix_token_ids = prompt_prefix_token_ids[-keep_len:]
                    prompt_prefix_class_ids = prompt_prefix_class_ids[-keep_len:]
                    prompt_prefix_pcov_ids = prompt_prefix_pcov_ids[-keep_len:]
                    prompt_prefix_npp_ids = prompt_prefix_npp_ids[-keep_len:]
                    prompt_prefix_pce_ids = prompt_prefix_pce_ids[-keep_len:]
                    kv_cache = None # 截斷後，KV 快取失效
            
            # --- 3. 準備 Yi 生成的完整輸入 (截斷後的 Context_Prefix + Yi_BOS) ---
            final_prompt_token_ids = prompt_prefix_token_ids + [bar_bos_id]
            final_prompt_class_ids = prompt_prefix_class_ids + [TGT_CLASS_ID]
            final_prompt_pcov_ids = prompt_prefix_pcov_ids + [current_target_yi_attributes["pitch_coverage_bin"]]
            final_prompt_npp_ids = prompt_prefix_npp_ids + [current_target_yi_attributes["note_per_pos_bin"]]
            final_prompt_pce_ids = prompt_prefix_pce_ids + [current_target_yi_attributes["pitch_class_entropy_bin"]]

            initial_prompt_ids_tensor = torch.tensor([final_prompt_token_ids], dtype=torch.long, device=device)
            initial_prompt_class_ids_tensor = torch.tensor([final_prompt_class_ids], dtype=torch.long, device=device)
            initial_prompt_pcov_ids_tensor = torch.tensor([final_prompt_pcov_ids], dtype=torch.long, device=device)
            initial_prompt_npp_ids_tensor = torch.tensor([final_prompt_npp_ids], dtype=torch.long, device=device)
            initial_prompt_pce_ids_tensor = torch.tensor([final_prompt_pce_ids], dtype=torch.long, device=device)
            initial_attention_mask_tensor = torch.ones_like(initial_prompt_ids_tensor)
            
            # --- 4. 內部迴圈：為 Yi 生成 token (與之前版本基本一致) ---
            generated_tokens_in_current_yi = []
            last_generated_token_id = bar_bos_id
            generated_len_this_bar = 0
            # kv_cache 可能從上一個bar的截斷邏輯中被設為None，或者保持原樣
            # 為了與您原始版本（每個Xi生成一個Yi，內部kv_cache獨立）的穩定性相似，
            # 並且因為我們現在的 prompt 每次都是新構建的（包含了更長的歷史或padding），
            # 所以對於 Yi 的第一個 token，kv_cache 應該為 None。
            current_kv_cache_for_inner_loop = None # 強制為每個 Yi 的生成重新計算 prompt 的 KV

            while True:
                if total_generated_target_tokens >= max_output_tokens: break
                if generated_len_this_bar >= max_bar_token_limit: break
                
                model_input_ids_step: torch.LongTensor
                # ... (聲明其他 model_xxx_ids_step)
                model_class_ids_step: torch.LongTensor
                model_pcov_ids_step: torch.LongTensor 
                model_npp_ids_step: torch.LongTensor
                model_pce_ids_step: torch.LongTensor
                model_attention_mask_step: torch.LongTensor
                kv_to_pass_to_model: Optional[Tuple[Tuple[torch.Tensor]]]


                if generated_len_this_bar == 0: # Yi 的第一個 token (BOS 之後的那個)
                    model_input_ids_step = initial_prompt_ids_tensor
                    model_class_ids_step = initial_prompt_class_ids_tensor
                    model_pcov_ids_step = initial_prompt_pcov_ids_tensor
                    model_npp_ids_step = initial_prompt_npp_ids_tensor
                    model_pce_ids_step = initial_prompt_pce_ids_tensor
                    model_attention_mask_step = initial_attention_mask_tensor
                    kv_to_pass_to_model = current_kv_cache_for_inner_loop # 應為 None
                else: # Yi 的後續 token
                    model_input_ids_step = torch.tensor([[last_generated_token_id]], dtype=torch.long, device=device)
                    model_class_ids_step = torch.tensor([[TGT_CLASS_ID]], dtype=torch.long, device=device)
                    # 後續 token 使用當前目標 Yi 的屬性
                    model_pcov_ids_step = torch.tensor([[current_target_yi_attributes["pitch_coverage_bin"]]], dtype=torch.long, device=device)
                    model_npp_ids_step = torch.tensor([[current_target_yi_attributes["note_per_pos_bin"]]], dtype=torch.long, device=device)
                    model_pce_ids_step = torch.tensor([[current_target_yi_attributes["pitch_class_entropy_bin"]]], dtype=torch.long, device=device)
                    
                    kv_len = current_kv_cache_for_inner_loop[0][0].shape[-2] 
                    model_attention_mask_step = torch.ones((1, kv_len + 1), dtype=torch.long, device=device)
                    kv_to_pass_to_model = current_kv_cache_for_inner_loop
                
                try:
                    outputs = self(
                        input_ids=model_input_ids_step, class_ids=model_class_ids_step,
                        attention_mask=model_attention_mask_step,
                        pitch_coverage_bin_ids=model_pcov_ids_step,
                        note_per_pos_bin_ids=model_npp_ids_step,
                        pitch_class_entropy_bin_ids=model_pce_ids_step,
                        past_key_values=kv_to_pass_to_model,
                        use_cache=True, return_dict=True )
                    next_token_logits = outputs.logits[:, -1, :]
                    current_kv_cache_for_inner_loop = outputs.past_key_values 
                except Exception as gen_e:
                    print(f"Error in model forward pass for Y_{i+1}, token {generated_len_this_bar}: {gen_e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr); break 

                # Sampling 
                if temperature > 0:
                    # ... (sampling logic)
                    if temperature != 1.0: next_token_logits = next_token_logits / temperature
                    if top_p is not None and 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        if torch.any(sorted_indices_to_remove):
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            next_token_logits[indices_to_remove] = -float("Inf")
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze().item()
                else: # Greedy
                    next_token_id = torch.argmax(next_token_logits, dim=-1).squeeze().item()

                generated_tokens_in_current_yi.append(next_token_id)
                last_generated_token_id = next_token_id
                generated_len_this_bar += 1
                total_generated_target_tokens += 1
                if next_token_id == bar_eos_id: break
            # --- 結束內部迴圈 (生成一個 Yi) ---

            if not generated_tokens_in_current_yi:
                print(f"Warning: No tokens generated for target bar Y_{i+1}.")
                current_yi_full_ids_with_bos = [bar_bos_id, bar_eos_id] # 空 bar
            else:
                current_yi_full_ids_with_bos = [bar_bos_id] + generated_tokens_in_current_yi
            
            # --- 5. 更新 history_bar_pairs 列表 ---
            # Xi 的屬性也使用 current_target_yi_attributes (與 prompt 構建時一致)
            history_bar_pairs.append((
                list(current_xi_token_ids),             # X_i tokens
                list(current_yi_full_ids_with_bos),    # Y_i tokens (BOS...EOS)
                dict(current_target_yi_attributes),    # Attributes for X_i (approximated)
                dict(current_target_yi_attributes)     # Attributes for Y_i
            ))
            # 保持 history_bar_pairs 的長度，只保留最近的 num_past_xy_pairs_for_context 個
            if len(history_bar_pairs) > num_past_xy_pairs_for_context:
                history_bar_pairs = history_bar_pairs[-num_past_xy_pairs_for_context:]

            # kv_cache = current_kv_cache_for_inner_loop # 保存這個 bar 生成後的 KV 狀態
            # 這一行是多餘的，因為我們在外部迴圈開始時會重置 kv_cache=None 以匹配原始的逐bar生成邏輯

            # 解碼當前生成的 Yi 用於最終輸出
            # ... (與您提供的 EtudeModel-v2.py 中類似的解碼邏輯) ...
            generated_events_final.append(Event(type_="Bar", value="BOS"))
            for tid_idx, tid in enumerate(generated_tokens_in_current_yi): 
                if tid == pad_id: continue
                try:
                    token_str = vocab.decode(tid)
                    parts = token_str.split('_', 1)
                    type_, value_str = parts[0], parts[1] if len(parts) > 1 else ''
                    value_parsed = int(value_str) if type_ in ["Note", "Pos"] else value_str 
                    generated_events_final.append(Event(type_=type_, value=value_parsed))
                except ValueError: generated_events_final.append(Event(type_=type_, value=value_str))
                except Exception as e_dec_token: print(f"Error decoding token {tid} ('{vocab.decode(tid)}'): {e_dec_token}", file=sys.stderr)
            
            if not generated_events_final or generated_events_final[-1].type_ != "Bar" or generated_events_final[-1].value != "EOS":
                if generated_tokens_in_current_yi and generated_tokens_in_current_yi[-1] != bar_eos_id:
                    generated_events_final.append(Event(type_="Bar", value="EOS"))
            elif not generated_tokens_in_current_yi : # 如果 Y_i 為空，確保 Event 列表有 EOS
                 generated_events_final.append(Event(type_="Bar", value="EOS"))
            
            if total_generated_target_tokens >= max_output_tokens:
                print(f"\nGlobal max_output_tokens ({max_output_tokens}) reached.")
                break 
        # --- 結束外部 for 迴圈 (遍歷所有 Xi) ---

        print(f"\nGeneration finished. Total target tokens generated: {total_generated_target_tokens}")
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