from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import Seq2SeqLMOutput
from constants import *
import torch
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import Seq2SeqLMOutput
from constants import *
import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import shift_tokens_right, BartEncoder
from models.userConditionedBase import UserConditionedBase
from transformers.modeling_outputs import BaseModelOutput


class UserConditionedEncoder(UserConditionedBase):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight", r"extra_encoder", r"persona_encoder", r"linear_persona", r"bart_attention", r"history_att", r"decoder_gate", r"lm_head_external.weight"]

    def __init__(self, config) -> None:
        super().__init__(config)
        #self.extra_encoder =  BartEncoder(config, self.model.shared)
        # self.activation_fn = ACT2FN[config.activation_function] 
        # self.fc1 = nn.Linear(self.model.config.d_model, self.model.config.decoder_ffn_dim)
        # self.fc2 = nn.Linear(self.model.config.decoder_ffn_dim, self.model.config.d_model)
        # self.final_layer_norm = nn.LayerNorm(self.model.config.d_model)
        # self.activation_dropout = self.model.config.activation_dropout
        # self.fc1.load_state_dict(self.model.decoder.layers[0].fc1.state_dict())
        # self.fc2.load_state_dict(self.model.decoder.layers[0].fc2.state_dict())
        # self.final_layer_norm.load_state_dict(self.model.decoder.layers[0].final_layer_norm.state_dict())
        #self.extra_encoder.load_state_dict(self.model.encoder.state_dict())
        
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, value):
        if len(value) == 1 and value[0] in {'encoder_history', 'encoder_persona', 'encoder_comments', 'encoder_titles', 'encoder_both', 'encoder_comment_bartAtt'}:
            self._mode = value
        else:
            raise ValueError("Wrong mode")
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        persona_inputs: torch.LongTensor = None,
        persona_attention_mask: Optional[torch.Tensor] = None,
        persona_lengths: Optional[torch.Tensor] = None,
        persona_cls_ids: Optional[torch.Tensor] = None,
        history_cls_ids: Optional[torch.Tensor] = None,
        
        history_titles_inputs: torch.LongTensor = None,
        history_titles_attention_mask: Optional[torch.Tensor] = None,
        
        history_comments_inputs: torch.LongTensor = None,
        history_comments_attention_mask: Optional[torch.Tensor] = None,
        history_lengths: Optional[torch.Tensor] = None,
        
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        comments_past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
           
        if self._mode is None:
            raise ValueError("Mode not set!")
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        
        encoder_outputs = self.bart_encode(input_ids, attention_mask, self.get_encoder(),head_mask, inputs_embeds, encoder_outputs, output_attentions, output_hidden_states, use_cache, return_dict)

        if len(persona_inputs.size()) == 3:
            persona_inputs = persona_inputs.view(-1, self.max_persona_length)
            persona_attention_mask = persona_attention_mask.view(-1, self.max_persona_length)
            
        
        if 'encoder_history' == self.mode[0]:
            title_outputs = self.bart_encode(history_titles_inputs, history_titles_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
            comment_outputs = self.bart_encode(history_comments_inputs, history_comments_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
            history_representation = self.attention_over_history(attention_mask, history_titles_attention_mask, history_comments_attention_mask, history_lengths, encoder_outputs, title_outputs, comment_outputs)
            encoder_outputs.last_hidden_state = self.compute_sum_with_history_cls_token(encoder_outputs.last_hidden_state, persona_cls_ids, history_representation)
        elif 'encoder_persona' == self.mode[0]:
            persona_outputs = self.bart_encode(persona_inputs, persona_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
            encoder_outputs.last_hidden_state = self.compute_sum_with_cls_token(encoder_outputs.last_hidden_state, 
                                                                                        persona_outputs.last_hidden_state,
                                                                                        persona_attention_mask, persona_cls_ids, persona_lengths)
        elif 'encoder_comments' == self.mode[0]:
            comment_outputs = self.bart_encode(history_comments_inputs, history_comments_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
            encoder_outputs.last_hidden_state = self.compute_sum_with_cls_token(encoder_outputs.last_hidden_state, 
                                                                                        comment_outputs.last_hidden_state,
                                                                                        history_comments_attention_mask, persona_cls_ids, history_lengths)
        elif 'encoder_titles' == self.mode[0]:
            title_outputs = self.bart_encode(history_titles_inputs, history_titles_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
            encoder_outputs.last_hidden_state = self.compute_sum_with_cls_token(encoder_outputs.last_hidden_state, 
                                                                                        title_outputs.last_hidden_state,
                                                                                        history_titles_attention_mask, persona_cls_ids, history_lengths)
        elif 'encoder_both' == self.mode[0]:
            title_outputs = self.bart_encode(history_titles_inputs, history_titles_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
            comment_outputs = self.bart_encode(history_comments_inputs, history_comments_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
            history_representation = self.attention_over_history(attention_mask, history_titles_attention_mask, history_comments_attention_mask, history_lengths, encoder_outputs, title_outputs, comment_outputs)
            encoder_outputs.last_hidden_state = self.compute_sum_with_history_cls_token(encoder_outputs.last_hidden_state, history_cls_ids, history_representation)
            
            persona_outputs = self.bart_encode(persona_inputs, persona_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
            encoder_outputs.last_hidden_state = self.compute_sum_with_cls_token(encoder_outputs.last_hidden_state, 
                                                                                        persona_outputs.last_hidden_state,
                                                                                        persona_attention_mask, persona_cls_ids, persona_lengths)
        elif 'encoder_comment_bartAtt' in self.mode[0]:
            comment_outputs = self.bart_encode(history_comments_inputs, history_comments_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
            # comment_attention = self.external_attention(comment_outputs, history_lengths, encoder_outputs.last_hidden_state)
            #     # @TODO: replace persona token with mean pool from comments representation.
            # comment_outputs = self.fully_connected_block(encoder_outputs.last_hidden_state, comment_attention)
            
            comment_outputs = BaseModelOutput(
                last_hidden_state=comment_outputs
            )
        elif 'encoder_persona_bartAtt' in self.mode[0]:
            persona_outputs = self.bart_encode(persona_inputs, persona_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
            comment_attention = self.external_attention(persona_outputs, persona_lengths, encoder_outputs)
        else:
            raise ValueError(self.mode[0])
     
        if 'encoder_comment_bartAtt' == self.mode[0]:
            outputs = self.bart_decode(decoder_input_ids, decoder_attention_mask, comment_outputs, history_comments_attention_mask, 
                                decoder_head_mask, cross_attn_head_mask, past_key_values, decoder_inputs_embeds, 
                                   use_cache, output_attentions, output_hidden_states, return_dict)
        else:
            outputs = self.bart_decode(self.get_decoder(), decoder_input_ids, decoder_attention_mask, encoder_outputs, attention_mask, 
                                decoder_head_mask, cross_attn_head_mask, past_key_values, decoder_inputs_embeds, 
                                   use_cache, output_attentions, output_hidden_states, return_dict)
        
        
        lm_logits = self.lm_head(outputs[0])    
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)  
              
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

  