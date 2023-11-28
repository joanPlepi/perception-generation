from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import Seq2SeqLMOutput
from models.userConditionedBase import UserConditionedBase
from constants import *
import torch
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput
from models.modeling_outputs_extended import *
from transformers.activations import ACT2FN

from constants import *
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class UserConditionedEncoderDecoder(UserConditionedBase):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight", r"persona_encoder", r"linear_persona", r"bart_attention", r"history_att", r"decoder_gate", r"lm_head_external.weight"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.decoder_gate = nn.Parameter(torch.randn(self.model.config.d_model))
        self.sigmoid = nn.Sigmoid()
        self.lm_head_external = nn.Linear(self.model.config.d_model, self.model.shared.num_embeddings, bias=False)
        
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(self.model.config.d_model, self.model.config.decoder_ffn_dim)
        self.fc2 = nn.Linear(self.model.config.decoder_ffn_dim, self.model.config.d_model)
        self.final_layer_norm = nn.LayerNorm(self.model.config.d_model)
        self.activation_dropout = self.model.config.activation_dropout

        # Loading weights
        self._init_weights(self.lm_head_external)
        self._init_weights(self.decoder_gate)
        self.fc1.load_state_dict(self.model.decoder.layers[0].fc1.state_dict())
        self.fc2.load_state_dict(self.model.decoder.layers[0].fc2.state_dict())
        self.final_layer_norm.load_state_dict(self.model.decoder.layers[0].final_layer_norm.state_dict())
        
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, value):
        if len(value) == 2 and value[0] in {'encoder_history', 'encoder_persona', 'encoder_comments', 'encoder_titles', 'encoder_both', 'encoder_comment_bartAtt'} and \
                    value[1] in {'decoder_residual_control', 'decoder_persona', 'decoder_comments', 'decoder_only_attention', 'decoder_residual', 'decoder_comments_swap'}:
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

        # if len(persona_inputs.size()) == 3:
        #     persona_inputs = persona_inputs.view(-1, self.max_persona_length)
        #     persona_attention_mask = persona_attention_mask.view(-1, self.max_persona_length)
            
        persona_outputs = self.bart_encode(persona_inputs, persona_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
        encoder_outputs.last_hidden_state = self.compute_sum_with_cls_token(encoder_outputs.last_hidden_state, 
                                                                                    persona_outputs.last_hidden_state,
                                                                                        persona_attention_mask, persona_cls_ids, persona_lengths)   
        outputs = self.bart_decode(decoder_input_ids, decoder_attention_mask, encoder_outputs, attention_mask, 
                                decoder_head_mask, cross_attn_head_mask, past_key_values, decoder_inputs_embeds, 
                                   use_cache, output_attentions, output_hidden_states, return_dict)
        
         
        comment_outputs = self.bart_encode(history_comments_inputs, history_comments_attention_mask, self.get_encoder(), None, None, None, None, None, None, None)
        
        if self.mode[1] == 'decoder_comments':
            comment_outputs = self.external_attention(comment_outputs, history_lengths, encoder_outputs.last_hidden_state)
            comment_outputs = BaseModelOutput(
                last_hidden_state=comment_outputs
            )
            decoded_comments = self.bart_decode(decoder_input_ids, decoder_attention_mask, comment_outputs, attention_mask, 
                                    decoder_head_mask, cross_attn_head_mask, past_key_values, decoder_inputs_embeds, 
                                    use_cache, output_attentions, output_hidden_states, return_dict)
        elif self.mode[1] == 'decoder_only_attention' or self.mode[1] == 'decoder_control_noll':
            decoded_comments = self.external_attention(comment_outputs, history_lengths, outputs[0])
        elif self.mode[1] == 'decoder_residual' or self.mode[1] == 'decoder_residual_control':
            decoded_comments = self.external_attention(comment_outputs, history_lengths, outputs[0])
            decoded_comments = self.fully_connected_block(outputs[0], decoded_comments)
        elif  self.mode[1] == 'decoder_comments_swap':
            comment_outputs = self.external_attention(comment_outputs, history_lengths, outputs[0], swapped=True)

            if not self.training and comment_outputs.size(0) / history_comments_attention_mask.size(0) == self.model.config.num_beams:
                history_comments_attention_mask = self.expand_dim(history_comments_attention_mask)

            comment_outputs = BaseModelOutput(
                last_hidden_state=comment_outputs
            )
            decoded_comments = self.bart_decode(decoder_input_ids, decoder_attention_mask, comment_outputs, history_comments_attention_mask, 
                                    decoder_head_mask, cross_attn_head_mask, comments_past_key_values, decoder_inputs_embeds, 
                                    use_cache, output_attentions, output_hidden_states, return_dict)
        else:
            raise ValueError("Wrong value given for mode")
        
        
        if self.mode[1] == 'decoder_residual':
            lm_logits = self.lm_head(decoded_comments)
            lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        else:
            control = self.get_control(outputs[0])
            lm_logits = control * self.lm_head(outputs[0]) + (1-control) * self.lm_head_external(decoded_comments[0])
            lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        
        return Seq2SeqLMOutputExtended(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            comments_past_key_values = decoded_comments.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

