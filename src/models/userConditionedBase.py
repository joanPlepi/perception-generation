from typing import Dict, Optional, Union, Tuple, List, Any
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput
from transformers import BartPretrainedModel, BartModel
from constants import *
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.utils import ModelOutput

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import shift_tokens_right, BartAttention
from transformers.utils import ModelOutput

class UserConditionedBase(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight", r"persona_encoder", r"linear_persona", r"bart_attention", r"history_att", r"decoder_gate", r"lm_head_external.weight"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.model = BartModel(config)
        
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.dropout = config.dropout
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        
        
        self.persona_cls_id = None
        self.history_cls_id = None
        self._mode = None
        self.acc_steps = None

        # Initialize weights and apply final processing
        self.post_init()
    
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, value):
        raise NotImplementedError("Not Implemented")
        
    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        
        if hasattr(self, 'lm_head_external') and self.lm_head_external is not None:
            self.lm_head_external = self._get_resized_lm_head(self.lm_head_external, new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        
        comments_past_key_values = kwargs['comments_past_key_values'] if 'comments_past_key_values' in kwargs else None
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            
            "persona_inputs": kwargs['persona_inputs'],
            "persona_attention_mask": kwargs['persona_attention_mask'],
            "persona_lengths": kwargs['persona_lengths'],
            "persona_cls_ids": kwargs['persona_cls_ids'],
            # "history_cls_ids": kwargs['history_cls_ids'],
            
            "history_titles_inputs": kwargs['history_titles_inputs'],
            "history_titles_attention_mask": kwargs['history_titles_attention_mask'],

            "history_comments_inputs": kwargs['history_comments_inputs'],
            "history_comments_attention_mask": kwargs['history_comments_attention_mask'],
            "history_lengths": kwargs['history_lengths'],
            "comments_past_key_values": comments_past_key_values,

            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "persona", "labels", "persona_cls_ids", "history"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        
        return model_kwargs
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def expand_dim(self, input_tensor):
        expanded_return_idx = (
            torch.arange(input_tensor.shape[0]).view(-1, 1).repeat(1, self.config.num_beams).view(-1).to(input_tensor.device)
            )
        return input_tensor.index_select(
                        0, expanded_return_idx.to(input_tensor.device)
                    )
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
    
    def bart_encode(
        self,  
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder=None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            
        return encoder_outputs
    
    def bart_decode(
        self,
        decoder,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_outputs,
        attention_mask,
        decoder_head_mask,
        cross_attn_head_mask,
        past_key_values,
        decoder_inputs_embeds,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict
    ):
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, persona_last_hidden_state, persona_attention_mask):
        input_mask_expanded = persona_attention_mask.unsqueeze(-1).expand(persona_last_hidden_state.size()).float()
        return torch.sum(persona_last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
     
    def compute_replace_verdict_token(self, encoder_outputs, persona_outputs, persona_attention_mask, verdict_ids):
        persona_representation = self.mean_pooling(persona_outputs.last_hidden_state, persona_attention_mask)
        encoder_outputs.last_hidden_state[verdict_ids[0], verdict_ids[1], :] = persona_representation
        return encoder_outputs
    
    def group_representations(self, representations, lengths):
        """Computes the average of each persona representations for one datapoint in the batch.
        If for one element we have (p1,p2,p3), we group all those 3 in one representation. 

        Args:
            representations (B*(L1 + L2 ... LB)xH): Representation of an extra information of the model. Li is equal to number of persona sentences in each data in batch.
            lengths (Tensor): [L1, L2, ..., LB] where B is batch length. Number of sentences for each element in the batch. 

        Returns:
            Tensor: BxH
        """
        if lengths is None or torch.sum(lengths) < 0:
            return representations
        temp = []
        s = 0
        for l in lengths:
            temp.append(1/l * torch.sum(representations[s:(s+l)], dim=0))
            s += l
        return torch.stack(temp)
    
    def compute_sum_with_cls_token(self, encoder_last_hidden_state, last_hidden_state, attention_mask, cls_ids, lengths=None):
        persona_cls_mask = F.one_hot(cls_ids[1], num_classes = encoder_last_hidden_state.size()[1]).type(torch.FloatTensor).to(encoder_last_hidden_state.device)
        persona_representation = self.mean_pooling(last_hidden_state, attention_mask)
        persona_representation = self.group_representations(persona_representation, lengths)                
        
        masked_persona = torch.bmm(persona_cls_mask.unsqueeze(-1), persona_representation.unsqueeze(1))
        masked_persona = nn.functional.dropout(masked_persona, p=self.dropout, training=self.training)
        
        if not self.training and encoder_last_hidden_state.size()[0] / masked_persona.size()[0] == self.model.config.num_beams:
            masked_persona = self.expand_dim(masked_persona)
            
        return masked_persona + encoder_last_hidden_state
    
    def compute_sum_with_history_cls_token(self, encoder_last_hidden_state, cls_ids, history_representation):
        cls_mask = F.one_hot(cls_ids[1], num_classes = encoder_last_hidden_state.size()[1]).type(torch.FloatTensor).to(encoder_last_hidden_state.device)
        cls_mask = cls_mask.unsqueeze(-1)

        if not self.training and history_representation.size()[0] / cls_mask.size()[0] == self.model.config.num_beams:
            cls_mask = self.expand_dim(cls_mask)
        
        masked_persona = torch.bmm(cls_mask, history_representation)
        masked_persona = nn.functional.dropout(masked_persona, p=self.dropout, training=self.training)
        
        return masked_persona + encoder_last_hidden_state

    def mean_group_representations(self, representations, attention_mask, lengths):
        representations = self.mean_pooling(representations, attention_mask) # B*N x T x H -> B*N x H
        return self.group_representations(representations, lengths) # B*N x H -> BxH   
    
    def compute_history_attention(self, encoder_outputs, title_outputs, comment_outputs, history_lengths):
        representations = []
        s = 0
        for i, l in enumerate(history_lengths):
            query, key, value = encoder_outputs[i].unsqueeze(0), title_outputs[s:(s+l)], comment_outputs[s:(s+l)]
            representations.append(self.history_att(query, key, value)[0])
            
        return torch.stack(representations)
    
    def compute_bart_attention(self, encoder_outputs, external_outputs, history_lengths):
        if torch.sum(history_lengths) > 0:
            representations = []
            s = 0
            for i, l in enumerate(history_lengths):
                query, key = encoder_outputs[i].unsqueeze(0), external_outputs[s:(s+l)]
                representations.append(self.bart_attention(query, key)[0].squeeze())
                
            return torch.stack(representations)
        else:
            return self.bart_attention(encoder_outputs, external_outputs)
            
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
        
        raise NotImplementedError("Implement Forward Function")

    def external_attention(self, external_outputs, lengths, query, swapped=False):
        if not self.training and query.size(0) / lengths.size(0) == self.model.config.num_beams:
            indeces = [i for i in range(0, query.size(0), self.model.config.num_beams)]
            if swapped:
                external_representation_att = self.compute_bart_attention(external_outputs.last_hidden_state, query[indeces], lengths)
            else:
                external_representation_att = self.compute_bart_attention(query[indeces], external_outputs.last_hidden_state, lengths)
            external_representation_att = self.expand_dim(external_representation_att)
        else:
            if swapped:
                external_representation_att = self.compute_bart_attention(external_outputs.last_hidden_state, query, lengths)
            else:
                external_representation_att = self.compute_bart_attention(query, external_outputs.last_hidden_state, lengths)
        return external_representation_att # B x S x D

    def get_control(self, hidden_state):
        temp = self.decoder_gate.unsqueeze(0).expand(hidden_state.size(0), 1,hidden_state.size(2)) 
        control = self.sigmoid(torch.bmm(temp, hidden_state.transpose(1, 2)))
        control = control.view(control.size(0), -1, 1)
        return control

    def attention_over_history(self, attention_mask, history_titles_attention_mask, history_comments_attention_mask, history_lengths, encoder_outputs, title_outputs, comment_outputs):
        encoder_representation = self.mean_pooling(encoder_outputs.last_hidden_state, attention_mask)
        title_representations = self.mean_pooling(title_outputs.last_hidden_state, history_titles_attention_mask)
        comment_representations = self.mean_pooling(comment_outputs.last_hidden_state, history_comments_attention_mask)
        history_representation = self.compute_history_attention(encoder_representation, title_representations, comment_representations, history_lengths)
        if encoder_representation.size()[0] / history_representation.size()[0] == self.config.num_beams:
            history_representation = self.expand_dim(history_representation)
        return history_representation

    def fully_connected_block(self, residual, hidden_states):
        # Fully Connected
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states
