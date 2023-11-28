from typing import Dict, Optional, Union, Tuple, List, Any
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput
from transformers import T5PreTrainedModel, T5Config
from constants import *
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.utils import ModelOutput
from transformers.models.t5.modeling_t5 import T5Stack
import copy
import warnings

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from models.modeling_outputs_extended import *


class T5ForConditionalGenerationUserEncoderExtended(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
        r"extra_decoder",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = config.dropout_rate

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        #self.extra_decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # control gate
        self.decoder_gate = nn.Parameter(torch.randn(config.d_model))
        self.sigmoid = nn.Sigmoid()
        self.lm_head_external = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        self._init_weights(self.decoder_gate)
        #self.extra_decoder.load_state_dict(self.decoder.state_dict())
        # Model parallel
        self.verdict_id = None
        self.persona_cls_id = None
        self.history_cls_id = None
        self.acc_steps = None
        self.mode = None
        
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        
        if hasattr(self, 'lm_head_external') and self.lm_head_external is not None:
            self.lm_head_external = self._get_resized_lm_head(self.lm_head_external, new_num_tokens)
        return new_embeddings
    
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
    

    def mean_pooling(self, persona_last_hidden_state, persona_attention_mask):
        input_mask_expanded = persona_attention_mask.unsqueeze(-1).expand(persona_last_hidden_state.size()).float()
        return torch.sum(persona_last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
     
    def expand_dim(self, input_tensor):
        expanded_return_idx = (
            torch.arange(input_tensor.shape[0]).view(-1, 1).repeat(1, self.config.num_beams).view(-1).to(input_tensor.device)
            )
        return input_tensor.index_select(
                        0, expanded_return_idx.to(input_tensor.device)
                    )
        
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
        
        if not self.training and encoder_last_hidden_state.size()[0] / masked_persona.size()[0] == self.config.num_beams:
            masked_persona = self.expand_dim(masked_persona)
            
        return masked_persona + encoder_last_hidden_state
            
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ####
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
        comments_past_key_values: Optional[List[torch.FloatTensor]] = None,

    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            
        if self.mode[0] == 'encoder_persona':
            # Convert encoder inputs in embeddings if needed
            persona_outputs = self.encoder(
                input_ids=persona_inputs,
                attention_mask=persona_attention_mask,
                inputs_embeds=None,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
            )
            encoder_outputs.last_hidden_state = self.compute_sum_with_cls_token(encoder_outputs.last_hidden_state, 
                                                                                        persona_outputs.last_hidden_state,
                                                                                        persona_attention_mask, persona_cls_ids, persona_lengths)
        else:
            raise Exception("Wrong Value")

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        #control = self.get_control(sequence_output)
        #lm_logits = control * self.lm_head(sequence_output) + (1-control) * self.lm_head_external(extra_decoder_outputs[0])
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
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
            input_ids = input_ids[:, -1:]
            
        comments_past_key_values = kwargs['comments_past_key_values'] if 'comments_past_key_values' in kwargs else None

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            
            "persona_inputs": kwargs['persona_inputs'],
            "persona_attention_mask": kwargs['persona_attention_mask'],
            "persona_lengths": kwargs['persona_lengths'],
            "persona_cls_ids": kwargs['persona_cls_ids'],
            #"history_cls_ids": kwargs['history_cls_ids'],
            
            "history_titles_inputs": kwargs['history_titles_inputs'],
            "history_titles_attention_mask": kwargs['history_titles_attention_mask'],

            "history_comments_inputs": kwargs['history_comments_inputs'],
            "history_comments_attention_mask": kwargs['history_comments_attention_mask'],
            "history_lengths": kwargs['history_lengths'],
            "comments_past_key_values": comments_past_key_values,
        }
    
    def get_control(self, hidden_state):
        temp = self.decoder_gate.unsqueeze(0).expand(hidden_state.size(0), 1,hidden_state.size(2)) 
        control = self.sigmoid(torch.bmm(temp, hidden_state.transpose(1, 2)))
        control = control.view(control.size(0), -1, 1)
        return control

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            print("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past