from typing import Optional, Tuple
from transformers.modeling_outputs import Seq2SeqLMOutput
from dataclasses import dataclass
import torch

@dataclass
class Seq2SeqLMOutputExtended(Seq2SeqLMOutput):
    comments_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

