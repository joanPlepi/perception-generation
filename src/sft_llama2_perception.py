from utils.utils import *
from utils.train_utils import *
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.train_utils import create_dataset
from peft import LoraConfig
# Load model directly
import torch
from argparse import ArgumentParser
from utils_prompts import get_persona_prompts
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

parser = ArgumentParser()
parser.add_argument("--model_name", dest="model_name", default= "meta-llama/Llama-2-7b-hf", type=str, help="flant5_user_extended,flan_t5,custom")
parser.add_argument("--path_to_data", dest="path_to_data", default='/home/plepi/perception-xxl/data/', type=str)
parser.add_argument("--text_to_use", dest="text_to_use", default= "title", type=str, help="Which text of situations to use, post or title")

if __name__ == "__main__":
    args = parser.parse_args()
    path_to_data = args.path_to_data
    text_to_use = args.text_to_use
    model_name = 'custom'
    dataset_size = -1
    model_name = args.model_name

    dataset, ds = create_dataset(dataset_size, text_to_use, 'flan_t5', path_to_data, persona_amount=20,
                                    persona_cond=5, historyPairs_cond=5, priming=False, sampled_text=None, user_ids=False) 
    
    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
                                            model_name, use_cache=False, 
                                            quantization_config=quant_config, 
                                            device_map="auto"
                                            )
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prompts_dataset = get_persona_prompts(dataset, ds)
    prompts_ds = DatasetDict({
        "train": Dataset.from_dict(mapping=prompts_dataset["train"], features=Features({'text': Value(dtype='string')})),
        "val": Dataset.from_dict(mapping=prompts_dataset["val"], features=Features({'text': Value(dtype='string')})),
        })
    

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    training_args = TrainingArguments(
        output_dir="../results/sft_llama2_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        save_total_limit=1,
        save_steps=0.1,
        logging_steps=100,
        learning_rate=1.4e-5,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=prompts_ds["train"],
        eval_dataset=prompts_ds["val"],
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

