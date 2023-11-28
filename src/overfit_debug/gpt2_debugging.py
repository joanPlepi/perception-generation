from utils.utils import *
from utils.train_utils import *
from constants import *
from functools import *

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling
import evaluate
import os
import math

def preprocess_causalLM(examples, tokenizer, max_input_length):
    assert len(examples["situation_text"]) == len(examples["verdict_text"])

    # {tokenizer.additional_special_tokens[1]} {tokenizer.additional_special_tokens[0]}
    outputs = tokenizer(
        [f"{examples['situation_text'][i]}{tokenizer.additional_special_tokens[0]}{examples['verdict_text'][i]}{tokenizer.eos_token}" for i in range(len(examples["verdict_text"]))],
        truncation=True,
        padding=True,
        max_length=max_input_length,
        #return_length=True
    )
    
    input_batch = []
    attention_batch = []
    #labels_batch = []
    for input_ids, att_mask  in zip(outputs["input_ids"], outputs["attention_mask"]):
        input_batch.append(input_ids)
     #   labels_batch.append(input_ids)
        attention_batch.append(att_mask)
            
    return {"input_ids": input_batch, "attention_mask": attention_batch, "labels": input_batch}#, "labels": labels_batch}

if __name__ == '__main__':
    path_to_data = '/app/public/joan/' 
    text_to_use = 'title'
    model_name = 'custom'
    dataset_size = 1000
    dataset, ds = create_dataset(dataset_size, text_to_use, 'bart', path_to_data, 
                                    persona_cond=5, historyPairs_cond=5, priming=False, sampled_text=None, user_ids=False) 
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = 'left'
    tokenizer.add_tokens(['yta', 'nta', 'YTA', 'NTA', 'AITA', 'aita'])
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|verdict|>', '<|post|>', '<|persona|>', '<|history_cls|>']})
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    
    tokenize = partial(preprocess_causalLM, tokenizer=tokenizer, max_input_length=model.config.n_ctx)
    tokenized_dataset = ds.map(
        tokenize, batched=True, remove_columns=ds["train"].column_names
    )
    tokenized_dataset.set_format("torch")
    
    def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)
            
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        
        return metric.compute(predictions=preds, references=labels)
    
    training_args = TrainingArguments(
        output_dir="gpt-test",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=100,
        push_to_hub=False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        report_to='wandb',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.save_model()
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    metrics = trainer.evaluate()
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
        
    metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
