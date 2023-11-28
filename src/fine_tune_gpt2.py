import random
from argparse import ArgumentParser
from functools import *

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Features, Value
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel,
                          get_cosine_schedule_with_warmup, DataCollatorForLanguageModeling)

from constants import *
from dataset import SocialNormDataset
from utils.utils import *
import logging

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TIMESTAMP = get_current_timestamp()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"../logs/gpt2_{TIMESTAMP}.log"),
        logging.StreamHandler()
    ]
)


def get_grouped_params(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]
    
def evaluate_decoder(model, eval_dataloader):
    model.eval()
    losses = []
    for _, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)

        losses.append(outputs.loss)
        
    loss = torch.mean(torch.tensor(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

# Examples running the script
#python fine_tune_gpt2.py --text_to_use=title --num_epochs=5 --batch_size=16 --max_input_length=512 --model_name=dialogpt2 --desc='Model trained with added token for post, length 512'
parser = ArgumentParser()
parser.add_argument("--model_name", dest="model_name", required=True, type=str, help="Which model to use to use, gpt2 or dialogpt2")
parser.add_argument("--text_to_use", dest="text_to_use", required=True, type=str, help="Which text of situations to use, post or title")
parser.add_argument("--path_to_data", dest="path_to_data", default='/app/public/joan/', type=str)
parser.add_argument("--num_epochs", dest="num_epochs", default=10, type=int)
parser.add_argument("--learning_rate", dest="learning_rate", default=5e-4, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=64, type=int)
parser.add_argument("--max_input_length", dest="max_input_length", default=512, type=int)
parser.add_argument("--desc", dest="desc", default='', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    text_to_use = args.text_to_use
    assert text_to_use in {'post', 'title'}, print(text_to_use)
    model_name = args.model_name
    assert model_name in {'gpt2', 'dialogpt2'}, print(model_name)
    
    path_to_data = args.path_to_data
    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts_and_authors.gzip', compression='gzip')

    with open(path_to_data+'social_norms_clean.csv') as file:
        social_comments = pd.read_csv(file)
        
    dataset = SocialNormDataset(social_comments, social_chemistry, '/data/joan/users_perception/users_history_persona_v3_noAmit')
    
    verdict_ids = [
        random.choice(dataset.postToVerdicts[post]) for post in dataset.idTopostId if post in dataset.postToVerdicts]
    train_verdicts, test_verdicts = train_test_split(verdict_ids, test_size=0.2, 
                                                                        random_state=SEED)
    train_verdicts, val_verdicts = train_test_split(train_verdicts, test_size=0.15, 
                                                                        random_state=SEED)
    
    logging.info('Train length {}, val length {}, test length {}'.format(len(train_verdicts), len(val_verdicts), len(test_verdicts)))
    raw_dataset = {
            'train': {'index': [], 'situation_text': [], 'verdict_text': [], 'author_id': []}, 
            'val': {'index': [], 'situation_text': [], 'verdict_text': [], 'author_id': []}, 
            'test': {'index': [], 'situation_text': [], 'verdict_text': [], 'author_id': []}
            }

    for i, verdict in enumerate(train_verdicts):
        raw_dataset['train']['index'].append(dataset.verdictToId[verdict])
        raw_dataset['train']['situation_text'].append(
            dataset.postIdToText[dataset.verdictToParent[verdict]] if text_to_use == 'post' else f'AITA for {dataset.postIdToTitle[dataset.verdictToParent[verdict]]}?'
            )
        raw_dataset['train']['verdict_text'].append(dataset.verdictToText[verdict])
        raw_dataset['train']['author_id'].append(dataset.authorsToId[dataset.verdictToAuthor[verdict]])

    for i, verdict in enumerate(val_verdicts):
        raw_dataset['val']['index'].append(dataset.verdictToId[verdict])
        raw_dataset['val']['situation_text'].append(
            dataset.postIdToText[dataset.verdictToParent[verdict]] if text_to_use == 'post' else f'AITA for {dataset.postIdToTitle[dataset.verdictToParent[verdict]]}?'
            )
        raw_dataset['val']['verdict_text'].append(dataset.verdictToText[verdict])
        raw_dataset['val']['author_id'].append(dataset.authorsToId[dataset.verdictToAuthor[verdict]])
        
    for i, verdict in enumerate(test_verdicts):    
        raw_dataset['test']['index'].append(dataset.verdictToId[verdict])
        raw_dataset['test']['situation_text'].append(
            dataset.postIdToText[dataset.verdictToParent[verdict]] if text_to_use == 'post' else f'AITA for {dataset.postIdToTitle[dataset.verdictToParent[verdict]]}?'
            )    
        raw_dataset['test']['verdict_text'].append(dataset.verdictToText[verdict])
        raw_dataset['test']['author_id'].append(dataset.authorsToId[dataset.verdictToAuthor[verdict]])
        
    ds = DatasetDict()

    for split, d in raw_dataset.items():
        ds[split] = Dataset.from_dict(mapping=d, features=Features({'verdict_text': Value(dtype='string'), 
                                                                        'situation_text': Value(dtype='string'), 'index': Value(dtype='int64'), 'author_id': Value(dtype='int64')}))
    
    max_input_length = args.max_input_length

    if model_name == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    elif model_name == 'dialogpt2':
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small").to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    
    tokenizer.add_tokens(['yta', 'nta', 'YTA', 'NTA', 'AITA', 'aita'])
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|verdict|>', '<|post|>']})
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    tokenize = partial(preprocess_causalLM, tokenizer=tokenizer, max_input_length=max_input_length)

    tokenized_dataset = ds.map(
        tokenize, batched=True, remove_columns=ds["train"].column_names
    )
    
    tokenized_dataset.set_format("torch")
    batch_size = args.batch_size

    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_dataset["val"], batch_size=batch_size, collate_fn=data_collator)
    
    model.resize_token_embeddings(len(tokenizer))
    model_size = sum(t.numel() for t in model.parameters())
    logging.info(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    
    weight_decay = 0.1
    learning_rate = args.learning_rate
    num_train_epochs = args.num_epochs
    
    optimizer = AdamW(get_grouped_params(model, weight_decay), lr=learning_rate)
    
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    completed_steps = 0
    best_loss = 100000000
    checkpoint = f"../results/checkpoints/{model_name}_{text_to_use}_time_{TIMESTAMP}.pt"
    writer = SummaryWriter(comment=f'{model_name}_{text_to_use}')
    writer.add_scalar('learning_rate', learning_rate)
    writer.add_scalar('batch_size', batch_size)
    writer.add_scalar('max_input_length', args.max_input_length)
    writer.add_text('checkpoint', checkpoint)
    writer.add_text('model_name', f'# {model_name}')
    writer.add_text('description', args.desc)
    
    for epoch in range(num_train_epochs):
        progress_bar = tqdm(len(train_dataloader))

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss.backward()

            #loss = loss / gradient_accumulation_steps
            if step % 10 == 0 or step == len(train_dataloader) - 1:
                progress_bar.set_description(f"Epoch {epoch + 1}/{num_train_epochs} | Iter {step} | loss {loss.item():.5f}")
                
            #if step % gradient_accumulation_steps == 0:
            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        
        eval_loss, perplexity = evaluate_decoder(model, eval_dataloader)
        writer.add_scalar('eval_loss', loss, epoch)
        writer.add_scalar('eval_pp', perplexity, epoch)

        print({"loss/eval": eval_loss, "perplexity": perplexity})
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': eval_loss,
            },
            checkpoint
        )
        

    writer.close()
