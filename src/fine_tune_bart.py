import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from functools import *

from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import DataCollatorForSeq2Seq, AutoTokenizer, get_scheduler, AutoModelForSeq2SeqLM
import evaluate

import pickle as pkl
from utils.utils import *
from utils.train_utils import *
from constants import *
from argparse import ArgumentParser
import logging
from tensorboardX import SummaryWriter


TIMESTAMP = get_current_timestamp()
MODEL_NAME = 'bart'

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"../logs/{TIMESTAMP}.log"),
        logging.StreamHandler()
    ]
)

# python fine_tune_bart.py --text_to_use='title' --batch_size=32 --max_input_length=512 --max_target_length=512 --num_epochs=6 --dataset_size=-1 --priming='false' --user_ids='false'
parser = ArgumentParser()
# parser.add_argument("--priming", dest="priming", default=False, type=str2bool, help="enable or not priming. Combine with bart or t5.")
# parser.add_argument("--user_ids", dest="user_ids", default=False, type=str2bool, help="enable or not user_ids. Combine with bart or t5.")
parser.add_argument("--priming", dest="priming", action='store_true', help="enable or not priming. Combine with bart or t5.")
parser.add_argument("--user_ids", dest="user_ids", action='store_true', help="enable or not user_ids. Combine with bart or t5.")
parser.add_argument("--primingText_path", dest="primingText_path", default='../data/primed_persona.pkl', type=str, help="Priming text path")
parser.add_argument("--dataset_size", dest="dataset_size", required=True, type=int, help="Size, if -1 we use full dataset.")
parser.add_argument("--text_to_use", dest="text_to_use", required=True, type=str, help="Which text of situations to use, post or title")
parser.add_argument("--path_to_data", dest="path_to_data", default='/app/public/joan/', type=str)
parser.add_argument("--num_epochs", dest="num_epochs", default=5, type=int)
parser.add_argument("--learning_rate", dest="learning_rate", default=5e-4, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=16, type=int)
parser.add_argument("--max_input_length", dest="max_input_length", default=512, type=int)
parser.add_argument("--max_target_length", dest="max_target_length", default=128, type=int)

if __name__ == '__main__':
    logging.info(DEVICE)
    args = parser.parse_args()
    text_to_use = args.text_to_use
    path_to_data = args.path_to_data
    dataset_size = args.dataset_size
    
    assert text_to_use in {'post', 'title'}, print(text_to_use)

    print_args(args, logging)
    logging.info(f'Timestamp for this run: {TIMESTAMP}')
    
    
    authorToSampledText = pkl.load(open(args.primingText_path, 'rb')) if args.priming else None
    dataset, ds = create_dataset(dataset_size, text_to_use, MODEL_NAME, path_to_data, 
                                 persona_cond=5, historyPairs_cond=5, priming=args.priming, sampled_text=authorToSampledText, user_ids=args.user_ids)  
     
    train_authors, val_authors, test_authors  = set(ds["train"]["author_id"]),  set(ds["val"]["author_id"]), set(ds["test"]["author_id"])
    logging.info("Number of authors in the full dataset {}".format(len(train_authors.union(val_authors).union(test_authors))))
    logging.info(f"Debugging: {ds['train'][0]['situation_text']}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    tokenizer.truncation_side = 'left'
    tokenizer.add_tokens(['yta', 'nta', 'YTA', 'NTA', 'AITA', 'aita'])
    special_tokens =  ['<|verdict|>', '<|post|>', '<|persona|>', '<|persona_cls|>', '<|history_cls|>']
    
    if args.user_ids:
        for author in dataset.authorVerdictToPersonas.keys():
            special_tokens.append(get_author_token(author))

    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    model.resize_token_embeddings(len(tokenizer))
    if args.user_ids:
        logging.info("Updating user id embeddings!")
        add_user_embeddings(dataset, model, tokenizer)
    
    root_foldername = f'../data/{MODEL_NAME}/priming_{args.priming}_userId_{args.user_ids}_initEnd_{TIMESTAMP}/'
    os.mkdir(root_foldername)
    preprocess_function = partial(preprocess_seq2seq, tokenizer=tokenizer, max_input_length=args.max_input_length, max_target_length=args.max_target_length)

    
    tokenized_datasets = ds.map(preprocess_function, batched=True, num_proc=NUM_WORKERS)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ds["train"].column_names
    )
    tokenized_datasets.set_format("torch")
    logging.info(tokenized_datasets)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # dataloaders
    batch_size = args.batch_size
    acc_steps = 128 // batch_size
 
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=NUM_WORKERS
        
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["val"], collate_fn=data_collator, batch_size=batch_size, num_workers=NUM_WORKERS
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], collate_fn=data_collator, batch_size=batch_size, num_workers=NUM_WORKERS
    )
    
    # hyperparameters
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_train_epochs = args.num_epochs
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    model_size = sum(t.numel() for t in model.parameters())
    logging.info(f"{MODEL_NAME} size: {model_size/1000**2:.1f}M parameters")
    metric = evaluate.load("sacrebleu")
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    #progress_bar = tqdm(range(num_training_steps))
    
    best_bleue_score = 0
    best_pp = 10000000
    best_loss = 10000000
    
    checkpoint = f"../results/checkpoints/{MODEL_NAME}_priming_{str(args.priming)}_userId_{args.user_ids}_{text_to_use}_{TIMESTAMP}.pt"
        
    writer = SummaryWriter(comment=f'{MODEL_NAME}_{text_to_use}')
    writer.add_text('model_size', f'{model_size/1000**2:.1f}M')
    writer.add_scalar('learning_rate', args.learning_rate)
    writer.add_scalar('batch_size', args.batch_size)
    writer.add_scalar('max_input_length', args.max_input_length)
    writer.add_scalar('max_target_length', args.max_target_length)
    writer.add_scalar('dataset_size', args.dataset_size)
    writer.add_text('priming', str(args.priming))
    writer.add_text('checkpoint', checkpoint)
    
    generate = False
    for epoch in range(num_train_epochs):
        progress_bar = tqdm(len(train_dataloader))
        model.zero_grad()
        model.train()
        for step, batch in enumerate(train_dataloader):
            verdictId = batch.pop('verdict_id')
            batch.pop('author')
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
            output = model(**batch)
            loss = output.loss
            loss.backward()
            
            if (step+1) % acc_steps == 0: 
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()
            
            if step % 100 == 0 or step == len(train_dataloader) - 1:
                progress_bar.set_description(f"Epoch {epoch + 1}/{num_train_epochs} | Iter {step} | loss {loss.item():.5f}")

        results = evaluate_seq2seq(model, eval_dataloader, metric, tokenizer, dataset, MODEL_NAME, args.max_target_length, generate=generate)
        
        if generate:
            logging.info(f"Epoch {epoch} | Validation BLEU score: {results['score']:.2f} | Validation Perplexity {results['perplexity']:.4f} | Validation Loss: {results['loss']:.4f}")
            writer.add_scalar('bleu_val', results['score'], epoch)
        else:
            logging.info(f"Epoch {epoch} | Validation Perplexity {results['perplexity']:.4f} | Validation Loss: {results['loss']:.4f}")
            
        writer.add_scalar('loss_val', results['loss'], epoch)
        writer.add_scalar('perplexity_val', results['perplexity'], epoch)
        
        if results['perplexity'] < best_pp:
            best_pp = results['perplexity']
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': results['loss'],
            'perplexity': results['perplexity']
            },
            checkpoint
            )
            
    writer.close()
    
    logging.info("Loading best model.")
    checkpoint_dict = torch.load(checkpoint)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    
    logging.info("Evaluating on test dataset.")
    results = evaluate_seq2seq(model, test_dataloader, metric, tokenizer, dataset, MODEL_NAME, args.max_target_length, generate=True)
    pkl.dump(results, open(f'{root_foldername}/test_results.pkl', 'wb'))
    
    train_sample = tokenized_datasets["train"].train_test_split(0.1, seed=SEED)["test"]
    sampled_train = DataLoader(
        train_sample,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size)
    results = evaluate_seq2seq(model, sampled_train, metric, tokenizer, dataset, MODEL_NAME, args.max_target_length, generate=True)
    pkl.dump(results, open(f'{root_foldername}/train_results.pkl', 'wb'))


        
    
        

            
        
