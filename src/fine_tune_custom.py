import random
from tqdm import tqdm
from functools import *

from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, get_scheduler, T5Tokenizer
import evaluate

import pickle as pkl
from utils.utils import *
from utils.train_utils import *
from constants import *
from model import BartModelUserConditionedGeneration
from models.userConditionedEncoder import UserConditionedEncoder
from models.userConditionedDecoder import UserConditionedDecoder
from models.userConditionedEncoderDecoder import UserConditionedEncoderDecoder
from models.userT5Model import T5ForConditionalGenerationUserExtended
from models.userT5EncoderModel import T5ForConditionalGenerationUserEncoderExtended
from argparse import ArgumentParser
import logging
from tensorboardX import SummaryWriter


TIMESTAMP = get_current_timestamp()
MODEL_NAME = None
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

# python fine_tune_custom.py --text_to_use='title' --batch_size=8 --max_input_length=100 --max_target_length=256 --num_epochs=6 --dataset_size=-1 --encoder_mode='encoder_persona' --split='authors' --model_name='custom_flant5' --persona_separate='True' --only_persona
parser = ArgumentParser()
parser.add_argument("--model_name", dest="model_name", required=True, type=str, help="flant5_user_extended,flan_t5,custom")
parser.add_argument("--persona_separate", dest="persona_separate", default=True, type=str2bool, help="Enable or not separate")
parser.add_argument("--history_separate", dest="history_separate", default=False, type=str2bool, help="Enable or not separate")
parser.add_argument("--random_persona", dest="random_persona", action='store_true', help="Enable or not random persona")
parser.add_argument("--only_persona", dest="only_persona", action='store_true', help="Enable or not only persona")
parser.add_argument("--encoder_mode", dest="encoder_mode", required=False, default="", type=str, help="encoder_history, encoder_persona")
parser.add_argument("--decoder_mode", dest="decoder_mode", required=False, type=str, default="", help="decoder_history, decoder_persona, decoder_comments")
parser.add_argument("--decoder_infuse", dest="decoder_infuse", required=False, type=str, default="", help="sum, concat, project")
parser.add_argument("--split", dest="split", required=True, type=str, help="authors,verdicts,situation")
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
    split = args.split
    MODEL_NAME = args.model_name
    
    assert split in {'authors', 'verdicts', 'situation'}, print(split)
    assert text_to_use in {'post', 'title'}, print(text_to_use)

    print_args(args, logging)
    logging.info(f'Timestamp for this run: {TIMESTAMP}')

    dataset, ds = create_dataset(dataset_size, text_to_use, 'custom', path_to_data, random_persona=args.random_persona, persona_cond=5, historyPairs_cond=5, split=split)   
    train_authors, val_authors, test_authors  = set(ds["train"]["author_id"]),  set(ds["val"]["author_id"]), set(ds["test"]["author_id"])
    logging.info("Number of authors in the full dataset {}".format(len(train_authors.union(val_authors).union(test_authors))))
    
    if MODEL_NAME == 'custom' or MODEL_NAME == 'custom_flant5':
        if args.decoder_mode != "" and args.encoder_mode != "":
            model = UserConditionedEncoderDecoder.from_pretrained("facebook/bart-base")
            model.mode = [args.encoder_mode, args.decoder_mode]
            logging.info("Initialized encoder decoder mode")
        elif args.decoder_mode != "" and args.encoder_mode == "":
            model = UserConditionedDecoder.from_pretrained("facebook/bart-base")
            model.mode = [args.decoder_mode]
            logging.info("Initialized decoder mode")
        else:
            if MODEL_NAME == 'custom':
                model = UserConditionedEncoder.from_pretrained("facebook/bart-base")
                tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
                logging.info("Initialized encoder mode BART")
            else:
                model = T5ForConditionalGenerationUserEncoderExtended.from_pretrained("google/flan-t5-base").to(DEVICE)
                tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            
                logging.info("Initialized encoder mode FLAN T5")
            model.mode = [args.encoder_mode]

    else:
        logging.info("Flan t5")
        model = T5ForConditionalGenerationUserExtended.from_pretrained("google/flan-t5-base").to(DEVICE)
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        model.mode = [args.decoder_mode]
        
    tokenizer.truncation_side = 'left'
    tokenizer.add_tokens(['yta', 'nta', 'YTA', 'NTA', 'AITA', 'aita'])
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|verdict|>', '<|post|>', '<|persona|>', '<|persona_cls|>', '<|history_cls|>']})
    
    model.resize_token_embeddings(len(tokenizer))
    
    model.verdict_id = tokenizer.convert_tokens_to_ids([tokenizer.additional_special_tokens[0]])[0]
    model.persona_cls_id = tokenizer.convert_tokens_to_ids([tokenizer.additional_special_tokens[3]])[0]
    model.history_cls_id = tokenizer.convert_tokens_to_ids([tokenizer.additional_special_tokens[4]])[0]
    model.acc_steps = 32 // args.batch_size 
    
    if hasattr(model, 'extra_encoder'):
        model.extra_encoder.embed_tokens = model.get_input_embeddings()
    if hasattr(model, 'extra_decoder'):
        model.extra_decoder.embed_tokens = model.get_input_embeddings()
   
    
    mode_name = "_".join(model.mode)
    if args.random_persona:
        root_foldername = f'../data/{MODEL_NAME}/{mode_name}_personaSeparate_{str(args.persona_separate)}_randomExtraction_{split}_{text_to_use}_{TIMESTAMP}/'
    else: 
        root_foldername = f'../data/{MODEL_NAME}/{mode_name}_personaSeparate_{str(args.persona_separate)}_{split}_{text_to_use}_{TIMESTAMP}/'
    os.mkdir(root_foldername)
    

    preprocess_function = partial(preprocess_seq2seq_persona_separate, tokenizer=tokenizer, max_input_length=args.max_input_length, max_target_length=args.max_target_length, \
        insert_history_token=False, max_personaHistory_length=100, persona_separate=args.persona_separate, history_separate=args.history_separate)
       
    tokenized_datasets = ds.map(preprocess_function, batched=True, num_proc=NUM_WORKERS)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ds["train"].column_names
    )
    tokenized_datasets.set_format("torch")
    logging.info(tokenized_datasets)
    model.to(DEVICE)
    
    data_collator = DataCollatorForSeq2SeqExtended(tokenizer, model=model, history_separate=args.history_separate, persona_separate=args.persona_separate, only_persona=args.only_persona)
 
    # dataloaders
    batch_size = args.batch_size
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
    logging.info(f"{MODEL_NAME} with mode {mode_name} size: {model_size/1000**2:.1f}M parameters")
    logging.info(f"Accumulation steps {model.acc_steps}")
    metric = evaluate.load("sacrebleu")
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    best_bleue_score = 0
    best_pp = 10000000
    best_loss = 10000000
    
    if dataset_size <= 0:
        checkpoint = f"../results/checkpoints/{MODEL_NAME}_{mode_name}_{split}_fullData_{text_to_use}_{TIMESTAMP}.pt"
    else:
        checkpoint = f"../results/checkpoints/{MODEL_NAME}_{mode_name}_sampleSize_{dataset_size}_{text_to_use}_{TIMESTAMP}.pt"
        
        
    writer = SummaryWriter(comment=f'{MODEL_NAME}_{mode_name}_{split}_{text_to_use}')
    writer.add_scalar('Parameters/learning_rate', args.learning_rate)
    writer.add_scalar('Parameters/batch_size', args.batch_size)
    writer.add_scalar('Parameters/max_input_length', args.max_input_length)
    writer.add_scalar('Parameters/max_target_length', args.max_target_length)
    writer.add_scalar('Parameters/dataset_size', args.dataset_size)
    
    writer.add_text('Details/mode', mode_name)
    writer.add_text('Details/model_size', f'{model_size/1000**2:.1f}M')
    writer.add_text('Details/checkpoint', checkpoint)
    writer.add_text('Details/split', split)
    
    writer.add_text('Details/persona_separate', str(args.persona_separate))
    writer.add_text('Details/history_separate', str(args.history_separate))
    writer.add_text('Details/only_persona', str(args.only_persona))

    
    generate = False
    for epoch in range(num_train_epochs):
        progress_bar = tqdm(len(train_dataloader))
        model.zero_grad()
        model.train()
        for step, batch in enumerate(train_dataloader):
            verdictId = batch.pop('verdict_id')
            batch.pop('author')
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            if MODEL_NAME == 'custom' or MODEL_NAME == 'custom_flant5':
                batch['persona_cls_ids'] = (batch['input_ids'] == model.persona_cls_id).nonzero(as_tuple=True)
           
            output = model(**batch)
            loss = output.loss / model.acc_steps
            loss.backward()
            
            if (step+1) % model.acc_steps == 0: 
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()
            
            if step % 100 == 0 or step == len(train_dataloader) - 1:
                progress_bar.set_description(f"Epoch {epoch + 1}/{num_train_epochs} | Iter {step} | loss {output.loss.item():.5f}")

        results = evaluate_seq2seq(model, eval_dataloader, metric, tokenizer, dataset, MODEL_NAME, args.max_target_length, generate=generate)
        
        if generate:
            logging.info(f"Epoch {epoch} | Validation BLEU score: {results['score']:.2f} | Validation Perplexity {results['perplexity']:.4f} | Validation Loss: {results['loss']:.4f}")
            writer.add_scalar('Metrics/bleu_val', results['score'], epoch)
        else:
            logging.info(f"Epoch {epoch} | Validation Perplexity {results['perplexity']:.4f} | Validation Loss: {results['loss']:.4f}")
            
        writer.add_scalar('Metrics/loss_val', results['loss'], epoch)
        writer.add_scalar('Metrics/perplexity_val', results['perplexity'], epoch)
        
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
    
        

            
        