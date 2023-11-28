import json
import os
from datasets import load_metric
from numpy import average
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from utils.clusters_utils import ListDict
from utils.loss_functions import CB_loss
from constants import DEVICE
import pickle as pkl
from utils.read_files import read_splits, write_splits
from utils.utils import get_author_token, get_verdicts_labels_from_sit, get_verdicts_labels_from_authors
from transformers import DataCollatorForSeq2Seq
from constants import SEED
import pandas as pd
from dataset import SocialNormDataset
from datasets import DatasetDict, Dataset, Features, Value, Sequence
import numpy as np
import random
from utils.utils import postprocess_text
random.seed(SEED)


class AuthorsEmbedder:
    def __init__(self, embeddings_path, dim):
        self.authors_embeddings = pkl.load(open(embeddings_path, 'rb'))
        self.dim = dim
    
    
    def embed_author(self, author):
        return torch.tensor(self.authors_embeddings.get(author, torch.rand(self.dim)))
    

class DataCollatorForSeq2SeqExtended():
    def __init__(self, tokenizer, model, history_separate=True, persona_separate=True, only_persona=False) -> None:
        self.data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        self.history_separate = history_separate
        self.persona_separate = persona_separate
        self.only_persona = only_persona

    def __call__(self, features, return_tensors=None):
        if self.persona_separate:
            persona_inputs = self.stack_separate_features(features, 'persona_inputs')
            persona_attention_mask = self.stack_separate_features(features, 'persona_attention_mask')
        
        if not self.only_persona and self.history_separate:
            history_titles_inputs = self.stack_separate_features(features, 'history_titles_inputs')
            history_titles_attention_mask = self.stack_separate_features(features, 'history_titles_attention_mask')

            history_comments_inputs = self.stack_separate_features(features, 'history_comments_inputs')
            history_comments_attention_mask = self.stack_separate_features(features, 'history_comments_attention_mask')
        
        features = self.data_collator(features, return_tensors)
        
        if self.persona_separate:
            features['persona_inputs'] = torch.concat(persona_inputs, dim=0)
            features['persona_attention_mask'] = torch.concat(persona_attention_mask, dim=0)
        
        if not self.only_persona and self.history_separate:
            features['history_titles_inputs'] = torch.concat(history_titles_inputs, dim=0)
            features['history_titles_attention_mask'] = torch.concat(history_titles_attention_mask, dim=0)
            
            features['history_comments_inputs'] = torch.concat(history_comments_inputs, dim=0)
            features['history_comments_attention_mask'] = torch.concat(history_comments_attention_mask, dim=0)
        
        return features

    def stack_separate_features(self, features, feature_name):
        return [torch.stack(feature.pop(feature_name)) for feature in features]    


def loss_fn(output, targets, samples_per_cls, no_of_classes=2, loss_type = "softmax"):
    beta = 0.9999
    gamma = 2.0

    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma)


def get_verdicts_by_situations_split(dataset):
    if not os.path.exists('../data/splits/train_sit.txt'):
        all_situations = set(dataset.postIdToId.keys())
        annotated_situations = json.load(open('../data/conflict_aspect_annotations.json', 'r'))
        annotated_situations = set(annotated_situations['data'].keys())
        all_situations = list(all_situations.difference(annotated_situations))

        train_situations, test_situations = train_test_split(all_situations, test_size=0.18, random_state=SEED)
        train_situations, val_situations = train_test_split(train_situations, test_size=0.15, random_state=SEED)
        test_situations.extend(list(annotated_situations))
        write_splits('../data/splits/train_sit.txt', train_situations)
        write_splits('../data/splits/test_sit.txt', test_situations)
        write_splits('../data/splits/val_sit.txt', val_situations)
    else:
        print("Loading situations splits.")
        train_situations = read_splits('../data/splits/train_sit.txt')
        val_situations = read_splits('../data/splits/val_sit.txt')
        test_situations = read_splits('../data/splits/test_sit.txt')
        
    postToVerdicts = ListDict()
    for v, s in dataset.verdictToParent.items():
        #if dataset.verdictToTokensLength[v] > 5:
        postToVerdicts.append(s, v)
        
    train_verdicts, train_labels = get_verdicts_labels_from_sit(dataset, train_situations, postToVerdicts)
    val_verdicts, val_labels = get_verdicts_labels_from_sit(dataset, val_situations, postToVerdicts)
    test_verdicts, test_labels = get_verdicts_labels_from_sit(dataset, test_situations, postToVerdicts)
    return train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels


def get_verdicts_by_author_split(dataset):
    if not os.path.exists('../data/splits/train_author.txt'):
            all_authors = list(dataset.authorsToVerdicts.keys())
            train_authors, test_authors = train_test_split(all_authors, test_size=0.2, random_state=SEED)
            train_authors, val_authors = train_test_split(train_authors, test_size=0.14, random_state=SEED)
            write_splits('../data/splits/train_author.txt', train_authors)
            write_splits('../data/splits/val_author.txt', val_authors)
            write_splits('../data/splits/test_author.txt', test_authors)
    else:
        print("Reading authors splits.")
        train_authors = read_splits('../data/splits/train_author.txt')
        val_authors = read_splits('../data/splits/val_author.txt')
        test_authors = read_splits('../data/splits/test_author.txt')
        # train_authors.remove('Judgement_Bot_AITA')
        
    train_verdicts, train_labels = get_verdicts_labels_from_authors(dataset, train_authors)
    val_verdicts, val_labels = get_verdicts_labels_from_authors(dataset, val_authors)
    test_verdicts, test_labels = get_verdicts_labels_from_authors(dataset, test_authors)
    return train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def create_author_graph(graphData, dataset, authors_embeddings, authorToAuthor, limit_connections=100):
    leave_out = {'Judgement_Bot_AITA'}
    for author, _ in dataset.authorsToVerdicts.items():
        if author not in leave_out:
            graphData.addNode(author, 'author', authors_embeddings[author], None, None)
            
    # Add author to author edges
    source = []
    target = []
    for author, neighbors in tqdm(authorToAuthor.items()):
        neighbors.sort(key=lambda x: x[1], reverse=True)
        if len(neighbors) > limit_connections:
            neighbors = neighbors[:limit_connections]
            
        for neighbor in neighbors:
            # neighbor[0] = author, neighbor[1] = number_of_connections
            if author in graphData.nodesToId and neighbor[0] in graphData.nodesToId:
                source.append(graphData.nodesToId[author])
                target.append(graphData.nodesToId[neighbor[0]])
            
    
    return graphData, torch.tensor([source, target], dtype=torch.long)

def add_user_embeddings(dataset, model, tokenizer):
    author_embeddings = pkl.load(open('/app/home/plepi/users-perception/data/embeddings/emnlp/roberta_sbert_authorAMIT.pkl', 'rb'))
        
    with torch.no_grad():
        for _, author in enumerate(dataset.authorVerdictToPersonas.keys()):
            token_id = torch.tensor(tokenizer.convert_tokens_to_ids([get_author_token(author)]), device=DEVICE)
            embedding = torch.tensor(author_embeddings[author], device=DEVICE).unsqueeze(0)
            model.shared.weight[token_id] = embedding
            assert torch.equal(embedding.cpu(), model.shared(token_id).cpu())

def create_dataset(dataset_size, text_to_use, model_name, path_to_data, persona_cond=5, historyPairs_cond=5, random_persona=False, priming=False, sampled_text=None, user_ids=False, split=None):
    assert model_name in {'custom', 'bart', 't5', 'flan_t5', 'flant5_user_extended'} 
    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts_and_authors.gzip', compression='gzip')

    with open(path_to_data+'social_norms_clean.csv') as file:
        social_comments = pd.read_csv(file)
        
    dataset = SocialNormDataset(social_comments, social_chemistry, '/data/joan/users_perception/users_history_persona_v3_noAmit', persona_cond=persona_cond, historyPairs_cond=historyPairs_cond)
    verdict_ids = dataset.get_filtered_verdicts() # full dataset if dataset_size < 0, or priming is activated 

    if dataset_size > 0: # means we are not working with full dataset
        random.seed(SEED)
        verdict_ids = random.sample(verdict_ids, dataset_size) 
    elif model_name in {'bart', 't5'} and not priming and not user_ids: # for the case where we use plain models, we need to decouple situations from verdicts.
        random.seed(SEED)
        verdict_ids = [random.choice(verdicts) for _, verdicts in dataset.postToVerdicts.items()]
            
    raw_dataset = {
            'index': [], 'situation_text': [], 'verdict_text': [], 'author_id': [], 'persona': [], 'history_titles': [], 'history_comments': []
            }

    for _, verdict in enumerate(verdict_ids):
        author = dataset.verdictToAuthor[verdict]
        if author in dataset.authorVerdictToPersonas:
            raw_dataset['index'].append(dataset.verdictToId[verdict])
            if priming:
                raw_dataset['situation_text'].append(f'I am the following persona: {sampled_text[author]} \nThe question is as follows: AITA for {dataset.postIdToTitle[dataset.verdictToParent[verdict]]}?')
            elif user_ids:
                author_token = get_author_token(author)
                raw_dataset['situation_text'].append(f'AITA for {dataset.postIdToTitle[dataset.verdictToParent[verdict]]}? {author_token}')
            else:   
                raw_dataset['situation_text'].append(
                        dataset.postIdToText[dataset.verdictToParent[verdict]] if text_to_use == 'post' else f'AITA for {dataset.postIdToTitle[dataset.verdictToParent[verdict]]}?'
                    )
            raw_dataset['verdict_text'].append(dataset.verdictToText[verdict])
            raw_dataset['author_id'].append(dataset.authorsToId[author])
            if random_persona:
                raw_dataset['persona'].append(list(dataset.authors_randomPersonas[author]))
            else:
                raw_dataset['persona'].append([text for text, sim in dataset.authorVerdictToPersonas[author][verdict]])
            #raw_dataset['persona'].append([text for text in dataset.authors_personas[author]])
            raw_dataset["history_titles"].append([title for title, _ in dataset.verdictAuthorsToPairs[verdict][author]])
            #raw_dataset["history_comments"].append([comment for _,  comment in dataset.verdictAuthorsToPairs[verdict][author]]) 
            
            if type(dataset.verdictsToNeighbors[verdict]) != list:
                neighbors = [dataset.verdictToText[dataset.verdictsToNeighbors[verdict]]]
            elif len(dataset.verdictsToNeighbors[verdict]) > 1:
                neighbors = [dataset.verdictToText[n] for n in dataset.verdictsToNeighbors[verdict]]
            elif len(dataset.verdictsToNeighbors[verdict]) == 0:
                neighbors = [""]
            else:
                raise ValueError("")
    
            raw_dataset["history_comments"].append(neighbors)
            
    ds = Dataset.from_dict(mapping=raw_dataset, features=Features({'verdict_text': Value(dtype='string'), 
                                                                        'situation_text': Value(dtype='string'), 
                                                                        'index': Value(dtype='int64'), 'author_id': Value(dtype='int64'),
                                                                        'persona': Sequence(Value(dtype='string')), 
                                                                        'history_titles': Sequence(Value(dtype='string')), 
                                                                        'history_comments': Sequence(Value(dtype='string'))}))
    
    if split is None or split == 'verdicts':
        train_testvalid = ds.train_test_split(test_size=0.2, seed=SEED)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=SEED)
        ds = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'val': test_valid['train']})
    elif split == 'authors':
        train_authors, test_authors = train_test_split(list(set(ds['author_id'])), test_size=0.2, random_state=SEED)
        test_authors, val_authors = train_test_split(test_authors, test_size=0.5, random_state=SEED)

        train_authors = set(train_authors)
        val_authors = set(val_authors)
        test_authors = set(test_authors)
        
        ds = DatasetDict({
            'train': ds.filter(lambda d: d['author_id'] in train_authors),
            'test': ds.filter(lambda d: d['author_id'] in test_authors),
            'val': ds.filter(lambda d: d['author_id'] in val_authors)})
    elif split == 'situation':
        all_situations = list({dataset.verdictToParent[dataset.idToVerdict[index]] for index in ds['index']})

        train_sit, test_sit = train_test_split(all_situations, test_size=0.2, random_state=SEED)
        test_sit, val_sit = train_test_split(test_sit, test_size=0.5, random_state=SEED)

        train_sit = set(train_sit)
        val_sit = set(val_sit)
        test_sit = set(test_sit)
            
        ds = DatasetDict({
            'train': ds.filter(lambda d: dataset.verdictToParent[dataset.idToVerdict[d['index']]] in train_sit),
            'test': ds.filter(lambda d: dataset.verdictToParent[dataset.idToVerdict[d['index']]] in test_sit),
            'val': ds.filter(lambda d: dataset.verdictToParent[dataset.idToVerdict[d['index']]] in val_sit)})
    else:
        raise ValueError("Wrong Split")
        
        
    return dataset,ds

def evaluate_seq2seq(model, eval_dataloader, metric, tokenizer, dataset, model_name, max_target_length, generate=True):
    # Evaluation
    temperature = 0.7
    top_p = 0.95
    predictions = []
    gold_labels = []
    verdictIds = []
    losses = []
    model.eval()
    
    for _, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch_verdictId = batch.pop('verdict_id')
            batch.pop('author')
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            if model_name == 'custom' or model_name == 'custom_flant5':
                batch['persona_cls_ids'] = (batch['input_ids'] == model.persona_cls_id).nonzero(as_tuple=True)
                #batch['history_cls_ids'] = (batch['input_ids'] == model.history_cls_id).nonzero(as_tuple=True) if model.mode[0] == 'encoder_both' else None


            output = model(**batch)
            losses.append(output.loss)
            labels = batch.pop('labels')
            batch.pop('decoder_input_ids')
            
            if generate:
                generated_tokens = model.generate(
                    **batch,
                     max_length=max_target_length,
                    top_p=float(top_p),
                    do_sample=False,
                    no_repeat_ngram_size=3,
                )

                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )

                predictions.extend(decoded_preds)
                gold_labels.extend(decoded_labels)
                verdictIds.extend([dataset.idToVerdict[id] for id in batch_verdictId])

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    loss = torch.mean(torch.tensor(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
        
    if generate:
        return {'bleu_score': metric.compute()['score'], 'loss': loss.item(), 'predictions': predictions, 
                'labels': gold_labels, 'verdictIds': verdictIds, 'perplexity': perplexity.item()}
    else:
        return  {'loss': loss.item(), 'perplexity': perplexity.item()}
