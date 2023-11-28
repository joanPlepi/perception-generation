from utils.utils import *
from utils.train_utils import *
from constants import *
from functools import *
from tqdm import tqdm
import json
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import pipeline
import random
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--onlyComments", dest="onlyComments", action='store_true', help="Only Comments")
parser.add_argument("--base", dest="base", action='store_true', help="use base model")

if __name__ == '__main__':
    args = parser.parse_args()
    model_name = "google/flan-t5-xxl"
    if args.base:
        model_name = "google/flan-t5-base"

    print(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='/data/joan/hf_cache')
    tokenizer.truncation_side = 'left'
    device = torch.device('cuda:0')
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='/data/joan/hf_cache', torch_dtype=torch.float16)
    model.to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Size: {model_size/1000**2:.1f}M parameters")
    
    path_to_data = '/app/public/joan/' 
    text_to_use = 'title'
    model_name = 'custom'
    dataset_size = -1
    dataset, ds = create_dataset(dataset_size, text_to_use, 'flan_t5', path_to_data, 
                                    persona_cond=5, historyPairs_cond=5, priming=False, sampled_text=None, user_ids=False) 
    
    authorsToSituationVerdicts = ListDict()
    authorsToPersona = dict() 
    split = "train"
    for data in tqdm(ds[split]):
        verdict_id = dataset.idToVerdict[data['index']]
        author_id = dataset.idToAuthors[data['author_id']]
        assert author_id == dataset.verdictToAuthor[verdict_id]
        authorsToSituationVerdicts.append(author_id, (data['situation_text'], data['verdict_text']))
        authorsToPersona[(author_id, verdict_id)] = data['persona']
        
    # Get Few Shot examples
    fewShot_test_examples = {}
    for data in ds["test"]:
        verdict_id = dataset.idToVerdict[data['index']]
        author_id = dataset.idToAuthors[data['author_id']]
        assert author_id == dataset.verdictToAuthor[verdict_id]
        
        if author_id in authorsToSituationVerdicts:
            prompt = "Example answers: \n"
            examples = authorsToSituationVerdicts[author_id]
            k = 10 if args.onlyComments else 5
            sampled_examples = random.sample(examples, k if len(examples) > k else len(examples))
            for situation, verdict in sampled_examples:
                if args.onlyComments:
                    prompt += f"{verdict}\n"
                else:
                    prompt += f"Q: {situation}\nA: {verdict}\n"
            prompt += "Generate an answer to Q, given some example answers above:\n"
            prompt += f"Q: {data['situation_text']}\nA: "
            fewShot_test_examples[verdict_id] = prompt
            
    fewShot_results = {}
    text2text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

    for v, p in tqdm(fewShot_test_examples.items()):
        output = text2text_generator(p, 
                        max_length=512,
                        do_sample=True,
                        no_repeat_ngram_size=3,
                        top_p = 0.95,
                        top_k = 0)
        for o in output:
            fewShot_results[v] = o['generated_text']
    
    if args.onlyComments:
        json.dump(fewShot_results, open(f'../results/fewShot_{model_name}_onlyComments_results.json', 'w'))
    else:
        json.dump(fewShot_results, open(f'../results/fewShot_{model_name}_results.json', 'w'))

    
    