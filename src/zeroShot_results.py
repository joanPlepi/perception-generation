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
from constants import CUDA

parser = ArgumentParser()
parser.add_argument("--model_name", dest="model_name", required=True, type=str, help="Which model to use to use, gpt2 or dialogpt2")
parser.add_argument("--text_to_use", dest="text_to_use", required=True, type=str, help="Which text of situations to use, post or title")
parser.add_argument("--path_to_data", dest="path_to_data", default='/app/public/joan/', type=str)
parser.add_argument("--results_dir", dest="results_dir", default='../results', type=str)


if __name__ == '__main__':
    model_name = args.model_name
    path_to_data = args.path_to_data
    text_to_use = args.text_to_use
    results_dir = args.results_dir
    
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='/data/joan/hf_cache')
    tokenizer.truncation_side = 'left'
    device = torch.device(CUDA)
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='/data/joan/hf_cache', torch_dtype=torch.float16)
    model.to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Size: {model_size/1000**2:.1f}M parameters")
    
    
    model_name = 'custom'
    dataset_size = -1
    dataset, ds = create_dataset(dataset_size, text_to_use, 'flan_t5', path_to_data, 
                                    persona_cond=5, historyPairs_cond=5, priming=False, sampled_text=None, user_ids=False) 
    
    authorsToSituationVerdicts = ListDict()
    authorsToPersona = dict() 
    
    for data in tqdm(ds["train"]):
        verdict_id = dataset.idToVerdict[data['index']]
        author_id = dataset.idToAuthors[data['author_id']]
        assert author_id == dataset.verdictToAuthor[verdict_id]
        authorsToSituationVerdicts.append(author_id, (data['situation_text'], data['verdict_text']))
        authorsToPersona[(author_id, verdict_id)] = data['persona']
        
    zero_shot_test_examples = {}
    for data in ds["test"]:
        verdict_id = dataset.idToVerdict[data['index']]
        author_id = dataset.idToAuthors[data['author_id']]
        assert author_id == dataset.verdictToAuthor[verdict_id]
        
        if author_id in authorsToSituationVerdicts:
            prompt = "Writing Style\n"
            
            sampled_persona = random.sample(data['persona'], 10)
            for persona in sampled_persona:
                prompt += persona + '\n'
            prompt += f"Following the writing style, generate a comment for this situation: {data['situation_text']}"
            zero_shot_test_examples[verdict_id] = prompt
            
    zeroShot_results = {}
    text2text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

    for v, p in tqdm(zero_shot_test_examples.items()):
        output = text2text_generator(p, 
                        max_length=512,
                        do_sample=True,
                        no_repeat_ngram_size=3,
                        top_p = 0.95,
                        top_k = 0)
        for o in output:
            zeroShot_results[v] = o['generated_text']
            
    json.dump(zeroShot_results, open(os.path.join(results_dir, f'zeroShot_{model_name}_results_generateCommentPrompt.json'), 'w'))
    
    