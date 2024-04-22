from utils.utils import *
from utils.train_utils import *
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.train_utils import create_dataset
from peft import PeftModel
# Load model directly
import torch
from utils_prompts import get_persona_prompts
from argparse import ArgumentParser
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
parser.add_argument("--checkpoint_path", dest="checkpoint_path", default="/home/plepi/perception-generation/results/sft_llama2_checkpoints/checkpoint-6309", type=str)
parser.add_argument("--text_to_use", dest="text_to_use", default= "title", type=str, help="Which text of situations to use, post or title")

if __name__ == '__main__':
    args = parser.parse_args()
    path_to_data = args.path_to_data
    text_to_use = args.text_to_use
    model_name = args.model_name

    model_name = 'custom'
    dataset_size = -1
    dataset, ds = create_dataset(dataset_size, text_to_use, 'flan_t5', path_to_data, persona_amount=20,
                                    persona_cond=5, historyPairs_cond=5, priming=False, sampled_text=None, user_ids=False) 


    model = AutoModelForCausalLM.from_pretrained(
                                            model_name, use_cache=False, 
                                            # quantization_config=quant_config, 
                                            device_map="auto"
                                            )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    path = args.checkpoint_path
    model = PeftModel.from_pretrained(model, path)
    model = model.merge_and_unload()

    text_generator = pipeline("text-generation", model=model, torch_dtype=torch.float16, tokenizer=tokenizer, device_map="auto", return_full_text=False)

    prompts_dataset = get_persona_prompts(dataset, ds, splits=["train", "val", "test"])

    prompts_ds = DatasetDict({
        "train": Dataset.from_dict(mapping=prompts_dataset["train"], features=Features({'text': Value(dtype='string')})),
        "val": Dataset.from_dict(mapping=prompts_dataset["val"], features=Features({'text': Value(dtype='string')})),
        "test": Dataset.from_dict(mapping=prompts_dataset["test"], features=Features({'text': Value(dtype='string'), 'verdict': Value(dtype='string')})),
        })
    
    print("Starting inference...")
    results = {}
    for data in tqdm(prompts_ds["test"]): 
        verdict = data["verdict"]

        output = text_generator(data["text"], 
                            max_new_tokens=100,
                            # do_sample=False,
                            # num_return_sequences=1,
                            # eos_token_id=text_generator.tokenizer.eos_token_id,
                            # no_repeat_ngram_size=3,
                            # top_p = 0.95,
                            # top_k = 0
                            )
        for o in output:
            results[verdict] = o['generated_text']

    
    print("Saving results...")
    json.dump(results, open(f'../results/sft_llama2_7b_persona.json', 'w'))
