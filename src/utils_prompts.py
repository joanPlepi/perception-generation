import random
from tqdm import tqdm
from utils.clusters_utils import ListDict



def extract_authorsToSituationVerdicts(dataset, ds, onlyComments = False):
    authorsToSituationVerdicts = ListDict()
    authorsToPersona = dict() 
    splits = ["train", "val"]

    for split in splits:
        for data in tqdm(ds[split], desc="Extracting the few shot examples"):
            verdict_id = dataset.idToVerdict[data['index']]
            author_id = dataset.idToAuthors[data['author_id']]
            assert author_id == dataset.verdictToAuthor[verdict_id]
            authorsToSituationVerdicts.append(author_id, (data['situation_text'], data['verdict_text']))
            authorsToPersona[(author_id, verdict_id)] = data['persona']
    return authorsToSituationVerdicts


def get_persona_prompts(dataset, ds, splits=["train", "val"]):
    prompts_dataset = {}
    for split in splits:
        if split == "test":
            prompts_dataset[split] = {"text": [], "verdict": []}
        else:
            prompts_dataset[split] = {"text": []}


    system_prompt = """I will provide persona sentences and perspectives written from a user. 
            Generate a perspective for the given situation. Perspective should be aligned with the user inferred personality. 
            The perspective should start with the verdict YTA or NTA. YTA means You're the asshole and NTA means Not the asshole.
            """
    authorsToSituationVerdicts = extract_authorsToSituationVerdicts(dataset, ds)

    for split in splits:
        for data in ds[split]:
            verdict_id = dataset.idToVerdict[data['index']]
            author_id = dataset.idToAuthors[data['author_id']]
            assert author_id == dataset.verdictToAuthor[verdict_id]
            
            if author_id in authorsToSituationVerdicts:
                prompt = "Sampled persona and perspectives for user: \n"
                
                k = 10
                sampled_persona = random.sample(data['persona'], k)
                for persona in sampled_persona:
                    prompt += persona + '\n'
                        
                prompt += f"Situation: {data['situation_text']}\n"
            

            if split == "test":
                llama_prompt_template = f"""### Instruction
                        {system_prompt}
                        
                        ### Input:
                        {prompt}

                        ### Users Perspective:
                        """
                prompts_dataset[split]["verdict"].append(verdict_id)
            else:
                llama_prompt_template = f"""### Instruction
                    {system_prompt}
                    
                    ### Input:
                    {prompt}

                    ### Users Perspective:
                    {data['verdict_text']}
                    """

                #example = {"prompt": llama_prompt_template, "completion": data['verdict_text']}
            prompts_dataset[split]["text"].append(llama_prompt_template)
        

    print(llama_prompt_template)
    return prompts_dataset