from utils.utils import sentSimilarity
from utils.read_files import read_persona
import pandas as pd
import pickle as pkl
from tqdm import tqdm 
from dataset import SocialNormDataset


def process_personas_most_sim(persona_sentences, comment, k=20):
    persona = list(set(persona_sentences))
    sentSimilarity.update_sentences(persona)
    similarities = sentSimilarity.calculate_similarity_list(comment, range(len(persona)))
    
    persona_sim = list(zip(persona, similarities))
    persona_sim = sorted(persona_sim, reverse=True, key=lambda item: item[1])
    return persona_sim[:k]


if __name__=='__main__':
    path_to_data = '/app/public/joan/' 
    persona_dirname = '/data/joan/users_perception/users_history_persona_v3_noAmit'

    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts_and_authors.gzip', compression='gzip')

    with open(path_to_data+'social_norms_clean.csv') as file:
        social_comments = pd.read_csv(file)
        
    dataset = SocialNormDataset(social_comments, social_chemistry, persona_dirname)
    authors = set(dataset.authorsToVerdicts.keys())
    
    authors_personas = read_persona(persona_dirname, authors)
        
    filteredAuthorsToPersonas = {a: v for a, v in authors_personas.items() if len(v) > 20 and len(v) < 500}
    print(len(filteredAuthorsToPersonas))
    
    authorVerdictToPersonas = {author: dict() for author in filteredAuthorsToPersonas}

    for situation, verdicts in tqdm(dataset.postToVerdicts.items()):
        for verdict in verdicts:
            author = dataset.verdictToAuthor[verdict]
            if author in filteredAuthorsToPersonas:
                authorVerdictToPersonas[author][verdict] = process_personas_most_sim(filteredAuthorsToPersonas[author], dataset.verdictToText[verdict])
                
                
    pkl.dump(authorVerdictToPersonas, open('/data/joan/perception_generation/verdictsToTop20Personas.pkl', 'wb'))
                