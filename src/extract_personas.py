import pandas as pd
from tqdm import tqdm 
from dataset import SocialNormDataset
from utils.utils import *
from utils.train_utils import *
import glob
import os
from joblib import Parallel, delayed
from spacy.lang.en import English

from utils.read_files import extract_authors_vocab_fullHistory, write_persona


if __name__ == '__main__':
    path_to_data = '/app/public/joan/' 
    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts_and_authors.gzip', compression='gzip')

    path_to_data = '/data/joan/users_perception/data_bela/'
    with open(path_to_data+'social_norms_clean.csv') as file:
        social_comments = pd.read_csv(file)
        
    dataset = SocialNormDataset(social_comments, social_chemistry)
    authors = set(dataset.authorsToVerdicts.keys())
    
    dirname = '/data/joan/users_perception/users_history_full'
    print(f'Processing text files from directory {dirname}')
    filenames = sorted(glob.glob(os.path.join(dirname, '*')))
    
    # -30, starts from 2019
    results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_fullHistory)(filename, authors) for filename in tqdm(filenames[-30:]))
    
    print("Merging results")
    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)
        
    

    nlp = English()
    nlp.add_pipe("sentencizer")
        
    for author, vocabTimes in tqdm(authors_vocab.items()):
        path = os.path.join('/data/joan/users_perception/users_history_persona_v3_noAmit', author) + '.txt'
        write_persona(path, vocabTimes, nlp)