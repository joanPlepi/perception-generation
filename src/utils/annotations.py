import random
from tqdm import tqdm
from utils.utils import sentSimilarity

def write_author(f, persona_sentences, history_situation_pairs, idx, full):
    f.write('<u>Persona sentences:</u> \n<br><br>')
    
    for i in idx:
        sentence = persona_sentences[i]
        keywords = ['NTA','YTA','NAH','ESH','INFO']
        if not any(key in sentence.upper() for key in keywords):
            f.write(f'{sentence} \n<br>')
        
    if full:
        f.write('\n<br><br>')
        f.write('<u>Other situation title- comment pairs for the author</u> \n<br><br>')
        
        for title, otherVerdict in history_situation_pairs:
            f.write(f'<strong> Title: {title} </strong> <br><em>Comment:</em> {otherVerdict} \n<br><br>')
            

def write_example(ex, f, full):
    id = ex['id']
    title = ex['title']
    situation = ex['situation_text']
    comment = ex['comment']
    
    f.write(f'\n\n[[Block:MC Block]]\n[[Question:MC:SingleAnswer:Horizontal]]\n[[ID:q{id}]]\n')
    f.write(f'<strong>{title}</strong> \n<br> {situation}\n <hr> <em>Comment:</em> {comment} \n<hr>')
    
    rand = random.randint(0,1)
    
    if rand == 0:
        f.write('<br><br> <center>Author: <strong> A </strong></center> <br>')
        write_author(f, list(set(ex['persona'])), ex['history_pairs'], ex['sampled_idx'], full)
        f.write('\n<br><hr><hr>')
        f.write('<br> <center>Author: <strong> B </strong></center> <br>')
        write_author(f, list(set(ex['other_persona'])), ex['other_history_pairs'], ex['other_sampled_idx'], full)
        correct = 'A'
    elif rand == 1:
        f.write('<br><br> <center>Author: <strong> A </strong></center> <br>')
        write_author(f, list(set(ex['other_persona'])), ex['other_history_pairs'], ex['other_sampled_idx'], full)
        f.write('\n<br><hr><hr>')
        f.write('<br> <center>Author: <strong> B </strong></center> <br>')
        write_author(f, list(set(ex['persona'])), ex['history_pairs'], ex['sampled_idx'], full)
        correct = 'B'
    else:
        raise Exception("")
    
    f.write('\n<br><br><hr><br><br>')
    f.write('\nWho is the author that wrote the initial comment?\n [[Choices]]\n A \n B \n\n')
    
    return correct

def find_most_sim_diverse(persona_sim, k, only_most_sim=True):
    if only_most_sim:
        return set(range(k))
    
    sampled_idx = set()
    sampled_idx.add(0) 

    for _ in range(k-1):
        current_diverse_sim = {}
        
        for i, (sentence, sim) in enumerate(persona_sim):
            if i not in sampled_idx:
                similarities = sentSimilarity.calculate_similarity_list(sentence, sampled_idx)
                current_diverse_sim[i] = sim + min(similarities)
        
        sampled_idx.add(max(current_diverse_sim, key=current_diverse_sim.get))            
        
    return sampled_idx

def process_personas(persona_sentences, comment, k=20):
    persona = list(set(persona_sentences))
    sentSimilarity.update_sentences(persona)
    similarities = sentSimilarity.calculate_similarity_list(comment, range(len(persona)))
    
    persona_sim = list(zip(persona, similarities))
    persona_sim = sorted(persona_sim, reverse=True, key=lambda item: item[1])
    sampled_idx = find_most_sim_diverse(persona_sim, k, only_most_sim=False)
    
    return persona_sim, sampled_idx

def get_final_examples_v2(dataset, examples, filteredAuthorsToPersonas):
    final_examples = {'description': 'Examples created with the second version of annotations', 'data': []}

    n = len(examples)

    for i, ex in tqdm(enumerate(examples), desc='Extracting examples'):
        ex['id'] = i
        if i < n//2:
            author, failed = sample_author_from_situation(dataset,filteredAuthorsToPersonas, ex['postId'], ex['author'], ex['label'], same_label_sample=True)
            ex['desc'] = f'Tried negative example sampled with the same label. Failed = {failed}'
        else:
            author, failed = sample_author_from_situation(dataset, filteredAuthorsToPersonas, ex['postId'], ex['author'], ex['label'], same_label_sample=False)
            ex['desc'] = f'Tried negative example sampled with different label. Failed = {failed}'

        assert ex['author'] != author

        ex['other_author'] = author
        ex['other_persona'] = filteredAuthorsToPersonas[author]
        ex['other_history_pairs'] = get_pairs_sit_verdict_for_author(dataset, author, ex['postId'])

        final_examples['data'].append(ex)  
        
    return final_examples

def sample_author_from_situation(dataset, filteredAuthorsToPersonas, postId, author, label, same_label_sample, limit=1):
    sampled_verdict = random.choice(dataset.postToVerdicts[postId])
    new_author = dataset.verdictToAuthor[sampled_verdict]
    
    if same_label_sample:
        tries = 0
        while dataset.verdictToLabel[sampled_verdict] != label or new_author == author or new_author not in filteredAuthorsToPersonas:
            sampled_verdict = random.choice(dataset.postToVerdicts[postId])
            new_author = dataset.verdictToAuthor[sampled_verdict]
            
            tries += 1
            if tries > limit and new_author in filteredAuthorsToPersonas and new_author != author:
                #print("Could not find the same label post")
                return new_author, True
            
        return new_author, False

    else:
        tries = 0
        while dataset.verdictToLabel[sampled_verdict] == label or dataset.verdictToAuthor[sampled_verdict] == author or new_author not in filteredAuthorsToPersonas:
            sampled_verdict = random.choice(dataset.postToVerdicts[postId])
            new_author = dataset.verdictToAuthor[sampled_verdict]

            tries += 1
            if tries > limit and new_author in filteredAuthorsToPersonas and new_author != author:
                #print("Could not find the different label post")
                return new_author, True
            
        return new_author, False
