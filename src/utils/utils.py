import datetime
import string
from .clusters_utils import ListDict
#from .sent_similarity import SentSimilarity

from tqdm import tqdm 
import re
import emoji
from constants import DATETIME_PATTERN, NTA_KEYWORDS, YTA_KEYWORDS
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
import json
import numpy as np
#import nltk
import random
from constants import SEED
#nltk.download('punkt')

TIMESTAMP = str(datetime.datetime.now()).replace(" ", "_").replace(".", ":")

#sentSimilarity = SentSimilarity()


# class ExplicitEnum(Enum):
#     """
#     Enum with more explicit error message for missing values.
#     """

#     @classmethod
#     def _missing_(cls, value):
#         raise ValueError(
#             f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
#         )

def log_cuda_allocation(device, logger=print):
    #Additional Info when using cuda

    if device.type == 'cuda':
        logger(torch.cuda.get_device_name(0))
        logger('Memory Usage:')
        logger('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        logger('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        
        return json.JSONEncoder.default(self, obj)
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def preprocess_causalLM(examples, tokenizer, max_input_length):
    assert len(examples["situation_text"]) == len(examples["verdict_text"])

    # {tokenizer.additional_special_tokens[1]} {tokenizer.additional_special_tokens[0]}
    outputs = tokenizer(
        [f"{examples['situation_text'][i]} {examples['verdict_text'][i]} {tokenizer.eos_token}" for i in range(len(examples["verdict_text"]))],
        truncation=True,
        padding=True,
        max_length=max_input_length,
        #return_length=True
    )
    input_batch = []
    attention_batch = []
    labels_batch = []
    for input_ids, att_mask  in zip(outputs["input_ids"], outputs["attention_mask"]):
        input_batch.append(input_ids)
        labels_batch.append(input_ids)
        attention_batch.append(att_mask)
            
    return {"input_ids": input_batch, "attention_mask": attention_batch}#, "labels": labels_batch}

def preprocess_seq2seq(examples, tokenizer, max_input_length, max_target_length, flan_t5=False):
    if flan_t5:
        input_texts = []
        for i, _ in enumerate(examples['situation_text']):
            input_text = ''
            all_ind = range(len(examples['history_titles'][i]))
            random.seed(SEED)
            indeces = random.sample(all_ind, 10 if len(all_ind) > 10 else len(all_ind))
            #k = 10 if len(all_ind) > 10 else len(all_ind)
            
            for idx in indeces:
                input_text += f"Writing examples: \n Q: AITA for {examples['history_titles'][i][idx]} \n A: {examples['history_comments'][i][idx]} \n "
                #input_text += f"{examples['history_comments'][i][idx]} \n "

            input_text += f"Generate an opinion giving the writing examples. \n Q: {examples['situation_text'][i]}\n A: "
            #input_text += f" {examples['situation_text'][i]} {tokenizer.additional_special_tokens[0]}"
            input_texts.append(input_text)
               
        model_inputs = tokenizer(
            input_texts,
            max_length=max_input_length,
            truncation=True,
        )
    else:
        model_inputs = tokenizer(
            [f'{examples["situation_text"][i]}{tokenizer.additional_special_tokens[0]}' for i in range(len(examples["situation_text"]))],
            max_length=max_input_length,
            truncation=True,
        )
    labels = tokenizer(
        examples["verdict_text"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["verdict_id"] = examples["index"]
    model_inputs["author"] = examples["author_id"]
    return model_inputs

def preprocess_seq2seq_persona(examples, tokenizer, max_input_length, max_target_length):
    assert tokenizer.additional_special_tokens[0] == '<|verdict|>' and tokenizer.additional_special_tokens[2] == '<|persona|>'
    text =  [f'{examples["situation_text"][i]}{tokenizer.additional_special_tokens[2]}{tokenizer.additional_special_tokens[0]}' for i in range(len(examples["situation_text"]))]
    model_inputs = tokenizer(
       text,
        max_length=max_input_length,
        truncation=True,
    )
    # {tokenizer.additional_special_tokens[2]} 
    persona_inputs = tokenizer(
        [f' '.join(personas) for personas in examples["persona"]],
        max_length=max_input_length, truncation=True, padding=True
        )
    
    labels = tokenizer(
        examples["verdict_text"], max_length=max_target_length, truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["verdict_id"] = examples["index"]
    model_inputs["author"] = examples["author_id"]
    model_inputs["persona_inputs"] = persona_inputs["input_ids"]
    model_inputs["persona_attention_mask"] = persona_inputs["attention_mask"]
    return model_inputs    

def preprocess_seq2seq_persona_separate(examples, tokenizer, max_input_length, max_target_length, insert_history_token=False, max_personaHistory_length=50, persona_separate=True, history_separate=False):
    assert tokenizer.additional_special_tokens[0] == '<|verdict|>' and tokenizer.additional_special_tokens[3] == '<|persona_cls|>' and tokenizer.additional_special_tokens[4] == '<|history_cls|>'
    if insert_history_token:
        text =  [f'{examples["situation_text"][i]}{tokenizer.additional_special_tokens[3]}{tokenizer.additional_special_tokens[4]}{tokenizer.additional_special_tokens[0]}' for i in range(len(examples["situation_text"]))]
    else: # {tokenizer.additional_special_tokens[0]}
        text =  [f'{examples["situation_text"][i]}{tokenizer.additional_special_tokens[3]}{tokenizer.additional_special_tokens[0]}' for i in range(len(examples["situation_text"]))]
    
    model_inputs = tokenizer(
       text,
        max_length=max_input_length,
        truncation=True,
    )
    
    persona_inputs, persona_attention_mask, persona_lengths = get_inputIds_attMask(examples["persona"], tokenizer, max_personaHistory_length, persona_separate)
    history_titles_inputs, history_titles_attention_mask, history_titles_lengths = get_inputIds_attMask(examples["history_titles"], tokenizer, max_personaHistory_length, history_separate)
    history_comments_inputs, history_comments_attention_mask, history_comments_lengths = get_inputIds_attMask(examples["history_comments"], tokenizer, max_personaHistory_length, history_separate)

    
    labels = tokenizer(
        examples["verdict_text"], max_length=max_target_length, truncation=True
    )
    
    assert history_titles_lengths == history_comments_lengths
    
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["verdict_id"] = examples["index"]
    model_inputs["author"] = examples["author_id"]
    model_inputs["persona_inputs"] = persona_inputs
    model_inputs["persona_attention_mask"] = persona_attention_mask
    model_inputs["persona_lengths"] = persona_lengths
    
    model_inputs["history_titles_inputs"] = history_titles_inputs
    model_inputs["history_titles_attention_mask"] = history_titles_attention_mask
    model_inputs["history_comments_inputs"] = history_comments_inputs
    model_inputs["history_comments_attention_mask"] = history_comments_attention_mask
    
    model_inputs["history_lengths"] = history_comments_lengths
    return model_inputs

def get_inputIds_attMask(inputs, tokenizer, max_persona_length, separate=True):
    if separate:
        input_ids = []
        attention_masks = []
        lengths = []
        for example in inputs:
            tokenized = tokenizer(
                example,
                max_length=max_persona_length, truncation=True, padding='max_length'
                )
            input_ids.append(torch.stack([torch.tensor(a) for a in tokenized['input_ids']]))
            attention_masks.append(torch.stack([torch.tensor(a) for a in tokenized['attention_mask']]))
            lengths.append(len(example))
        return input_ids,attention_masks,lengths

    else:
        random.seed(SEED)
        tokenized = tokenizer(
            [f"{tokenizer.eos_token}".join(example) for example in inputs],
            max_length=max_persona_length, truncation=True, padding='max_length'
        )
        return tokenized['input_ids'], tokenized['attention_mask'], [-1 for _ in inputs]
    

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def postprocess_text_rogue(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
     # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def remove_extra_spaces(text):
    return re.sub("\s+",' ', re.sub("\s\s+",' ', text))        

def get_priming_sentences(authorCleanedTexts, seperating_token, TOKENS_SIZE = 100):
    authorToSampledText = dict()

    for author, texts in tqdm(authorCleanedTexts.items()):
        sampledText = ''
        tries = 0
        sample_tries = 0
        while len(sampledText.split(' ')) < TOKENS_SIZE:
            text = random.choice(texts)
            tries += 1
            
            if len(text) > 0:
                if (len(text.split(' ')) + len(sampledText.split(' ')) > TOKENS_SIZE and len(sampledText) != 0) or tries > 20:
                    sample_tries += 1
                    if sample_tries > 5 or tries > 20:
                        break
                else:
                    sampledText += text + f' {seperating_token} '
            elif tries > 20:
                break
                
        authorToSampledText[author] = sampledText
    
    return authorToSampledText

def print_args(args, logger):
    for arg in vars(args):
        logger.info("{} \t \t {}".format(arg, getattr(args, arg)))
        
        
def get_samples_per_class(labels):
    return torch.bincount(labels).tolist()


def get_current_timestamp():
    return str(datetime.datetime.now()).replace(" ", "_").replace(".", ":")


def get_verdicts_labels_from_authors(dataset, authors):
    verdicts = []
    labels = []
    for a in authors:
        if a in dataset.authorsToVerdicts:
            for v in dataset.authorsToVerdicts[a]:
                if v in dataset.verdictToId: #and dataset.verdictToTokensLength[v] > 5:
                    verdicts.append(v)
                    labels.append(dataset.verdictToLabel[v])
    return verdicts, labels


def get_verdicts_labels_from_sit(dataset, situations, postToVerdicts):
    verdicts = []
    labels = []
    for s in situations:
        if s in postToVerdicts:
            for v in postToVerdicts[s]:
                verdicts.append(v)
                labels.append(dataset.verdictToLabel[v])
    return verdicts, labels


def get_and_print_metrics(gold, predictions):
    cm = confusion_matrix(gold, predictions)
    print(cm)
    f1Score_1 = f1_score(gold, predictions, average='macro')
    print("Total f1 score macro {:3f}: ".format(f1Score_1))
    f1Score_2 = f1_score(gold, predictions, average='micro')
    print("Total f1 score micro {:3f}:".format(f1Score_2))
    f1Score_3 = f1_score(gold, predictions, average='binary')
    print("Total f1 score binary {:3f}:".format(f1Score_3))
    f1Score_4 = f1_score(gold, predictions, average='weighted')
    print("Total f1 score weighted {:3f}:".format(f1Score_4))
    accuracy = accuracy_score(gold, predictions)
    print("Accuracy {:3f}:".format(accuracy))
    
    return {'macro': f1Score_1, 'micro': f1Score_2, 'binary': f1Score_3, 'weighted': f1Score_4, 'accuracy': accuracy, 'cm': cm}


def get_metrics(gold, predictions):
    return {'macro': f1_score(gold, predictions, average='macro'), 'micro': f1_score(gold, predictions, average='micro'), 
            'binary': f1_score(gold, predictions, average='binary'), 'weighted': f1_score(gold, predictions, average='weighted'), 
            'accuracy': accuracy_score(gold, predictions), 'cm': confusion_matrix(gold, predictions)}


def timestamp_to_string(timestamp):
    dt = datetime.utcfromtimestamp(timestamp)
    return dt.strftime(DATETIME_PATTERN) 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_author_token(author):
    return f'<{author}>'
    
def has_link(string):
    # findall() has been used 
    # with valid conditions for urls in string
    #regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url = re.findall(regex,string)      
    return len(url) != 0


def split_verdicts_comments_amit(author_comments):
    author_verdicts = ListDict()
    author_otherComments = ListDict()

    for author, comments in tqdm(author_comments.items(), desc="Using keywords to extract verdicts"):
        for comment, id, parent_id in comments:
            AMIT_COMMENT_FLAG = False
            MODERATOR_FLAG = False
            
            if type(comment) == dict:
                text = comment['body']
                MODERATOR_FLAG = comment['distinguished'] != 'moderator'
            else:
                text = comment
            
            if not has_link(text):
                for key in (NTA_KEYWORDS + YTA_KEYWORDS):
                    if key in text.strip().lower() and not MODERATOR_FLAG:
                        author_verdicts.append(author, (text.strip(), id, parent_id))
                        AMIT_COMMENT_FLAG = True
                        break

                if not AMIT_COMMENT_FLAG and not MODERATOR_FLAG: #and 'distinguished' in comment and comment['distinguished'] != 'moderator':
                    author_otherComments.append(author, (text.strip(), id, parent_id))
                
    return author_otherComments, author_verdicts


def clean_keywords_from_verdicts(input):
    # Preparing replacing groups
    keywords_rep = {'ampx200b': ""}
    for key in NTA_KEYWORDS + YTA_KEYWORDS:
        keywords_rep[key] = ""
    keywords_rep = dict(sorted(keywords_rep.items(), key=lambda k: len(k[0]), reverse=True))

    rep = dict((re.escape(k), v) for k, v in keywords_rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    
    if type(input) == str:
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], input.lower())
        return text.translate(str.maketrans('', '', string.punctuation))
    elif type(input) == dict:
        print("Returning cleaned dictionary. Assuming the input dictionary is of type verdicts->text")
        cleanedDict = dict()
        for verdict, text in tqdm(input.items(), desc="Removing keywords from verdicts"):
            cleanedDict[verdict] = pattern.sub(lambda m: rep[re.escape(m.group(0))], text.lower())
            cleanedDict[verdict] = cleanedDict[verdict].translate(str.maketrans('', '', string.punctuation))
    else:
        raise Exception("Wrong input type")


EMOJI_DESCRIPTION_SCRUB = re.compile(r':(\S+?):')
HASHTAG_BEFORE = re.compile(r'#(\S+)')
BAD_HASHTAG_LOGIC = re.compile(r'(\S+)!!')
FIND_MENTIONS = re.compile(r'@(\S+)')
LEADING_NAMES = re.compile(r'^\s*((?:@\S+\s*)+)')
TAIL_NAMES = re.compile(r'\s*((?:@\S+\s*)+)$')

def process_tweet(s, save_text_formatting=True, keep_emoji=False, keep_usernames=False):
    if save_text_formatting:
        s = re.sub(r'https\S+', r'', str(s))
        s = re.sub(r'http\S+', r'', str(s))
    else:
        s = re.sub(r'http\S+', r'', str(s))
        s = re.sub(r'https\S+', r' ', str(s))
        s = re.sub(r'x{3,5}', r' ', str(s))
    
    s = re.sub(r'\\n', ' ', s)
    s = re.sub(r'\s', ' ', s)
    s = re.sub(r'<br>', ' ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'&#039;', "'", s)
    s = re.sub(r'&gt;', '>', s)
    s = re.sub(r'&lt;', '<', s)
    s = re.sub(r'\'', "'", s)

    if save_text_formatting:
        s = emoji.demojize(s)
    elif keep_emoji:
        s = emoji.demojize(s)
        s = s.replace('face_with', '')
        s = s.replace('face_', '')
        s = s.replace('_face', '')
        s = re.sub(EMOJI_DESCRIPTION_SCRUB, r' \1 ', s)
        s = s.replace('(_', '(')
        s = s.replace('_', ' ')

    s = re.sub(r"\\x[0-9a-z]{2,3,4}", "", s)
    
    if save_text_formatting:
        s = re.sub(HASHTAG_BEFORE, r'\1!!', s)
    else:
        s = re.sub(HASHTAG_BEFORE, r'\1', s)
        s = re.sub(BAD_HASHTAG_LOGIC, r'\1', s)
    
    if save_text_formatting:
        #@TODO 
        pass
    else:
        # If removing formatting, either remove all mentions, or just the @ sign.
        if keep_usernames:
            s = ' '.join(s.split())

            s = re.sub(LEADING_NAMES, r' ', s)
            s = re.sub(TAIL_NAMES, r' ', s)

            s = re.sub(FIND_MENTIONS, r'\1', s)
        else:
            s = re.sub(FIND_MENTIONS, r' ', s)
    #s = re.sub(re.compile(r'@(\S+)'), r'@', s)
    user_regex = r".?@.+?( |$)|<@mention>"    
    s = re.sub(user_regex," @user ", s, flags=re.I)
    
    # Just in case -- remove any non-ASCII and unprintable characters, apart from whitespace  
    s = "".join(x for x in s if (x.isspace() or (31 < ord(x) < 127)))
    s = ' '.join(s.split())
    return s
                
                