import spacy


class SentSimilarity:
    def __init__(self, listOfSentences=None, process=False) -> None:
        self.nlp = spacy.load("en_core_web_lg")
        self.process = process
        
        if not process:
            self.nlp.remove_pipe('tagger')
            self.nlp.remove_pipe('attribute_ruler')
            self.nlp.remove_pipe('ner')
            self.nlp.remove_pipe('lemmatizer')
            self.nlp.remove_pipe('parser')
            self.nlp.remove_pipe('senter')
        
        self.listOfSentences = listOfSentences
        if listOfSentences is not None:
            self.sentObjects = [self.nlp(sent) for sent in self.listOfSentences]

    
    def update_sentences(self, newSentences):
        self.listOfSentences = newSentences
        self.sentObjects = [self.nlp(sent) for sent in self.listOfSentences]


    def remove_stopwords_fast(self, text):
        doc = self.nlp(text.lower())
        result = [token.text for token in doc if token.text not in self.nlp.Defaults.stop_words]
        return " ".join(result)


    def remove_pronoun(self, text):
        doc = self.nlp(text.lower())
        result = [token for token in doc if token.lemma_ != '-PRON-']
        return " ".join(result)


    def remove_pronoun(self, text):
        doc = self.nlp(text.lower())
        result = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
        return " ".join(result)


    def process_text(self, text):
        doc = self.nlp(text.lower())
        result = []
        for token in doc:
            if token.text in self.nlp.Defaults.stop_words:
                continue
            if token.is_punct:
                continue
            if token.lemma_ == '-PRON-':
                continue
            result.append(token.lemma_)
        return " ".join(result)


    def calculate_similarity(self, text1, text2, process=False):
        if process:
            base = self.nlp(self.process_text(text1))
            compare = self.nlp(self.process_text(text2))
        else:
            base = self.nlp(text1)
            compare = self.nlp(text2)
            
        return base.similarity(compare)


    def calculate_similarity_list(self, context, listIds):
        if self.process:
            base = self.nlp(self.process_text(context))
            return [base.similarity(self.nlp(self.process_text(self.listOfSentences[i]))) for i in listIds]
        else:
            base = self.nlp(context)
            return [base.similarity(self.sentObjects[i]) for i in listIds]