from transformers import pipeline, set_seed
import spacy
import sys
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import wn
import re
import string

nlp = spacy.load("en_core_web_sm")

stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)
# en = wn.Wordnet('oewn:2023')
translator = str.maketrans('', '', string.punctuation)


def preprocessText(text):
    #  remove numbers
    text = text.replace('.com','')
    text = text.replace('http','')
    text = text.replace('https','')
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(translator)
    text = " ".join(text.split())

    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if word not in stop_words]

    return " ".join(text)

def fetchLLM(query):

    generator = pipeline('text-generation', model='gpt2-xl')
    set_seed(42)
    data = generator("what is " + query, max_length=30, num_return_sequences=20, truncation=True)
    
    word_list = []

    for sentence in data:

        tokens = preprocessText(sentence['generated_text'])
        tags = nlp(tokens)

        tokens = [word.text for word in tags if (word.pos_ == 'NOUN')]
        word_list.extend(tokens)

    disticnt = set(word_list)
    return Counter(list(disticnt))
    