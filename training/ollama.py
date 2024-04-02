from transformers import pipeline, set_seed
import spacy
import sys
import nltk
import json
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import wn
import re
import string
import requests

nlp = spacy.load("en_core_web_sm")

stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)
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

    data = { 
       "model": "llama2", 
        "prompt": "describe " + query  + " in english language", 
        "stream": False 
    }
    json_data = json.dumps(data)

    # Send the POST request with JSON data
    headers = {'Content-type': 'application/json'}
    response = requests.post('http://localhost:11434/api/generate', data=json_data, headers=headers)
    response = response.json()

    response = response['response']

    print(response)
    # sys.exit(0)
    word_list = []        

    tokens = preprocessText(response)

    # print(tokens)
    tags = nlp(tokens)

    tokens = [word.text for word in tags if (word.pos_ == 'NOUN')]
    word_list.extend(tokens)

    disticnt = set(word_list)
    return Counter(list(disticnt))
    