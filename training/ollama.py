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

    # Prepare the JSON data
    data = { 
       "model": "llama2", 
        "prompt": "What is " + query, 
        "stream": False 
    }
    json_data = json.dumps(data)

    # Send the POST request with JSON data
    headers = {'Content-type': 'application/json'}
    response = requests.post('http://localhost:11434/api/generate', data=json_data, headers=headers)
    response = response.json()



    # data = {
    #     "model": "llama2",
    #     "created_at": "2024-04-02T14:04:03.963737321Z",
    #     "response": "PlayStation 5 (PS5) is a next-generation video game console developed by Sony Interactive Entertainment. It was released on November 12, 2020, and is the successor to the PlayStation 4. The PS5 features improved hardware and software capabilities compared to its predecessor, including faster processors, more memory, and support for ray tracing and artificial intelligence-based graphics.\n\nSome of the key features of the PS5 include:\n\n1. Improved performance: The PS5 features a custom 8-core AMD Zen 2 processor with 36 compute units, along with 8 GB of GDDR6 RAM and a 512 MB PS5 Memory Unit. This provides significantly faster performance than the PS4, with improved load times and smoother gameplay.\n2. Next-generation graphics: The PS5 features a custom Radeon RDNA 2 GPU with support for ray tracing and artificial intelligence-based graphics. Ray tracing allows for more realistic lighting and reflections in games, while AI-based graphics can create more detailed and realistic environments.\n3. High-fidelity audio: The PS5 features a custom 3D spatial audio system that can simulate the sound of footsteps, weapons fire, and other game elements in 3D space. This provides a more immersive gaming experience for players.\n4. Enhanced controller: The PS5's controller has been redesigned with haptic feedback, which allows for more realistic tactile sensations during gameplay. It also features a new \"Create\" button that allows players to easily access various creative tools and features.\n5. Cloud gaming: The PS5 supports cloud gaming through the PlayStation Now service, allowing players to stream games directly from the cloud to their console or PC without the need for a physical copy of the game.\n6. Cross-platform play: The PS5 supports cross-platform play with other consoles and devices, such as PCs and mobile devices, allowing players to connect with friends and other gamers across different platforms.\n7. Virtual reality: The PS5 is compatible with PlayStation VR, which provides an immersive virtual reality gaming experience for players.\n8. Remote play: The PS5 allows players to remotely play games on their console using the PlayStation App on their smartphone or tablet.\n9. Content creation: The PS5 features a built-in video editor and sharing tools, allowing players to create and share their own content directly from their console.\n10. Expandable storage: The PS5 has an expandable storage capacity through the use of external hard drives, allowing players to store more games and other content on their console.\n\nOverall, the PS5 offers a significant improvement over its predecessor in terms of performance, graphics, and features, providing a more immersive and engaging gaming experience for players.",
    #     "done": True,
    #     "context": [],
    #     "total_duration": 77515291934,
    #     "load_duration": 339248,
    #     "prompt_eval_count": 12,
    #     "prompt_eval_duration": 764285000,
    #     "eval_count": 633,
    #     "eval_duration": 76749783000
    #     }
    # response =  data['response']


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
    