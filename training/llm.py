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
en = wn.Wordnet('oewn:2023')
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
    # return x

    # _x = [{'generated_text': "what is playstation but what i do you can't beat up a mook i'm good with that\n\nUser Info: Luka_P"}, {'generated_text': 'what is playstation 4 and nintendo console and how long before are we going to be done with the Wii U".\n\n"We still have'}, {'generated_text': "what is playstation 4 really like?\n\nWhat does Sony have to offer the PS4's dedicated gaming audience that has been waiting so long for"}, {'generated_text': 'what is playstation3 and do i need it to get the game, is it the same for gba and xbox. so whats next?'}, {'generated_text': 'what is playstation.com all about and why it is so vital for us as a consumer to understand and appreciate it. This week we get an'}, {'generated_text': 'what is playstation, and how do we start playing? Playstation is a computer game console. There is a wide variety of games played on a'}, {'generated_text': 'what is playstation?\n\n\nI want someone to explain how games are made and distributed. What is a developer? So you can make all this'}, {'generated_text': "what is playstation 4?\n\nPlaystation 4 is the all new slim version of the iconic Sony PlayStation console - with a gorgeous body that's"}, {'generated_text': 'what is playstation?" and I thought "oh wow, this is really a huge deal". So I made the effort to get the code, and'}, {'generated_text': 'what is playstation3.com, and the playstation3.com login page where you enter a special code that enables the free download of the'}, {'generated_text': 'what is playstation 4 for me", is basically the "warranty of new games" argument. That is that if you buy 2 games,'}, {'generated_text': 'what is playstation 4 really good for?" and then the other three answers with a question mark at the end and click on all of them to get'}, {'generated_text': 'what is playstation 4?"\n\n\nFor me playstation 4 is an ipad, a tablet, another ipad, a mpc, a'}, {'generated_text': "what is playstation 2's most important characteristic and why it is used and how Sony has evolved it).\n\n- Sony was working on a ps"}, {'generated_text': 'what is playstation 4 games compatible with my ipad 8gb with thunderbolt 4 and am I capable of running 1080p60)\n\n\nI'}, {'generated_text': 'what is playstation4network.com? What are you looking for? [spoiler=spoiler=weird dude][img]https://'}, {'generated_text': 'what is playstation4.com.'}, {'generated_text': 'what is playstation 4).\n\nWhen I was a little kid, I loved playing arcade games.\n\nSo, when I was 7,'}, {'generated_text': 'what is playstation network, which is not a real answer\n\nAnonymous 05/11/15 (Tue) 09:02:43 AM No'}, {'generated_text': 'what is playstation) I would do the whole thing over.... I was looking and trying to figure out what "Playstation" was, I found'}]
    # return _x

    # data = [{'generated_text': "what is playstation but what i do you can't beat up a mook i'm good with that\n\nUser Info: Luka_P"}, {'generated_text': 'what is playstation 4 and nintendo console and how long before are we going to be done with the Wii U".\n\n"We still have'}, {'generated_text': "what is playstation 4 really like?\n\nWhat does Sony have to offer the PS4's dedicated gaming audience that has been waiting so long for"}, {'generated_text': 'what is playstation3 and do i need it to get the game, is it the same for gba and xbox. so whats next?'}, {'generated_text': 'what is playstation.com all about and why it is so vital for us as a consumer to understand and appreciate it. This week we get an'}, {'generated_text': 'what is playstation, and how do we start playing? Playstation is a computer game console. There is a wide variety of games played on a'}, {'generated_text': 'what is playstation?\n\n\nI want someone to explain how games are made and distributed. What is a developer? So you can make all this'}, {'generated_text': "what is playstation 4?\n\nPlaystation 4 is the all new slim version of the iconic Sony PlayStation console - with a gorgeous body that's"}, {'generated_text': 'what is playstation?" and I thought "oh wow, this is really a huge deal". So I made the effort to get the code, and'}, {'generated_text': 'what is playstation3.com, and the playstation3.com login page where you enter a special code that enables the free download of the'}, {'generated_text': 'what is playstation 4 for me", is basically the "warranty of new games" argument. That is that if you buy 2 games,'}, {'generated_text': 'what is playstation 4 really good for?" and then the other three answers with a question mark at the end and click on all of them to get'}, {'generated_text': 'what is playstation 4?"\n\n\nFor me playstation 4 is an ipad, a tablet, another ipad, a mpc, a'}, {'generated_text': "what is playstation 2's most important characteristic and why it is used and how Sony has evolved it).\n\n- Sony was working on a ps"}, {'generated_text': 'what is playstation 4 games compatible with my ipad 8gb with thunderbolt 4 and am I capable of running 1080p60)\n\n\nI'}, {'generated_text': 'what is playstation4network.com? What are you looking for? [spoiler=spoiler=weird dude][img]https://'}, {'generated_text': 'what is playstation4.com.'}, {'generated_text': 'what is playstation 4).\n\nWhen I was a little kid, I loved playing arcade games.\n\nSo, when I was 7,'}, {'generated_text': 'what is playstation network, which is not a real answer\n\nAnonymous 05/11/15 (Tue) 09:02:43 AM No'}, {'generated_text': 'what is playstation) I would do the whole thing over.... I was looking and trying to figure out what "Playstation" was, I found'}]
    word_list = []

    for sentence in data:

        tokens = preprocessText(sentence['generated_text'])
        tags = nlp(tokens)

        tokens = [word.text for word in tags if (word.pos_ == 'NOUN')]
        word_list.extend(tokens)

    disticnt = set(word_list)
    return Counter(list(disticnt))
    