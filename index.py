import json
import wn
import sys
# import os, sys
# from wn.similarity import path
# from nltk.corpus import stopwords
# import re
# from string import punctuation
# import transformers
import numpy



# from transformers import AutoTokenizer
# from transformers import pipeline
# # model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# model_name = "FacebookAI/roberta-base"
# # model = transformers.RobertaModel.from_pretrained('roberta-base')

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# classifier = pipeline('feature-extraction',model=model_name, tokenizer=tokenizer)

# # text = 'This is a sample text'
# # tokens = tokenizer(text)
# # emb = classifier(text, max_length=512)
# # data = pipeline("this is a test")

# # print(tokens)
# # print(emb)
# # print(len(emb[0]))
# # sys.exit(0)


# # Load the RoBERTa model

# # Tokenize and encode the texts
# text1 = "This is the first text."
# text2 = "This is the second text."

# # model.
# # print(dir(model))
# # sys.exit(0)
# encoding1 = classifier(text1, max_length=50)
# encoding2 = classifier(text2, max_length=512)


# print(encoding1)
# sys.exit(0)


# n12 = numpy.squeeze(numpy.asarray(encoding1))
# X12 = numpy.squeeze(numpy.asarray(encoding2))


# # # Calculate the cosine similarity between the embeddings
# # similarity = numpy.dot(encoding1, encoding2) / (numpy.linalg.norm(encoding1) * numpy.linalg.norm(encoding2))


# similarity = numpy.dot(encoding1, encoding2) / (numpy.linalg.norm(encoding1) * numpy.linalg.norm(encoding2))
# print(similarity)
# sys.exit(0)



en = wn.Wordnet('oewn:2023')
# stopwords_ = set(stopwords.words("english"))



# def cleanText(sample_text):

#     sample_text = sample_text.replace('-', ' ')
#     sample_text = re.sub(r"<.*?>", " ", sample_text)

#     tokens = sample_text.split()
#     clean_tokens = [t for t in tokens if not t in stopwords_]
#     # clean_tokens = [t for t in clean_tokens if not t.is in stopwords_]

#     clean_tokens = " ".join([w for w in clean_tokens if not w.isdigit()]) # Side effect: removes extra spaces
#     clean_tokens = re.sub(f"[{re.escape(punctuation)}]", "", clean_tokens)
#     clean_tokens = "".join([w for w in clean_tokens if not w.isdigit()]) # Side effect: removes extra spaces


#     # clean_text = " ".join(clean_tokens)
#     # return clean_text
#     return clean_tokens
#     # print_text(sample_text, clean_text)

ss = en.synsets('game console', pos='n')[0]
print(ss.definition())

sys.exit(0)

# # sl = en.synsets('win', pos='v')
# # for s in sl:
# #     print(s.definition())


# # turn_over = en.synsets('win', pos='v')
# # print(len(turn_over))

# sys.exit(0)
f = open('result.json')
data = json.load(f)


sentenceList = []

for index, child in enumerate(data):
    # print("hstsno : " + child['htsno'])
    # print("description : " + child['description'])
    # print('***************')
    # # print("clean description : " + cleanText(child['description']))
    # print("Index : " + str(index))
    # print("Child length : " + str(len(data[index])))
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print("")
    sentenceList.append(child['description'])


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

query_embedding = model.encode("Live chicken")
# query_embedding = model.encode("black v-neck shirt with button")
# passage_embedding = model.encode([
#     "London has 9,787,426 inhabitants at the 2011 census",
#     "London is known for its finacial district",
# ])

# passage_embedding = model.encode(sentenceList)
# numpy.save('heirachy.npy', passage_embedding)
passage_embedding = numpy.load('heirachy.npy')


search_result = util.dot_score(query_embedding, passage_embedding)
# print("Similarity:", search_result[0])

map_result = []
for index,res in  enumerate(search_result[0].numpy()):

    map_result.append({'index' : index, 'score' : res})
    # print(res)

# print(map_result)

reduced_result = list(filter(lambda x: x['score'] >= 0.5, map_result))
# print(reduced_result)

for x in reduced_result:
    print("Document Index : " + str(x['index']))
    print("Similarity Score : "  + str(x['score']))
    print(data[x['index']]['description'])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")