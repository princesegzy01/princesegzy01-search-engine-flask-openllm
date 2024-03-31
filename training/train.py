import json
import os
from sentence_transformers import SentenceTransformer, util
import numpy
import sys

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


f = open('result.json')
data = json.load(f)


def trainModel(path, documents):

    print(str(len(documents)))
    # sys.exit()
    os.makedirs("models", exist_ok = True)
    full_path = os.path.join("models", path)

    os.makedirs(full_path, exist_ok=True)
    # pass

    sentenceList = []

    for index, child in enumerate(documents):
        sentenceList.append(child['description'])

    
    passage_embedding = model.encode(sentenceList)
    numpy.save(os.path.join(full_path, 'emb.npy'), passage_embedding)     

    
    passage_embedding = numpy.load('training/models/0/emb.npy') 
    # print(str(len(passage_embedding)))  
    # sys.exit(0)

def dataLooper(leaf, tree, prev_index):
    
    if(leaf == 0):
        trainModel("0", tree)
    else:
        trainModel(prev_index, tree)
        
    
    for index, parent in enumerate(tree):

        # train model here
        # path = path+"/"+ str(leaf)

        # print(("\t" * leaf + " >> Htsno: ") + parent['htsno'])
        # print(("\t" * leaf + " >> Description: ") + parent['description'])
        # print(("\t" * leaf + " >> ") + "leaf : " +  str(leaf))
        # print(("\t" * leaf + " >> ") + "index " + str(index))
        # print(("\t" * leaf + " >> ") +  "parent len : " + str(len(parent)))
        # print(("\t" * leaf + " >> ") +  "path : " + path)
        
       

        if "children" in parent:

            new_leaf = leaf + 1
            str_index = prev_index + "/" +str(index)

            # print(parent['description'])
            print( ("\t" * leaf + " >> ") +  "child len : " + str(len(parent['children'])))
            # print( ("\t" * leaf + " >> ") +  "path : " + path)
            print(("\t" * leaf + " >> ") + "leaf : " +  str(leaf))
            print(("\t" * leaf + " >> ") + "index " + str(index))
            print( ("\t" * leaf + " >> ") +  "path : " + prev_index)

            # trainModel(prev_index, parent['children'])
            # # print(parent['children'])
            # new_leaf = leaf + 1
            # str_index = prev_index + "/" +str(index)
            dataLooper(new_leaf, parent['children'],str_index)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(type(child))

        # if type(child) is list and len(child) > 0:
        #     



# print(data)
# print(str(len(data)))

dataLooper(0, data, "0")
# trainModel("")