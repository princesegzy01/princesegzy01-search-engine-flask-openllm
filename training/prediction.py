import json
import numpy
import operator
import sys
import os
from flask import json
from flask import jsonify,request, make_response
import timeit
from sentence_transformers import SentenceTransformer, util


f = open('training/result.json')
BASE_MODEL_PATH="training/models"

data = json.load(f)
THRESHOLD_SCORE = 0.5

import os
print(os.path.abspath("."))

start = timeit.default_timer()

trigger_word_list = {'Other', 'greater', 'cm', 'inches', 'diameter', 'measuring', 'less', 'minimum', 'Weighing', 'Other:'}


model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def predict(query, path):

    tree = ""
    if(path == "0"):
        tree = data
    else:

    # print()
    # split path into list and remove first path
        path_list = path[2:].split("/")
        print("path  list : ")
        print(path_list)

        _temp_data = data

        for leaf in path_list:

            #  check if data has children,
            # if not, return it as the last part
            # print(_temp_data[int(leaf)])
            print("********************************")
            if not "children" in _temp_data[int(leaf)]:
                response_data = {
                'q' : query,
                "data" : {
                    "htsno" : _temp_data[int(leaf)]['htsno'],
                    "description" :_temp_data[int(leaf)]['description'],
                    "score" : "0.0"
                },
                "type" : 'response',
                "path" : path,
                "timer" : timeit.default_timer() - start
                }

                return response_data
        
            _temp_data = _temp_data[int(leaf)]['children']
        tree  = _temp_data

    query_embedding = model.encode(query)

    model_path = os.path.join(BASE_MODEL_PATH, path)
    print(model_path)

    
    passage_embedding = numpy.load(model_path + '/emb.npy')
    # passage_embedding = numpy.load('training/models/0/emb.npy')
    search_result = util.dot_score(query_embedding, passage_embedding)

    map_result = []
    for index,res in  enumerate(search_result[0].numpy()):
        map_result.append({'index' : index, 'score' : res})
      
    
    reduced_result = list(filter(lambda x: x['score'] >= THRESHOLD_SCORE, map_result))
    reduced_result_sorted = sorted(reduced_result, key=lambda x: x['score'], reverse=True)


    print(len(reduced_result_sorted))

    # if no match was found, return all children
    if(len(reduced_result_sorted) == 0):
        
        # if path is 0
        # return empty response
        if(path == "0"):
            response_data = {
                'q' : query,
                "data" : [],
                "type" : 'empty_base',
                "path" : path,
                "timer" : timeit.default_timer() - start
            }

            return response_data


        # else print the element of the current node
        print("list is empty")
        watch_list_token = []
        [watch_list_token.extend(t['description'].split(" ")) for t in tree]
    
        i=set(watch_list_token).intersection(trigger_word_list)


        _response_data = []
        for index, branch in enumerate(tree):
            _response_data.append({
                'index' : index,
                'htsno' : branch['htsno'],
                'description' : branch['description']
            })
            

        response_data = {
                'q' : query,
                "data" : _response_data,
                "type" : 'request',
                "path" : path,
                "timer" : timeit.default_timer() - start
            }

        return response_data



    current_data = reduced_result_sorted[0]


    new_path = os.path.join(path, str(current_data['index']))
    new_path_with_base = os.path.join(BASE_MODEL_PATH, new_path)

    print("npb > : " + new_path_with_base)
    print("base > : " + BASE_MODEL_PATH)
    print("np > : " + new_path)
    print("path > : " + path)
    print("crd > : " + str(current_data['index']))

    # pull all documnets  tokent in a list
    # for document in tree[current_data['index']]:

    # check if documents contains trigger wordlist

    if(os.path.isdir(new_path_with_base)):
        print("path x :" + new_path)
        print("Similarity Score : "  + str(current_data['score']))
        print("Description : " + tree[current_data['index']]['description'])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")
        # predict(query, tree[current_data['index']]['children'], new_path)
        return predict(query, new_path)
    else:
        print("###############################################")
        print("path :" + new_path)
        print("Document Index : " + str(current_data['index']))
        print("Similarity Score : "  + str(current_data['score']))
        # print(data[current_data['index']]['description'])
        print("Description :" + tree[current_data['index']]['description'])
        print("final htsno : " + tree[current_data['index']]['htsno'])

        response_data = {
                'q' : query,
                "data" : {
                    "htsno" : tree[current_data['index']]['htsno'],
                    "description" : tree[current_data['index']]['description'],
                    "score" : str(current_data['score'])
                },
                "type" : 'response',
                "path" : new_path,
                "timer" : timeit.default_timer() - start
            }

        return response_data

# response  = predict("Live chicken", "0")
# print("response : >>>>>>>>>>>" )
# print(response)
    
    

