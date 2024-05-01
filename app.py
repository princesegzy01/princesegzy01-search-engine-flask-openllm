from flask import Flask
from training import prediction
from training import llm
from training import ollama
from training import cluster
from flask import jsonify,request, make_response
import timeit

app = Flask(__name__)
 
@app.route('/hello/<name>',  methods=['POST', 'GET'])
def hello_name(name):
   return 'Hello %s!' % name
 
@app.route('/predict',  methods=['POST', 'GET'])
def predict():
    content = request.json
    print(content)

    query = content['q']
    path = content['path']
    
    if(path == "0"):
      print(">>>>> fetchLLM")

      response = prediction.predict(query, path, "No")
      print("01 >>>>>>>>>>>>>>>>>")
      print(response)

      # if response at path 0 return empty
      # use AI
      if(response['type'] == "empty_base"):
         # use AI
         # response = llm.fetchLLM(original_query)
         response = ollama.fetchLLM(query)
         print("02 >>>>>>>>>>>>>>>>>")
         print(response)

         response_list = [k for k, v in response.items()]
         
         # pass response from lamma2 in array of string
         processed_query = cluster.processCluster(response_list, query)
         return prediction.predict(processed_query, path, "Yes")

      return response

      # start = timeit.default_timer()
      # return jsonify({"data" : response, "processed_query" : processed_query, "timer" : timeit.default_timer() - start})
    else:
       print("direct prediction")
       return prediction.predict(query, path, "N/A")
       


    # return content
    # return prediction.predict("Live chicken", "0")
    # return prediction.predict(content['q'], content['path'])

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=3004)