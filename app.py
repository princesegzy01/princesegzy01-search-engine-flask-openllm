from flask import Flask
from training import prediction
from training import llm
from training import ollama
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
    original_query = content['original_query']
    path = content['path']
    helper = content['helper']

    if(path == "0" and helper =="yes"):
       print(">>>>> fetchLLM")

    #    response = llm.fetchLLM(original_query)
       response = ollama.fetchLLM(original_query)
       print(response)

       start = timeit.default_timer()
       return jsonify({"data" : response, "timer" : timeit.default_timer() - start})
    else:
       print("direct prediction")
       return prediction.predict(query, path)
       


    # return content
    # return prediction.predict("Live chicken", "0")
    # return prediction.predict(content['q'], content['path'])

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=3004)