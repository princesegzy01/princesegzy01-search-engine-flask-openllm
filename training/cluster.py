from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import spacy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import spacy_universal_sentence_encoder



def processCluster(ws, query):

    modelx = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    
    print("Thank you")


    # Load the large english model
    nlp = spacy.load("en_core_web_lg")
    nlp_sentence_encoder = spacy_universal_sentence_encoder.load_model('en_use_lg')

    # nlp = spacy.load("en_core_web_sm")

    


    # word_list = "dog cat banana apple teaching teacher mom mother mama mommy berlin paris"
    # token_list = []

    # for w in word_list.split(" "):
    #     token_list.append(model.encode(w))
    # vectors = np.array(token_list)
    # print(np.array(token_list).shape)
    # return


    # ws = ["abo",
    #     "aspects",
    #     "bank",
    #     "banks",
    #     "blood",
    #     "care",
    #     "cases",
    #     "cells",
    #     "collection",
    #     "collects",
    #     "compatibility",
    #     "components",
    #     "control",
    #     "crossmatching",
    #     "days",
    #     "diseases",
    #     "distribution",
    #     "donating",
    #     "donations",
    #     "donor",
    #     "donors",
    #     "drink",
    #     "education",
    #     "ensure",
    #     "environment",
    #     "extend",
    #     "facilities",
    #     "facility",
    #     "factor",
    #     "fluids",
    #     "group",
    #     "healthcare",
    #     "hospitals",
    #     "hours",
    #     "information",
    #     "life",
    #     "lifethreatening",
    #     "meal",
    #     "measures",
    #     "members",
    #     "monitoring",
    #     "patient",
    #     "patients",
    #     "period",
    #     "phlebotomy",
    #     "place",
    #     "platelet",
    #     "platelets",
    #     "prepare",
    #     "preservatives",
    #     "prevent",
    #     "process",
    #     "processing",
    #     "products",
    #     "protocols",
    #     "purposes",
    #     "quality",
    #     "reaction",
    #     "recipients",
    #     "risk",
    #     "role",
    #     "safety",
    #     "selection",
    #     "series",
    #     "shelf",
    #     "solution",
    #     "spoilage",
    #     "staff",
    #     "storage",
    #     "stores",
    #     "supply",
    #     "system",
    #     "temperature",
    #     "testing",
    #     "tests",
    #     "time",
    #     "tracking",
    #     "training",
    #     "transfusion",
    #     "transfusions",
    #     "type",
    #     "typing",
    #     "volunteer"]
    
    # ws = ["bar", "bluray", "builtin", "button", "casing", "community", "console", "consoles", "control", "controller", "crossplatform", "custom", "design", "displays", "experience", "featurerich", "features", "feedback", "followers", "footage", "friends", "front", "gameplay", "gamers", "games", "gaming", "generation", "headset", "information", "launch", "light", "lighting", "match", "media", "motion", "options", "panel", "performance", "play", "player", "playstation", "processor", "ps", "range", "ray", "reality", "reflections", "resolution", "screenshots", "sensation", "sensors", "series", "share", "size", "specifications", "style", "technology", "titles", "tv", "types", "village", "visuals", "zen"]

    # noun_list = []
    doc = nlp(" ".join(ws))
    # noun_list = [chunk.text for chunk in doc.noun_chunks]

    noun_list = [chunk.lemma_ for chunk in doc if chunk.pos_ == 'NOUN']

    llm_list = list(set(noun_list))
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(ws)
    # return



    tokens = nlp(" ".join(llm_list))

    # # Generate word embedding vectors
    vectors = np.array([token.vector for token in tokens])
    print(vectors.shape)
    # # (12, 300)
    # print(vectors[0])



    pca_vecs = PCA(n_components=5).fit_transform(vectors)
    pca_vecs.shape
    # (12, 3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = pca_vecs[:, 0], pca_vecs[:, 1], pca_vecs[:, 2]
    _ = ax.scatter(xs, ys, zs)

    for x, y, z, lable in zip(xs, ys, zs, tokens):
        ax.text(x+0.3, y, z, str(lable))
    

    plt.show()


    # model = DBSCAN(eps=0.002, min_samples=1)
    # model.fit(vectors)

    # print(model.labels_)

    # word_map = {}
    # for word, cluster in zip(tokens, model.labels_):
    #     # print(word, '->', cluster)  
    #     word_map.update({word : cluster})

    # v = defaultdict(list)

    # for key, value in sorted(word_map.items()):
    #     v[value].append(key)


    doc_1 = nlp_sentence_encoder(query)

    word_smilarity = {}

    for key in llm_list:
        # phrase = " ".join(str(x) for x in v[key])
        doc_2 = nlp_sentence_encoder(key)

        similarity = doc_1.similarity(doc_2)
        word_smilarity.update({ key : similarity})
        
        # print(similarity, " -- ", key)

        # doc_2 = nlp(" ".join(v['0']))
        # use the similarity method that is based on the vectors, on Doc, Span or Token
        # print(doc_1.similarity(doc_2[0:7]))
    
    # print(word_smilarity)

    d = {key: value for key, value in word_smilarity.items() if value > 0.5}
    
    print(d)

    final = [k for k, v in d.items()]

    final = " ".join(final)

    print(final)


    return final




# processCluster()