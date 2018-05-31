import numpy as np
import sys
import math
import os
import string
from nltk import word_tokenize
from nltk.corpus import stopwords

sys.path.append('../')
from Helpers import utils

def centroid(wordvecs):
    vec = np.array(wordvecs).reshape((-1, 200))
    return vec.mean(axis=0)


def get_word_vecs(line, glove):
    l = []
    for word in line.split():
        l.append(utils.get_vector(word.lower(), glove, dim=200))
    return l


def magnitude(vector):
    return np.sqrt(vector.dot(vector))


def cosine_similarity(A, B):
    return np.dot(A, B) / (magnitude(A) * magnitude(B))


def get_most_relevant(paras, query_measure):

    top_paras = []
    cosine_sim = []

#     print(paras)

    for i in range(0, len(paras)):
        cosine_sim.append(cosine_similarity(query_measure, paras[i][1]))
        # print("Para", i, " Measure:", similarity)

    top_indices = np.array(cosine_sim).argsort()[-5:][::-1]

#     print('Top Indices:', top_indices)
#     print('Cos:', cosine_sim)

    for index in top_indices:
        params = list(paras[index])
        # print(params)
        params.append(cosine_sim[index])
        top_paras.append(params)
        # print(top_paras)

    return top_paras


# TF-IDF Algorithm
def tf_idf(doc, query):
    stop = set(stopwords.words('english'))
    tfidf = []
    wcount_total = []
    doc_count_total = len(doc)
    imp_tokens = [i for i in word_tokenize(query.lower()) if i not in stop]

    for i in range(0, doc_count_total):
        tfidf.append(0)
        #para = " ".join(l for l in doc[i].split('\n'))
        wcount_total.append(len(doc[i].split()))
        # print(wcount_total[i])

    for term in imp_tokens:
        doc_freq = 0
        tf = []
        for i in range(0, doc_count_total):
            term_count = doc[i].lower().count(term)
            if term_count > 0:
                doc_freq += 1
            tf.append(term_count / wcount_total[i])
            # print(term, term_count, doc_freq)

        # doc_freq += 1e-7
        # doc_count_total += 1e-7

        # print(term, term_count)
        idf = math.log((doc_count_total + 1e-7) / (doc_freq + 1e-7))

        for i in range(0, doc_count_total):
            tfidf[i] += tf[i] * idf

        # print(tfidf)
    return tfidf, imp_tokens


def retrieve_info(doc, query):
    glove = utils.load_glove(dim=200)
    vector = []

    # query = sys.argv[1]
    # file_name = sys.argv[2]
    # print(tf_idf(doc, query))
    doc = doc.split('\n')

    tidf_measure = np.array(tf_idf(doc, query)[0])
    top_indices = tidf_measure.argsort()[-10:][::-1]
#   print(top_indices)

    for index in top_indices:
        para = doc[index]
        para_word_vec = get_word_vecs(para, glove)
        measure = centroid(para_word_vec)
        vector.append((para, measure))

    # print(vector)
    query_measure = centroid(get_word_vecs(query, glove))

    # print(get_most_relevant(vector, query_measure))
    return get_most_relevant(vector, query_measure)

def _retrieve_info(doc, query):
    glove = utils.load_glove(dim=200)
    vector = []
    ir_dict = {}

    doc = [x for x in doc.split('\n') if x != '']
    print(doc)

    tidf_measure = np.array(tf_idf(doc, query)[0])
    print(tidf_measure)
    top_indices = tidf_measure.argsort()[-10:][::-1]

    for index in top_indices:
        para = doc[index]
        print(para)
        print('-'*10)
        para_word_vec = get_word_vecs(para, glove)
        p_centr = centroid(para_word_vec)
        p_tfidf = tidf_measure[index]
        vector.append((para, p_centr, p_tfidf))

    # print(vector)
    query_measure = centroid(get_word_vecs(query, glove))

    # Ranked Paras - (Para, centroid, tfidf, cosine_sim)
    top_para_list = get_most_relevant(vector, query_measure)
    # print(get_most_relevant(vector, query_measure))
    top_para_dict = []
    for para in top_para_list:
        entry = {}
        entry['para'] = para[0]
        entry['centroid'] = para[1]
        entry['tf_idf'] = para[2]
        entry['cosine_sim'] = para[3]
        top_para_dict.append(entry)

    ir_dict['question'] = query
    ir_dict['q_centroid'] = query_measure
    ir_dict['top_paras'] = top_para_dict
    
    return ir_dict

if __name__ == '__main__':
    file_name = os.path.join("../data/corpus/cricket.txt")
    query = "what is the role of bat in cricket"

    with open(file_name, 'r') as f:
        doc = list(filter(('\n').__ne__, f.readlines()))

    print(retrieve_info(doc, query))
