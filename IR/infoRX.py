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
	vec = np.array(wordvecs).reshape((-1, 50))
	return vec.mean(axis=0)


def get_word_vecs(line, glove):
	l = []
	for word in line.split():
		l.append(utils.get_vector(word.lower(), glove))
	return l


def magnitude(vector):
	return np.sqrt(vector.dot(vector))


def cosine_similarity(A, B):
	return np.dot(A, B) / (magnitude(A) * magnitude(B))


def get_most_relevant(paras, query_measure):
	res = paras[0]
	mini = cosine_similarity(query_measure, paras[0][1])
	print("Para", 0, " Measure:", mini)
	index = 0
	for i in range(1, len(paras)):
		similarity = cosine_similarity(query_measure, paras[i][1])
		print("Para", i, " Measure:", similarity)
		if similarity > mini:
			mini = similarity
			index = i
			res = paras[i]
	print("Para", index, " Measure:", mini)
	return res[0]


# TF-IDF Algorithm
def tf_idf(doc, query):
	stop = set(stopwords.words('english'))
	tfidf = []
	wcount_total = []
	doc_count_total = len(doc)
	imp_tokens = [i for i in word_tokenize(query.lower()) if i not in stop]

	for i in range(0, doc_count_total):
		tfidf.append(0)
		para = " ".join(l for l in doc[i] if l not in string.punctuation)
		wcount_total.append(len(para.split()))
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

		doc_freq += 1e-7
		doc_count_total += 1e-7

		# print(term, term_count)
		idf = math.log(doc_count_total / doc_freq)
		
		for i in range(0, int(doc_count_total)):
			tfidf[i] += tf[i] * idf

		# print(tfidf)
		return tfidf, imp_tokens


def main():
	glove = utils.load_glove()
	vector = []

	# query = sys.argv[1]
	# file_name = sys.argv[2]

	file_name = "../data/corpus/cricket.txt"
	query = "what is the role of bat in cricket"

	with open(file_name, 'r') as f:
		doc = list(filter(('\n').__ne__, f.readlines()))
		tidf_measure = np.array(tf_idf(doc, query))
		top_indices = tidf_measure.argsort()[-5:][::-1]

	for index in top_indices:
		para = doc[i]
		para_word_vec = get_word_vecs(para, glove)
		measure = centroid(para_word_vec)
		vector.append((para, measure))

	query_measure = centroid(get_word_vecs(query, glove))
	print(get_most_relevant(vector, query_measure))


if __name__ == '__main__':
	main()
