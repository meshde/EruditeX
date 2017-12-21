import numpy as np
import sys
import os
sys.path.append(os.path.abspath('./'))
# print(sys.path)
from Helpers.utils import load_glove
from Helpers.utils import get_vector
import sys
def centroid(wordvecs):
	vec = np.array(wordvecs).reshape((-1,50))
	return vec.mean(axis=0)
def get_word_vecs(line,glove):
	l = []
	for word in line.split():
		l.append(get_vector(word.lower(),glove))
	return l
def magnitude(vector):
	return np.sqrt(vector.dot(vector))
def cosine_similarity(A,B):
	return np.dot(A,B)/(magnitude(A)*magnitude(B))
def get_most_relevant(paras,query_measure):
	res = paras[0]
	mini = cosine_similarity(query_measure,paras[0][1])
	# print("Para",0," Measure:",mini)
	index = 0
	for i in range(1,len(paras)):
		similarity = cosine_similarity(query_measure,paras[i][1])
		print("Para",i," Measure:",similarity)
		if similarity > mini:
			mini = similarity
			index = i
			res = paras[i]
	# print("Para",index," Measure:",mini)
	return res[0]
def main():
	glove = load_glove()
	vector = []
	with open("data/corpus/cricket.txt",'r') as f:
		for line in f:
			line = line.strip()
			# print(line)
			l = get_word_vecs(line,glove)
			measure = centroid(l)
			vector.append((line,measure))
	query = sys.argv[1]
	query_measure = centroid(get_word_vecs(query,glove))
	print(get_most_relevant(vector,query_measure))
if __name__ == '__main__':
	main()




