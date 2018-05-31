import sys
from os import path
from nltk import word_tokenize

def generate_vocab(file_path):
	if path.exists(file_path) == False:
		print('File does not exist')
		sys.exit()

	vocab_dict = {}
	vocab = []
	vocab_flag = -1
	with open (file_path, 'r') as f:
		ex_count = 0
		for line in f:
			if(line[0:8] == "Context:"):
				vocab_flag = 1
				continue
			elif(line[0:9] == "Question:"):
				vocab_flag = -1
				ex_count += 1
				vocab_dict[ex_count] = vocab
				vocab = []

			if(vocab_flag == 1):
				tokens = word_tokenize(line)
				for token in tokens:
					if not((token in vocab)):
						vocab.append(token)

	return vocab_dict


if __name__== "__main__":
	file_path = sys.argv[1]
	vocab_dict = generate_vocab(file_path)
	print(vocab_dict)

