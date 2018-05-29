from IR import infoRX
from Helpers import utils
from Helpers import deployment_utils as deploy
from Models import abcnn_model
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.texttiling import TextTilingTokenizer
import numpy as np
from sklearn.decomposition import PCA

def vis_tokenize(context, question):

    glove = utils.load_glove(dim=200)

    ttt = TextTilingTokenizer()

    para_list = []
    paras = [para for para in context.split('\\n') if para != '' ]
    for para in paras:
        sent_list = []
        for sent in sent_tokenize(para):
            temp = {}
            temp['words'] = word_tokenize(sent)
            temp['vectors'] = [np.array(glove[word.lower()]) for word in temp['words']]
            sent_list.append(temp)
        para_list.append(sent_list)

    q_dict = {}
    q_dict['words'] = word_tokenize(question)
    q_dict['vectors'] = [np.array(glove[word.lower()]) for word in q_dict['words']]
    return para_list, q_dict

def get_pca_glove(paras, question):
    vectors = []
    for para in paras:
        for sentence in para:
            vectors.extend(sentence['vectors'])
    vectors.extend(question['vectors'])
    vectors = np.array(vectors).reshape(-1,200)
    pca = PCA(n_components=3)
    pca.fit(vectors)
    return pca

def get_pca_hidden(hidden_states):
    vectors = np.array(hidden_states).reshape(-1, 50)
    pca = PCA(n_components=3)
    pca.fit(vectors)
    return pca

with open('vis.txt', 'r') as f:
    context = f.read()
question = 'Where is John?'

# print(para_list, q_dict)

print('The Context:')
print(context)
print('The Question:',question)
paras, quest = vis_tokenize(context, question)

print('')

pca_glove = get_pca_glove(paras, quest)

print('Preprocessing:')
for i,para in enumerate(paras):
    print('-> Paragraph no.',i)
    print('The paragraph was tokenised into the following sentences:')
    for j,sent in enumerate(para):
        print('---> Sentence no.',j)
        print('The sentence was tokenised into the following words:')
        print('| '.join(sent['words']))
        print('The associated word embbeddings are:')
        print(pca_glove.transform(sent['vectors']))
print('Preprcoessing the question:')
print('| '.join(quest['words']))
print(pca_glove.transform(quest['vectors']))
print('-'*100)

print('Passage Retrieval:')

para_select = infoRX._retrieve_info(context, question)
para_sents = []
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

print(para_select)

# print(type(para_select[0]), para_select[0])

for para in para_select:
    para_sents.extend(tokenizer.tokenize(para[0]))

print('Sentences selected by IR Module:')
print(para_sents)
print('-'*100)

# Select Ans Sents - ABCNN
abcnn = abcnn_model()
ans_sents = abcnn.ans_select(question, para_sents)

print('\nSentence Ranking:')
for sentence,score in ans_sents:
    print('{0:50}\t{1}'.format(sentence, score[0]))
print('-'*100)

results  = deploy.extract_answer_from_sentences(
    ans_sents,
    question,
    vis=True,
)
best_ans, score, answers, tree_dict, hidden_states = results

pca_hidden = get_pca_hidden(hidden_states)

print('VDT Generation:')
for tree in tree_list:
    print('Statement:', tree['sentence'])
    print('Tree:')
    tree['tree'].print(pca_glove, pca_hidden)
print('-'*100)

ans_list = []
for x in answers[:5]:
    ans_list.append({'word':x[0], 'score': float(x[1][0])})

print('\nCandidate answers scored by Answer Extraction Module')
for answer in ans_list:
    print('{0:10}\t{1}'.format(answer['word'], answer['score']))
