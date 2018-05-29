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

context = """
"""
question = ''

print('The Context:')
print(context)
print('The Question:',question)
paras, question = tokenise(context, question)

print('')

pca_glove = get_pca_glove(paras, question)

print('Preprocessing:')
for i,para in enumerate(paras):
    print('-> Paragraph no.',i)
    print('The paragraph was tokenised into the following sentences:')
    for j,sent in enumerate(para):
        print('---> Sentence no.',j)
        print('The sentence was tokenised into the following words:')
        print('| '.join(sen['words']))
        print('The associated word embbeddings are:')
        print(pca_glove(sent['vectors']))
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
ans_sents = abcnn.ans_select(query, para_sents)

print('\nSentence Ranking:')
for sentence,score in ans_sents:
    print('{0:50}\t{1}'.format(sentence, score[0]))
print('-'*100)

results  = extract_answer_from_sentences(
    ans_sents,
    query,
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
