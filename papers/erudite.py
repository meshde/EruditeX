class EuditeX(object):

    def __init__(self):
        self.file = ''
        self.context = []
        self.query = ''

    def get_query(self, query):
        
        self.query = query
        print(self.query)

        # Filter top 5 paras using Info Retrieval
        para_select = infoRX.retrieve_info(self.context, self.query)
        para_sents = []
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        print(type(para_select[0]), para_select[0])

        for para in para_select:
            para_sents.extend(tokenizer.tokenize(para[0]))

        print('Sentences selected by IR Module:')
        print(para_sents)

        try:
            # Select Ans Sents - ABCNN
            ans_sents = abcnn.ans_select(query, para_sents)

            print('\nSystem: Sentences scored by Sentence Selection Module')
            for sentence,score in ans_sents:
                print('{0:50}\t{1}'.format(sentence, score[0]))
            print('')

            best_ans, score, answers = deploy.extract_answer_from_sentences(
                ans_sents,
                query,
                verbose=True,
            )

        except Exception as e:

            return {'answers': [{'word': 'ERROR', 'score': str(e)}]}


        # Ignore: Phase 2-3: Input Module and Answer Module
        # answers = []
        # for ans, a_score in ans_sents.iteritems():
        #   words = deploy.extract_answer_from_sentence(ans, self.query)
        #   words = sorted(words, key=operator.itemgetter(1))
        #   for word, w_score in words.iteritems()[:5]:
        #       answers.append((word, w_score * a_score))
        # answers = sorted(answers, key=operator.itemgetter(1))
        # proc = subprocess.Popen(['python','test.py',query],shell=False,stdout=subprocess.PIPE)

        ans_list = []
        for x in answers[:5]:
            ans_list.append({'word':x[0], 'score': float(x[1][0])})

        ans_dict = {'answers': ans_list}

        print('\nSystem: Candidate answers scored by Answer Extraction Module')
        for answer in ans_list:
            print('{0:10}\t{1}'.format(answer['word'], answer['score']))


        return ans_dict
