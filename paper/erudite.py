import sys
sys.path.append('../')
from tqdm  import tqdm
from paper.babi import get_babi
from IR import infoRX
from Models import abcnn_model
from Helpers import deployment_utils as deploy
import nltk
import datetime
import logging


class EruditeX(object):

    def __init__(self):
        self.file = ''
        self.context = []
        self.query = ''
        self.abcnn = abcnn_model(verbosity=False)

    def get_babi_task_num(self, babi_task_num=1):
        babi_data_dict = get_babi(str(int(babi_task_num)))
        count = 0
        total = 0
        dataset_size = len(babi_data_dict)

        logs_file = "paper/logs/task_{0}_{1}".format(
            babi_task_num,
            str(datetime.datetime.now()),
        )
        logging.basicConfig(
            filename=logs_file,
            level=logging.INFO,
        )

        for element in tqdm(babi_data_dict, total=dataset_size, unit=' Question', ncols=75):
            total += 1
            self.context = element['context']
            actual_answer = element['ans_token']
            q = element['question']
            

            ans_dict = self.get_query(q)

            # print(ans_dict)

            ans_list = ans_dict['answers']
            predicted_answer = ans_list[0]['word']

            # print('predicted_answer:', predicted_answer)
            # print('actual_answer:', actual_answer)

            if(predicted_answer == actual_answer):
                count += 1

            logging.info("Example %d:\n"%(total))
            logging.info("Context:\n"+self.context+"\n")
            logging.info("Question: "+q+"\n")
            logging.info("Actual Answer: "+actual_answer+"\n")
            logging.info("Predicted Answer: "+predicted_answer+"\n")
            logging.info("\nTotal correct predictions as of now:%d\n"%(count))

        accuracy = (count/dataset_size)*100
        print("\n Accuracy after testing on bAbI task %d is %f"%(babi_task_num,accuracy))

        logging.info("Accuracy after testing on bAbI task %d is %f"%(babi_task_num,accuracy))



    def get_query(self, query):
        
        self.query = query
        # print('\n' + self.query)
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # Filter top 5 paras using Info Retrieval
        para_select = infoRX.retrieve_info(tokenizer.tokenize(self.context), self.query)
        para_sents = []

        # print(self.context)
        # print(type(para_select[0]), para_select[0])

        for para in para_select:
            para_sents.extend(tokenizer.tokenize(para[0]))

        # print('Sentences selected by IR Module:')
        # print(para_sents)

        # try:
        #     # Select Ans Sents - ABCNN
        #     ans_sents = abcnn.ans_select(query, para_sents)

        #     print('\nSystem: Sentences scored by Sentence Selection Module')
        #     for sentence,score in ans_sents:
        #         print('{0:50}\t{1}'.format(sentence, score[0]))
        #     print('')

        #     best_ans, score, answers = deploy.extract_answer_from_sentences(
        #         ans_sents,
        #         query,
        #         verbose=True,
        #     )

        # except Exception as e:

        #     return {'answers': [{'word': 'ERROR', 'score': str(e)}]}

        ans_sents = self.abcnn.ans_select(query, para_sents)

        # print('\nSystem: Sentences scored by Sentence Selection Module')
        # for sentence,score in ans_sents:
            # print('{0:50}\t{1}'.format(sentence, score[0]))
        # print('')

        best_ans, score, answers = deploy.extract_answer_from_sentences(
            ans_sents,
            query,
            verbose=False,
        )

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

        # print('\nSystem: Candidate answers scored by Answer Extraction Module')
        # for answer in ans_list:
            # print('{0:10}\t{1}'.format(answer['word'], answer['score']))


        return ans_dict


if __name__ == '__main__':
    test_obj = EruditeX()
    test_obj.get_babi_task_num()
