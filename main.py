def preprocessSick():
    from Helpers.preprocess import SICK
    SICK.get_final_input()
    return

def train_dtrnn():

    from Model_Trainer.dtrnnTrain import DT_RNN_Train
    model = DT_RNN_Train(n=8000, epochs=10, hid_dim=50)
    model.train_dtRNN()

def train_dtrnn_debug():
    from Model_Trainer.dtrnnTrain import DT_RNN_Train
    model = DT_RNN_Train("load", n=3, epochs=5, hid_dim=50)
    x = model.toposort()
    index = 189
    print(x[index])
    print(1000 * "-")
    print(x[index].inputs)
    print(1000 * "-")
    print([inp.owner for inp in x[index].inputs])
    print(1000 * "-")
    print(x[index].outputs)
    print(1000 * "-")
    print([out.owner for out in x[index].outputs])

def train_dtrnn_graph():
    from Model_Trainer.dtrnnTrain import DT_RNN_Train
    import theano

    model = DT_RNN_Train(n=8000, epochs=5, hid_dim=50)
    print(theano.printing.debugprint(model.grad))
    return

def preprocess_babi_ans_extract():
    from Helpers import create_dataset 
    sources = ['qa1_single-supporting-fact_train',
             'qa4_two-arg-relations_test', 'qa5_three-arg-relations_test',
             'qa12_conjunction_test']

    for source in sources:
        create_dataset(source)
    return

def get_babi_tree():
    from Helpers import preprocess
    preprocess.AnswerExtract.get_final_input_babi()
    return

def create_config(state_file_name, config_file_name):
    from Helpers.deployment_utils import create_config
    import os

    state_file_name = os.path.basename(state_file_name)
    create_config(state_file_name, config_file_name)
    return

def install_packages():
    from subprocess import call

    with open('requirements.txt','r') as f:
        requirements = [line.strip() for line in f.readlines()]
        for requirement in requirements:
            return_code = call("pip install {} > /dev/null".format(requirement), shell=True)

            if return_code:
                if 'tensorflow-gpu' in requirement:
                    print("There was an error installing tensorflow-gpu!")
                    print("It is assumed that it was a MemoryError!")
                    print("The developers need to think of a better way to \
                          detect MemoryError")
                    print("Trying to solve MemoryError")
                    call("pip install --no-cache-dir " + requirement,
                         shell=True)

                if 'Lasagne' in requirement:
                    print("Lasagne 0.2.dev1 could not be found on PyPI")
                    print("Installing from the GitHub repository...")

                    lasagne_req_link = "https://raw.githubusercontent.com/"\
                        "Lasagne/Lasagne/master/requirements.txt"
                    lasagne_link = \
                        "https://github.com/Lasagne/Lasagne/archive/master.zip"

                    call(
                        "pip install -r {0};pip install {1}".format(
                            lasagne_req_link,
                            lasagne_link
                        ),
                        shell=True
                    )
            else:
                print("Installed: {}".format(requirement))

    call("python -m spacy download en", shell=True)
    
    print("The following packages could not be installed:")
    call("pip freeze | diff requirements.txt - | grep '^<' | sed 's/^<\ //'", shell=True)
    return


def create_ans_ext_input():
    from Helpers.preprocess import AnswerExtract
    AnswerExtract.get_ans_model_input_babi()
    return


def train_ans_extract(inp_dim=50, hid_dim=200, epochs=10):
    from Model_Trainer import train_extraction_module
    train_extraction_module(
        inp_dim=int(inp_dim),
        hid_dim=int(hid_dim),
        epochs=int(epochs),
    )
    return


def get_output():
    print('System: Initializing...')
    from Models import abcnn_model
    from Helpers.deployment_utils import extract_answer_from_sentences

    sents = [
        'john went to the bathroom',
        'mary went to the kitchen',
        'john moved to the hallway',
        'kim journeyed to the garden',
        'sandra is in the bedroom',
    ]
    print('\nSystem: The context is:')
    for sentence in sents:
        print(sentence)

    query = 'where is john'
    print('System: The question is:')
    print(query)



    # Select Ans Sents - ABCNN
    print('\nSentence Selection Module: Initializing...')
    abcnn = abcnn_model()
    ans_sents = abcnn.ans_select(query, sents)

    print('\nSystem: Sentences scored by Sentence Selection Module')
    for sentence,score in ans_sents:
        print('{0:50}\t{1}'.format(sentence, score[0]))
    print('')

    best_ans, score, answers = extract_answer_from_sentences(
        ans_sents,
        query,
        verbose=True,
    )

    ans_list = []
    for x in answers:
        ans_list.append({'word':x[0], 'score': x[1]})

    print('\nSystem: Candidate answers scored by Answer Extraction Module')
    for answer in ans_list:
        print('{0:10}\t{1}'.format(answer['word'], answer['score']))


def get_answer():
    print('System: Initializing...')
    import spacy
    from Helpers import deployment_utils as deploy

    sentence = 'john went to the bathroom'
    question = 'where is john'

    config = deploy.get_config('dtrnn.cfg')
    print('System: Loading SpaCy...')
    nlp = spacy.load('en')
    print('System: SpaCy Loaded')

    print('System: Sentence- ', sentence)
    print('System: Question-', question)
    scores = deploy._extract_answer_from_sentence(
        sentence,
        question,
        nlp,
        config,
        verbose=True,
    )

    print(scores)

    question = 'who is in the bathroom'

    print('System: Sentence-', sentence)
    print('System: Question-', question)
    scores = deploy._extract_answer_from_sentence(
        sentence,
        question,
        nlp,
        config,
        verbose=True,
    )

    print(scores)

    return

def paper(task_num=1):
    task_num = int(task_num)
    assert task_num in [1, 4, 5, 12]

    from paper import erudite
    erudite.EruditeX().get_babi_task_num(task_num)

if __name__ == '__main__':
    train_dtrnn()


