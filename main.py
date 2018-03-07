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

def install_packages():
    from subprocess import call

    with open('requirements.txt','r') as f:
        requirements = [line.strip() for line in f.readlines()]
        for requirement in requirements:
            return_code = call("pip install " + requirement, shell=True)

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

                    lasagne_req_link = "https://raw.githubusercontent.com/\
                        Lasagne/Lasagne/master/requirements.txt"
                    lasagne_link = \
                        "https://github.com/Lasagne/Lasagne/archive/master.zip"

                    call(
                        "pip install -r {0};pip install {1}".format(
                            lasagne_req_link,
                            lasagne_link
                        ),
                        shell=True
                    )

    call("python -m spacy download en", shell=True)
    
    print("The following packages could not be installed:")
    call("pip freeze | diff requirements.txt - | grep '^<' | sed 's/^<\ //'", shell=True)
    return

if __name__ == '__main__':
    train_dtrnn()


