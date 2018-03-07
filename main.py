def preprocessSick():
    from Helpers.preprocess import SICK
    SICK.get_final_input()
    return

def train_dtrnn():

    from Model_Trainer.dtrnnTrain import DT_RNN_Train
    model = DT_RNN_Train(n=8000, epochs=5, hid_dim=50)
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
    from Helpers.delpoyment_utils import create_config
    create_config(state_file_name, config_file_name)
    return

if __name__ == '__main__':
    train_dtrnn()


