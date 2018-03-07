
def test_dt_rnn():
    import numpy as np
    from Models import DT_RNN
    from Models import np_dt_rnn
    from Helpers import utils

    model = DT_RNN(dim=3, word_vector_size=3)

    np_W_dep = model.W_dep.get_value()
    np_W_x = model.W_x.get_value()
    np_b = model.b.get_value()

    sentence = "welcome to my house"
    dtree = utils.get_dtree(sentence, dim=3)
    vectors,parent_indices,is_leaf,dep_tags = dtree.get_rnn_input()

    np_ans = np_dt_rnn(vectors, parent_indices, is_leaf, dep_tags, np_W_x, np_W_dep, np_b)
    theano_ans = model.get_hidden_states(vectors, parent_indices, is_leaf, dep_tags)

    print(np_ans)
    print(theano_ans)

    assert(np.allclose(np_ans, theano_ans, rtol=1e-04, atol=1e-07))
    return


def test_sick_preprocess():
    from Helpers.preprocess import SICK
    from Helpers import utils

    import spacy
    nlp = spacy.load('en')

    sick = SICK.get_data()
    glove = utils.load_glove(200)

    data = sick[0]
    assert('senetnce_A' not in data['A'])

    dtree_entry, dtne_entry = SICK.get_input_tree_single(data, nlp, glove)

    for entry in [dtree_entry, dtne_entry]:
        for x in ['A', 'B', 'score']:
            assert(x in entry)

            if x != 'score':
                for y in ['word_vectors', 'parent_indices', 'is_leaf', 'dep_tags', 'text']:
                    assert(y in entry[x])

    assert('ent_type' in dtne_entry['A'])
    assert('ent_type' in dtne_entry['B'])
    return

def test_ans_select():
    from Models import AnsSelect
    import numpy as np

    for inp_dim in [30, 50, 100, 200, 300]:
        q = np.random.rand(inp_dim)
        ans_sent = np.random.rand(inp_dim)
        ans_node = np.random.rand(inp_dim)
        ans_parent = np.random.rand(inp_dim)
        answer = 1
        
        initializations = [
            'glorot_normal','glorot_uniform',
            'he_uniform','he_normal'
        ]

        for optimization in ['adadelta', 'adam']:
            for initialization in initializations:
                model = AnsSelect(inp_dim,
                                  optimization=optimization,
                                  initialization=initialization
                                 )
                model.train(q, ans_sent, ans_node, ans_parent, answer)
        # x = model.predict(q, ans_sent, ans_node, ans_parent)
        # x = model.get_loss(q, ans_sent, ans_node, ans_parent, answer)
        # raise AssertionError(x)
    return

def test_dtrnn_train():
    from Model_Trainer import DT_RNN_Train as dttrain
    from Helpers import utils
    
    initializations = [
        'glorot_normal','glorot_uniform',
        'he_uniform','he_normal'
    ]

    for optimization in ['adadelta', 'adam']:
        for initialization in initializations:
            sent1 = "this is my house"
            sent2 = "this is my home"
            score = 5

            inputs1 = utils.get_dtree(sent1).get_rnn_input()
            inputs2 = utils.get_dtree(sent2).get_rnn_input()

            model = dttrain(
                n=1, epochs=2, hid_dim=200,
                optimization=optimization,
                initialization=initialization)
            model.train(
                inputs1[0],
                inputs1[1],
                inputs1[2],
                inputs1[3],
                inputs2[0],
                inputs2[1],
                inputs2[2],
                inputs2[3],
                score
            )
    return

def test_get_state_file_name():
    from Helpers import utils

    filename = utils.get_state_file_name(
        extension = 'pkl',
        first_name = 'mehmood shakeel deshmukh',
        username = 'meshde',
        age = 22
    )

    required = 'age:22__first_name:mehmood_shakeel_deshmukh__username:meshde.pkl'
    assert(filename == required)
    return

def test_imports():
    import Helpers
    import Models
    import Model_Trainer
    return
