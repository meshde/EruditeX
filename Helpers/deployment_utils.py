from Helpers import utils
from Helpers import path_utils
from Models import DT_RNN
import os

def get_config(filename):
    filepath = path_utils.get_config_file_path(filename)

    config = {}
    with open(fielpath, 'r') as f:
        for line in f:
            key,value = line.split('=')
            key = key.strip()
            value = value.strip()

            if value.isdigit():
                value = int(value)

            config[key] = value
    return config

def check_configurations():
    input_config = get_config('dtrnn.cfg')
    extraction_config = get_config('ans_select.cfg')

    if input_config['dim'] != extraction_config['inp_dim']:
        error_msg = "hidden_state dimensions of Input module (={0}) and "\
                "dimensions of input hidden_state of Extraction module (={1}) do not "\
                "match! (The output of Input module is fed as input to "\
                "the Extraction module)"
        
        error_msg = error_msg.format(
            input_config['dim'],
            extraction_config['inp_dim']
        )
        raise ValueError(error_msg)
    return

def extract_answer_from_sentence(sentence, question):
    check_configurations()
    config = get_config('dtrnn.cfg')
    
    sentence_tree = utils.get_dtree(sentence, dim=config['word_vector_size'])
    question_tree = utils.get_dtree(question, dim=config['word_vector_size'])
    sentence_text_traversal = sentence_tree.get_tree_taversal('text')

    temp = get_tree_hidden_states(sentence_tree, question_tree, config)
    sentence_hidden_states = temp[0]
    question_hidden_states = temp[1]

    sentence_tree.update_hidden_states(sentence_hidden_states)
    question_tree.update_hidden_states(question_hidden_states)

    answers = get_answer_nodes(sentence_tree, question_tree)
    answers = [(sentence_text_traversal[i], score) for i,score in answers]

    return answers

def get_dtrnn_model(config):
    model = DT_RNN(
        dep_len = config['dep_len'],
        dim = config['dim'],
        word_vector_size = config['word_vector_size']
    )
    model.theano_build()
    model.load_params(config['state'])
    return model

def get_tree_hidden_states(sentence_tree, question_tree, config):

    sentence_inputs = sentence_tree.get_rnn_input()
    question_inputs = question_tree.get_rnn_input()

    sentence_word_vectors = sentence_inputs[0]
    sentence_parent_indices = sentence_inputs[1]
    sentence_is_leaf = sentence_inputs[2]
    sentence_dep_tags = sentence_inputs[3]

    question_word_vectors = question_inputs[0]
    question_parent_indices = question_inputs[1]
    question_is_leaf = question_inputs[2]
    question_dep_tags = question_inputs[3]

    model = get_dtrnn_model(config)

    sentence_hidden_states = model.get_hidden_states(
        sentence_word_vectors,
        sentence_parent_indices,
        sentence_is_leaf,
        sentence_dep_tags
    )

    question_hidden_states = model.get_hidden_states(
        question_word_vectors,
        question_parent_indices,
        question_is_leaf,
        question_dep_tags
    )

    return sentence_hidden_states, question_hidden_states

def get_answer_extraction_model(config):
    model = AnsSelect(
        inp_dim = config['inp_dim'],
        hid_dim = config['hid_dim']
    )
    model.load_params(config['state'])
    return model

def get_answer_nodes(sentence_tree, question_tree):
    sentence_root = sentence_tree.get_root_hidden_state() 
    question_root = question_tree.get_root_hidden_state()

    answer_nodes = []

    config = get_config('ans_select.cfg')
    model = get_answer_extraction_model(config)

    parent_indices = sentence_tree.get_tree_traversal('parent_index')
    tree_traversal = sentence_tree.postorder()

    for i,node in enumerate(tree_traversal):
        parent_index = parent_indices[i]
        parent_node = tree_traversal[parent_index]
        parent_hidden_state = parent_node.get_hidden_state()

        score = model.predict(
            question_root,
            sentence_root,
            node.get_hidden_state(),
            parent_hidden_state
        )

        answer_nodes.append((i,score))

    return answer_nodes
