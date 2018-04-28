from Helpers import utils
from Helpers import path_utils
from Models import abcnn_ass
import os
import operator
import sys
from tqdm import tqdm

def get_config(filename):
    filepath = path_utils.get_config_file_path(filename)

    config = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()

                if value.isdigit():
                    value = int(value)

                config[key] = value
    except FileNotFoundError:
        raise FileNotFoundError(
            '{0} has not been created yet!'.format(
            filename,
            ),
        )
    except ValueError:
        raise ValueError(
            'The contents of file {0} are not in proper format\n'
            'Format: key=value'.format(
                filename,
            ),
        )
    return config

def create_config(state_file_name, config_file_name):
    config = {}
    state_file_name = state_file_name.strip('.pkl')

    for param in state_file_name.split('__'):
        try:
            key,value = param.split('.')
        except:
            key,value = param.split('.', maxsplit=1)
        config[key] = value
       
    config['state'] = state_file_name + '.pkl'

    config_file_path = path_utils.get_config_file_path(config_file_name)

    with open(config_file_path, 'w') as f:
        for key,value in sorted(config.items()):
            f.write('{}={}'.format(key, value))
            f.write('\n')
    return

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

def _extract_answer_from_sentence(sentence, question, nlp, config,
                                  verbose=False):
    # check_configurations()
    # config = get_config('dtrnn.cfg')

    if verbose:
        print('SpaCy: Generating Dependency Tree for ["{0}"]'.format(sentence))
    sentence_tree = utils.get_dtree(sentence, nlp, dim=config['word_vector_size'])

    if verbose:
        print('SpaCy: Generating Dependency Tree for ["{0}"]'.format(question))
    question_tree = utils.get_dtree(question, nlp, dim=config['word_vector_size'])

    sentence_text_traversal = sentence_tree.get_tree_traversal('text')

    temp = get_tree_hidden_states(
        sentence_tree,
        question_tree,
        config,
        verbose,
    )

    sentence_hidden_states = temp[0]
    question_hidden_states = temp[1]

    sentence_tree.update_hidden_states(sentence_hidden_states)
    question_tree.update_hidden_states(question_hidden_states)

    answers = get_answer_nodes(sentence_tree, question_tree, verbose)
    answers = [(sentence_text_traversal[i], score) for i, score in answers]

    return answers

def extract_answer_from_sentences(sentences, question, verbose=False):
    check_configurations()
    config = get_config('dtrnn.cfg')

    if verbose:
        print('Spacy: Initializing...')
    import spacy
    nlp = spacy.load('en')

    ans_sent_list = []
    final_list = []
    for sent_score_tuple in sentences:
        sentence, score = sent_score_tuple
        node_scores = _extract_answer_from_sentence(
            sentence,
            question,
            nlp,
            config,
            verbose,
        )
        print('')

        for ns in node_scores:
            node, n_score = ns
            f_score = score * n_score
            final_list.append((node, f_score))

    ans_node, score = max(final_list, key=operator.itemgetter(1))
    final_list = sorted(final_list, key=operator.itemgetter(1), reverse=True) # Uncomment this if want list of all nodes with scores
    return ans_node, score, final_list


def get_dtrnn_model(config):
    from Models import DT_RNN
    model = DT_RNN(
        dep_len = config['dep_len'],
        dim = config['dim'],
        word_vector_size = config['word_vector_size']
    )
    model.theano_build()
    model.load_params(config['state'])
    return model

def get_tree_hidden_states(sentence_tree, question_tree, config,
                           verbose=False):

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

    if verbose:
        print('Input Module: Initializing...')
    model = get_dtrnn_model(config)

    if verbose:
        print('Input Module: Genrating VDT for sentence...')
    sentence_hidden_states = model.get_hidden_states(
        sentence_word_vectors,
        sentence_parent_indices,
        sentence_is_leaf,
        sentence_dep_tags
    )

    if verbose:
        print('Input Module: Genrating VDT for question...')
    question_hidden_states = model.get_hidden_states(
        question_word_vectors,
        question_parent_indices,
        question_is_leaf,
        question_dep_tags
    )

    return sentence_hidden_states, question_hidden_states

def get_answer_extraction_model(config):
    from Models import AnsSelect
    model = AnsSelect(
        inp_dim = config['inp_dim'],
        hid_dim = config['hid_dim']
    )
    model.load_params(config['state'])
    return model

def get_answer_nodes(sentence_tree, question_tree, verbose=False):
    sentence_root = sentence_tree.get_root_hidden_state() 
    question_root = question_tree.get_root_hidden_state()

    answer_nodes = []

    if verbose:
        print('Extraction Module: Initializing...')
    config = get_config('ans_select.cfg')
    model = get_answer_extraction_model(config)

    parent_indices = sentence_tree.get_tree_traversal('parent_index')
    tree_traversal = sentence_tree.postorder()

    if verbose:
        print('Extraction Module: Scoring Answer Nodes...')

    def loop(i, node):
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
        return

    if verbose:
        for i,node in tqdm(
            enumerate(tree_traversal),
            total=len(tree_traversal),
            unit='node'
        ):
            loop(i,node)
    else:
        for i,node in enumerate(tree_traversal):
            loop(i,node)

    return answer_nodes
