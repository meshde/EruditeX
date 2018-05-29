from Helpers import utils
from Helpers import path_utils
from Models import abcnn_ass
import os
import operator
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
    except:
        raise FileNotFoundError(
            '{0} has not been created yet!'.format(
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

def _extract_answer_from_sentence(sentence, question_tree, nlp, config,
                                  verbose=False):
    # check_configurations()
    # config = get_config('dtrnn.cfg')

    if verbose:
        print('SpaCy: Generating Dependency Tree for ["{0}"]'.format(sentence))
    sentence_tree = utils.get_dtree(sentence, nlp, dim=config['word_vector_size'])


    sentence_text_traversal = sentence_tree.get_tree_traversal('text')

    sentence_hidden_states = get_tree_hidden_states(
        sentence_tree,
        config,
        verbose,
    )


    sentence_tree.update_hidden_states(sentence_hidden_states)

    answers = get_answer_nodes(sentence_tree, question_tree, verbose)
    answers = [(sentence_text_traversal[i], score) for i, score in answers]

    return answers

def _extract_answer_from_sentence_vis(sentence, question_tree, nlp, config,
                                  hidden_states):
    output = {}
    output['sentence'] = sentence
    sentence_tree = utils.get_dtree(sentence, nlp, dim=config['word_vector_size'])


    sentence_text_traversal = sentence_tree.get_tree_traversal('text')

    sentence_hidden_states = get_tree_hidden_states(
        sentence_tree,
        config,
        verbose,
    )

    hidden_states.extend(sentence_hidden_states)
    sentence_tree.update_hidden_states(sentence_hidden_states)

    answers = get_answer_nodes(sentence_tree, question_tree, verbose)
    answers = [(i, sentence_text_traversal[i], score) for i, score in answers]
    sentence_tree.update_node_scores(answers)
    output['tree'] = sentence_tree
    return answers,output
def extract_answer_from_sentences(sentences, question, verbose=False,
                                  vis=False):
    check_configurations()
    config = get_config('dtrnn.cfg')

    if verbose:
        print('Spacy: Initializing...')
    import spacy
    nlp = spacy.load('en')

    if verbose:
        print('SpaCy: Generating Dependency Tree for ["{0}"]'.format(question))
    question_tree = utils.get_dtree(question, nlp, dim=config['word_vector_size'])
    question_hidden_states = get_tree_hidden_states(
        question_tree,
        config,
        verbose,
    )
    question_tree.update_hidden_states(question_hidden_states)
    tree_list = [
        {
            'sentence': question,
            'tree': question_tree,
        },
    ]

    if vis:
       hidden_states.extend(question_hidden_states)

    ans_sent_list = []
    final_list = []
    for sent_score_tuple in sentences:
        sentence, score = sent_score_tuple
        if vis:
            node_scores, tree_dict = _extract_answer_from_sentence_vis(
                sentence,
                question_tree,
                nlp,
                config,
                hidden_states,
            )
            tree_list.append(tree_dict)
        else:
            node_scores = _extract_answer_from_sentence(
                sentence,
                question_tree,
                nlp,
                config,
                verbose,
            )
        if verbose: print('')

        for ns in node_scores:
            if vis:
                _, node, n_score = ns
            else:
                node, n_score = ns
            f_score = score * n_score
            final_list.append((node, f_score))

    ans_node, score = max(final_list, key=operator.itemgetter(1))
    final_list = sorted(final_list, key=operator.itemgetter(1), reverse=True) # Uncomment this if want list of all nodes with scores
    if vis:
        return ans_node, score, final_list, tree_list, hidden_states
    else:
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

def get_tree_hidden_states(tree, config, verbose=False):

    inputs = tree.get_rnn_input()

    word_vectors = inputs[0]
    parent_indices = inputs[1]
    is_leaf = inputs[2]
    dep_tags = inputs[3]

    if verbose:
        print('Input Module: Initializing...')
    model = get_dtrnn_model(config)

    if verbose:
        print('Input Module: Genrating VDT ...')
    hidden_states = model.get_hidden_states(
        word_vectors,
        parent_indices,
        is_leaf,
        dep_tags
    )


    return hidden_states

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
