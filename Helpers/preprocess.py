from . import path_utils
from . import utils
from . import deployment_utils as deploy

from tqdm import *
import os
import json
import pickle as pkl

class SICK:
    def get_data():
        PATH = path_utils.get_sick_path()

        with open(PATH,'r') as f:
            lines = [line for line in f]
            lines = lines[1:]

            dataset = []

            for line in lines:
                entry = dict()

                line = line.split('\t')

                entry['A'] = line[1]
                entry['B'] = line[2]
                entry['score'] = float(line[4])

                dataset.append(entry)

        return dataset

    def get_input_tree_single(data, nlp, glove):
        dtree_entry = dict()
        dtne_entry = dict()

        doc1 = nlp(data['A'])
        doc2 = nlp(data['B'])

        dtree_entry['A'] = get_single_sentence_input_dtree(doc1, glove)
        dtree_entry['B'] = get_single_sentence_input_dtree(doc2, glove)
        dtree_entry['score'] = data['score']

        dtne_entry['A'] = get_single_sentence_input_dtne(doc1, glove)
        dtne_entry['B'] = get_single_sentence_input_dtne(doc2, glove)
        dtne_entry['score'] = data['score']

        return dtree_entry, dtne_entry

    def get_input_tree():
        import spacy
        nlp = spacy.load('en')

        sick = SICK.get_data()
        glove = utils.load_glove(200)

        dataset_dtree = []
        dataset_dtne = []

        for i, data in tqdm(enumerate(sick), total=len(sick), unit='sent_pair'):
            try:
                dtree_entry, dtne_entry = SICK.get_input_tree_single(data, nlp, glove)
                dataset_dtree.append(dtree_entry)
                dataset_dtne.append(dtne_entry)
            except:
                print('Error in {}'.format(i))
                pass
        return dataset_dtree, dataset_dtne

    def get_final_input():
        from os import path as path
        import pickle as pkl
        file_path = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'data/cache/SICK_cache.pkl')
        dataset_dtree, dataset_dtne = get_final_input_from_path(
            file_path,
            SICK.get_input_tree
        )
        return dataset_dtree, dataset_dtne

def get_single_sentence_dtree(doc, glove):
    sent = utils.get_sentence_from_doc(doc)
    dtree = utils.get_tree_node(sent.root, glove, dim=200)
    return dtree

def get_input_from_dtree(dtree):
    dtree_entry = dict()
    word_vectors, parent_indices, is_leaf, dep_tags = dtree.get_rnn_input()

    dtree_entry['word_vectors'] = word_vectors
    dtree_entry['parent_indices'] = parent_indices
    dtree_entry['is_leaf'] = is_leaf
    dtree_entry['dep_tags'] = dep_tags
    dtree_entry['text'] = dtree.get_tree_traversal('text')

    return dtree_entry


def get_input_from_dtne(dtne):
    dtne_entry = dict()
    word_vectors, parent_indices, is_leaf, dep_tags, ent_type  = dtne.get_rnn_input()

    dtne_entry['word_vectors'] = word_vectors
    dtne_entry['parent_indices'] = parent_indices
    dtne_entry['is_leaf'] = is_leaf
    dtne_entry['dep_tags'] = dep_tags
    dtne_entry['ent_type'] = ent_type
    dtne_entry['text'] = dtne.get_tree_traversal('text')

    return dtne_entry

def get_single_sentence_input_dtree(doc, glove):
    dtree = get_single_sentence_dtree(doc, glove)
    dtree_entry = get_input_from_dtree(dtree)
    return dtree_entry

def get_single_sentence_input_dtne(doc, glove):
    dtne = get_single_sentence_dtne(doc, glove)
    dtne_entry = get_input_from_dtne(dtne)
    return dtne_entry 

def get_single_sentence_dtne(doc, glove):
    for ent in doc.ents:
        ent.merge()

    sent = utils.get_sentence_from_doc(doc)
    dtne = utils.get_dtne_node(sent.root, glove, dim=200)
    
    return dtne

def get_final_input_from_path(file_path, get_input_tree):
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            dataset_dtree, dataset_dtne = pkl.load(f)
        return dataset_dtree, dataset_dtne
    else:
        dataset_dtree, dataset_dtne = get_input_tree()
        with open(file_path, 'wb') as f:
            pkl.dump((dataset_dtree, dataset_dtne), f)
        return dataset_dtree, dataset_dtne

def get_ans_model_input_from_path(file_path, create_babi_dataset):
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            dataset_ = pkl.load(f)
        return dataset_
    else:
        dataset_ = create_babi_dataset()
        with open(file_path, 'wb') as f:
            pkl.dump(dataset_, f)
        return dataset_

class AnswerExtract(object):
    def get_qa_pairs(data):
        if data == 'babi':
            qa_pairs = []
            dir_path = path_utils.get_babi_ans_extract_path()
            for item in sorted(os.listdir(dir_path)):
                file_path = os.path.join(dir_path, item)
                with open(file_path, 'r') as f:
                    pairs = json.load(f)
                    qa_pairs.extend(pairs)
            return qa_pairs
        raise NotImplementedError("AnswerExtract.get_qa_pairs has not been \
                                  implemented for data value as {}".format(data))

    def get_input_tree(data):
        import spacy
        nlp = spacy.load('en')
        glove = utils.load_glove(200)

        qa_pairs = AnswerExtract.get_qa_pairs(data)
        dataset_dtree, dataset_dtne = [], []
        
        for i,pair in tqdm(enumerate(qa_pairs), total=len(qa_pairs), unit='qa_pairs'):
            dtree_entry, dtne_entry = AnswerExtract.get_input_tree_single(pair, nlp, glove)
            dataset_dtree.append(dtree_entry)
            dataset_dtne.append(dtne_entry)
        return dataset_dtree, dataset_dtne

    def get_input_tree_single(data, nlp, glove):
        dtree_entry = dict()
        dtne_entry = dict()

        doc1 = nlp(data['qstn'])
        doc2 = nlp(data['ans_sent'])

        dtree_entry['qstn'] = get_single_sentence_input_dtree(doc1, glove)
        ans_tree = get_single_sentence_dtree(doc2, glove)
        dtree_entry['ans'] = ans_tree.get_node(data['ans'])
        dtree_entry['ans_sent'] = get_input_from_dtree(ans_tree)

        dtne_entry['qstn'] = get_single_sentence_input_dtne(doc1, glove)
        ans_tree = get_single_sentence_dtne(doc2, glove)
        dtne_entry['ans'] = ans_tree.get_node(data['ans'])
        dtne_entry['ans_sent'] = get_input_from_dtne(ans_tree)
        return dtree_entry, dtne_entry

    def get_final_input_babi():
        file_path = path_utils.get_babi_ans_extract_input_path()
        dataset_dtree, dataset_dtne = get_final_input_from_path(
            file_path,
            lambda:AnswerExtract.get_input_tree(data='babi')
        )
        return dataset_dtree, dataset_dtne


    def get_ans_model_input_babi():
        file_path = path_utils.get_babi_ans_mod_path()
        dataset_ = get_ans_model_input_from_path(file_path, lambda: AnswerExtract.create_ans_mod_babi_dataset())
        return dataset_

    def create_ans_mod_babi_dataset():

        dataset_ = AnswerExtract.get_final_input_babi()[0]
        ans_mod_dataset = []

        config_ = deploy.get_config('dtrnn.cfg')
        dtrnn_model = deploy.get_dtrnn_model(config_)

        for x in dataset_dtree:

            amd_entry = {}
            q_node = x['qstn']
            a_node = x['ans_sent']

            a_index = x['ans']
            parent_index = a_node['parent_indices'][a_index]
            
            hid_states = dtrnn_model.get_hidden_states(
                q_node['word_vectors'], 
                q_node['parent_indices'], 
                q_node['is_leaf'],
                q_node['dep_tags'])

            amd_entry['qstn_root'] = hid_states['qstn'][-1] 
            
            hid_states = dtrnn_model.get_hidden_states(
                a_node['word_vectors'], 
                a_node['parent_indices'], 
                a_node['is_leaf'], 
                a_node['dep_tags'])

            hid_ans = hid_states['ans_sent']

            amd_entry['ans_root'] = hid_ans[-1]
            amd_entry['ans_node'] = hid_ans[a_index]
            amd_entry['ans_parent'] = hid_ans[parent_index]
            
            ans_mod_dataset.append(amd_entry)
            # print(x['ans'], parent_index)

        return ans_mod_dataset


if __name__ == '__main__':
    # SICK.get_final_input()
    AnswerExtract.create_ans_mod_babi_dataset()
