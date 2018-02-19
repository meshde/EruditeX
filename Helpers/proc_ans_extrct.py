import xml.etree.ElementTree
import json
import pickle
import os

def _process_ans_extract_dataset(source):

    file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/ans_extraction/'+source+'.xml')
    e = xml.etree.ElementTree.parse(file).getroot()

    qalist = []

    for qapairs in e.findall('qapairs'):

        qa = {}
        for child in qapairs:
        
            if child.tag == 'question':
                # print(child.text)
                qa['qstn'] = child.text.split('?')[0].replace('\n', '').replace(' ', '').replace('\t', ' ')

            elif child.tag == 'positive':
                child
                qa['ans_sent'] = child.text.split('.')[0].replace('\n', '').replace(' ', '').replace('\t', ' ')
                qa['ans'] = child.text.split('\n')[-3].replace('\t', ' ')
                # qa[ans_phrase] = ''

        qalist.append(qa)
    return qalist

def _process_babi(source):

    file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/en/'+source+'.txt')
    qalist = []
    ans_sents = []
    qa = {}

    with open(file) as fp:
        data = fp.readlines()
        for line in data:
            if line.split(' ', 1)[0] == '1':
                # print(ans_sents)
                ans_sents = []

            ans_sents.append(line.split(' ', 1)[1][:-2])
            if '?' in line:
                line = line.split('\t')
                # print(line)
                qa['qstn'] = line[0].split(' ', 1)[1]
                qa['ans_sent'] = ans_sents[int(line[2].replace('\n',''))-1]
                qa['ans'] = line[1]
                qalist.append(qa)
                # print(qa)
                qa = {}

    return qalist

def json_write(qalist, source):
    file_path = \
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'data/ans_extraction/babi/json/'+source+'.json')
    with open(file_path, 'w') as fp:
        json.dump(qalist, fp)
    return
#       for qa in qalist:
#           json.dump(qa, fp)

def ds_pickle(qalist, source):
    file_path = \
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'data/ans_extraction/babi/pickle/'+source+'.pkl')
    
    with open(file_path, 'wb') as fp:
        pickle.dump(qalist, fp)
    return

def create_dataset(source):

    if source == 'ans_extraction':
        qalist = _process_ans_extract_dataset(source)
        title = source
        
    else:   
        qalist = _process_babi(source)
        title = source.split('_')
        title = 'babi'+'_'+title[0]+'_'+title[2]

    json_write(qalist, title)
    ds_pickle(qalist, title)

def get_ans_ext_list(source):

    file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/ans_extraction/'+source+'.pkl')

    if os.path.isfile(file):
        with open(file, 'rb') as fp:
            qalist = pickle.load(fp)
    
    else:
        qalist = _process_ans_extract_dataset()
        ds_pickle(qalist)

    return qalist

if __name__ == '__main__':
    
    create_dataset('ans_extraction')
    create_dataset('qa1_single-supporting-fact_train')
    create_dataset('qa4_two-arg-relations_test')
    create_dataset('qa5_three-arg-relations_test')
    create_dataset('qa6_yes-no-questions_test')
    create_dataset('qa9_simple-negation_test')
    create_dataset('qa10_indefinite-knowledge_test')
    create_dataset('qa12_conjunction_test')
    create_dataset('qa20_agents-motivations_test')

    # get_ans_ext_list()
