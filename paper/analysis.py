from paper import babi
from Helpers import path_utils
import os

def analyse(path, vocab, search = 'Predicted Answer'):
    flag = False
    with open(path, 'r') as f:
        for line in f:
            if search in line:
                flag = True
                answer = line.split(':')[-1]
                answer = answer.strip()
                # if search == 'Actual Answer':
                #     assert answer in vocab
                try:
                    vocab[answer] += 1
                except:
                    vocab[answer] = 1
        assert flag
    return

def task(task_id, vocab, search='Predicted Answer'):
    BASE = path_utils.get_base_path()
    path = os.path.join(
        BASE,
        'paper/logs/task_{}'.format(task_id),
    )

    for filename in os.listdir(path):
        if not filename.endswith('.swp'):
            filepath = os.path.join(path, filename)
            analyse(filepath, vocab, search)

    return

