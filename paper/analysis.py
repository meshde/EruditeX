from paper import babi
from Helpers import path_utils
import os

def analyse(path, vocab, search = 'Actual Answer'):
    flag = False
    with open(path, 'r') as f:
        for line in f:
            if search in line:
                flag = True
                answer = line.split(':')[-1]
                answer = answer.strip()
                if search == 'Actual Answer':
                    assert answer in vocab
                try:
                    vocab[answer] += 1
                except:
                    vocab[answer] = 1
        assert flag
    return

def task1(search='Actual Answer'):
    BASE = path_utils.get_base_path()
    path = os.path.join(
        BASE,
        'paper/logs/task_1',
    )

    if search == 'Actual Answer':
        vocab = babi.get_vocab()
    else:
        vocab = {}

    for filename in os.listdir(path):
        if not filename.endswith('.swp'):
            filepath = os.path.join(path, filename)
            analyse(filepath, vocab, search)

    for key in vocab:
        print(key)

    for key in vocab:
        print(vocab[key])
    return

