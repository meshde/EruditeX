from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from Helpers.preprocess import AnswerExtract
from Models import AnsSelect
from Helpers import utils

import numpy as np
from tqdm import tqdm

def get_empty_y(test_size):
    y_true = np.random.rand(test_size)
    y_pred = np.random.rand(test_size)
    return y_true, y_pred

def get_prediction(score, threshold):
    if score >= threshold:
        return 1
    return 0

def train_normal_dataset(model, training_data):
    for item in tqdm(
        training_data,
        total=len(training_data),
        unit='sample',
        desc='Training/Epoch',
    ):
        model.train(
            item['question_root'],
            item['answer_root'],
            item['answer_node'],
            item['parent_node'],
            item['label']
        )
    return

def test_normal_dataset(model, testing_data, threshold):
    test_size = len(testing_data)
    y_true, y_pred = get_empty_y(test_size)

    for i,item in tqdm(
        enumerate(testing_data),
        total=test_size,
        unit='sample',
        desc='Testing/Epoch',
    ):
        score = model.predict(
            item['question_root'],
            item['answer_root'],
            item['answer_node'],
            item['parent_node'],
        )
        prediction = get_prediction(score, threshold)

        y_true[i] = item['label']
        y_pred[i] = prediction
    return y_true, y_pred

def train_compressed_dataset(model, training_data):
    for item in training_data:
        for answer_node in item['answer_nodes']:
            model.train(
                item['question_root'],
                item['answer_root'],
                answer_node['answer_node'],
                answer_node['parent_node'],
                answer_node['label']
            )
    return

def test_compressed_dataset(model, testing_data, threshold):
    test_size = get_data_length(testing_data)
    y_true, y_pred = get_empty_y(test_size)

    index = 0
    for item in testing_data:
        for answer_node in item['answer_nodes']:
            score = model.predict(
                item['question_root'],
                item['answer_root'],
                answer_node['answer_node'],
                answer_node['parent_node']
            )
            prediction = get_prediction(score, threshold)

            y_true[index] = answer_node['label']
            y_pred[index] = prediction

            index += 1
    return y_true, y_pred

def train_extraction_module(inp_dim, hid_dim, epochs,
    initialization='glorot_normal', optimization='adam', threshold=0.5,
    compressed_dataset=False, train_size=0.75, debug=False):

    model = AnsSelect(
        inp_dim = inp_dim,
        hid_dim = hid_dim,
        initialization = initialization,
        optimization = optimization
    )

    dataset = AnswerExtract.get_babi_dataset(compressed_dataset)

    if debug:
        dataset = dataset[:100]

    for epoch in tqdm(range(epochs), total=epochs, unit='epoch', desc='Epochs'):
        training_data, testing_data = train_test_split(
            dataset,
            train_size = train_size,
            shuffle = True
        )


        if compressed_dataset:
            train_compressed_dataset(model, training_data)
            y_true, y_pred = test_compressed_dataset(
                model,
                testing_data,
                threshold,
            )
        else:
            train_normal_dataset(model, training_data)
            y_true, y_pred = test_normal_dataset(
                model,
                testing_data,
                threshold,
            )

        score = f1_score(y_true, y_pred)
        print("Epoch: {} F1-score: {}".format(epoch, score))

        filename = utils.get_file_name(
            epoch_count = epoch,
            f1_score = score,
            initialization = initialization,
            optimization = optimization,
            inp_dim = inp_dim,
            hid_dim = hid_dim,
            threshold = threshold,
            extension = 'pkl'
        )
        model.save_params(filename, epochs=epoch)
    return
