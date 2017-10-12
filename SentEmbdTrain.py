import utils
import numpy as np
import os
import pickle
import sys
import SentEmbd
import datetime
def save_params(model,file_name,epochs):
    with open(file_name, 'wb') as save_file:
        pickle.dump(
            obj = {
                'params' : [x.get_value() for x in model.params],
                'epoch' : epochs,
            },
            file = save_file,
            protocol = -1
        )
        save_file.close()

def load_params(file_name,model):
    with open(file_name, 'rb') as load_file:
        dict = pickle.load(load_file)
        loaded_params = dict['params']
        for (x, y) in zip(model.params, loaded_params):
            x.set_value(y)
        load_file.close()

def read_dataset(n):
    #PreProcessing of Data before training our SentEmbd Model includes converting of words to their vector representation
    training_set=os.path.join(os.path.dirname(os.path.abspath(__file__)),"SICK.txt")
    with open(training_set,'r') as file1:
        raw_dataset=file1.read().split('\n')
    file1.close()
    # print(raw_dataset) #TESTING PURPOSE
    dataset=[]
    training_dataset=[]
    sim_dataset=[]
    relatedness_scores=[]
    raw_dataset=raw_dataset[1:-1]
    for item in raw_dataset:
        temp=item.split('\t')
        temp2=temp[4]
        temp=temp[1:3]
        temp.append(temp2)
        dataset.append(temp)

    glove=utils.load_glove()

    for item in dataset:
        sent1=item[0].split(' ')
        sent2=item[1].split(' ')
        sent_1=[]
        sent_2=[]
        for word in sent1:
            sent_1.append(utils.get_vector(word,glove))
        for word in sent2:
           sent_2.append(utils.get_vector(word,glove))
        training_dataset.append(sent_1)
        sim_dataset.append(sent_2)
        relatedness_scores.append(float(item[2]))
    return dataset,training_dataset,sim_dataset,relatedness_scores

def main():
    n=int(sys.argv[1])

    dataset,training_dataset,sim_dataset,relatedness_scores = read_dataset(n)
    sent_embd=SentEmbd.SentEmbd(50,len(dataset)) #GRU INITIALIZED
    batch_size=1 #By default
    epochs=int(sys.argv[2])

    sent_embd.trainx(training_dataset[:n],sim_dataset[:n],relatedness_scores[:n],epochs) #Training THE GRU using the SICK dataset

    #To Check for accuracy after training.
    ch=input('Do want to test the trained model on remaining sentences ? (y/n): ')
    if(ch is 'y'):
        sent_embd.testing(training_dataset[n+1:],sim_dataset[n+1:],relatedness_scores[n+1:])

    # Saving the trained Model:
    z=str(datetime.datetime.now()).split(' ')
    # print(z)
    file_name="SentEmbd_"+str(epochs)+"_"+str(n)+"_"+z[0]+"_"+z[1]
    save_params(sent_embd,file_name,epochs)

    print("Current model params saved in- ",file_name)



if __name__ == '__main__':
    main()
