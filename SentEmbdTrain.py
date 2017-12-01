from Helpers import utils
import numpy as np
import os
import pickle
import sys
from Models import SentEmbd
import datetime
import time
import spacy

BASE = os.path.dirname(os.path.abspath(__file__))

def read_dataset():
    global glove
    global dep_tags
    global nlp
    nlp=spacy.load('en')
    #PreProcessing of Data before training our SentEmbd Model includes converting of words to their vector representation
    training_set=os.path.join(os.path.join(BASE,'data'),"SICK.txt")
    with open(training_set,'r') as file1:
        raw_dataset=file1.read().split('\n')
    file1.close()
    # print(raw_dataset) #TESTING PURPOSE
    dataset=[]
    training_dataset=[]
    sim_dataset=[]
    relatedness_scores=[]
    depTags_training_dataset=[]
    depTags_sim_dataset=[]
    raw_dataset=raw_dataset[1:-1]
    for item in raw_dataset:
        temp=item.split('\t')
        temp2=temp[4]
        temp=temp[1:3]
        temp.append(temp2.strip())
        dataset.append(temp)

    glove=utils.load_glove()
    dep_tags=utils.load_dep_tags()

    for item in dataset:

        vectorized_sent1,dep_tags_1=utils.get_sent_details(item[0].strip(),glove,dep_tags,nlp)
        vectorized_sent2,dep_tags_2=utils.get_sent_details(item[1].strip(),glove,dep_tags,nlp)

        training_dataset.append(vectorized_sent1)
        depTags_training_dataset.append(dep_tags_1)
        sim_dataset.append(vectorized_sent2)
        depTags_sim_dataset.append(dep_tags_2)

        relatedness_scores.append(float(item[2]))
    return dataset,training_dataset,sim_dataset,relatedness_scores,depTags_training_dataset,depTags_sim_dataset

def main():
    global glove
    global dep_tags
    global nlp
    n=int(sys.argv[1])
    hid_dim=int(sys.argv[3])
    dataset,training_dataset,sim_dataset,relatedness_scores,depTags_training_dataset,depTags_sim_dataset= read_dataset()
    # print(training_dataset[1].shape)
    
    batch_size=1 #By default
    epochs=int(sys.argv[2])
    choice = int(sys.argv[4])
    test = sys.argv[5]
    start=time.time()
    if(choice == 1):
        print("SentEmbd_Normal")
        sent_embd=SentEmbd.SentEmbd_basic(50,hid_dim) #GRU INITIALIZATION
        start = time.time()
        sent_embd.trainx(training_dataset[:n],sim_dataset[:n],relatedness_scores[:n],epochs) #Training THE GRU using the SICK dataset


    else:
        print("SentEmbd_syntactic")
        sent_embd=SentEmbd.SentEmbd_syntactic(50,hid_dim,len(dep_tags)) #GRU INITIALIZATION
        start = time.time()
        sent_embd.trainx(training_dataset[:n],sim_dataset[:n],relatedness_scores[:n],depTags_training_dataset[:n],depTags_sim_dataset[:n],epochs) #Training THE GRU using the SICK dataset
    

    print("Time taken for training:\t"+str(time.time()-start))

    z=str(datetime.datetime.now()).split(' ')

    #To Check for accuracy after training.
    acc="Nan"
    if test == 'test':
        file_name = "SentEmbd_"+str(epochs)+"_"+str(n)+"_"+str(hid_dim)+"_"+z[0]+"_"+z[1].split('.')[0]+".txt"
        path = os.path.join(os.path.join(os.path.join(BASE,'logs'),'SentEmbd'),file_name)
        if(choice == 1):
            accuracy=sent_embd.testing(training_dataset[n+1:],sim_dataset[n+1:],relatedness_scores[n+1:],path,choice)
        if(choice == 2):
            additional_inputs=[depTags_training_dataset[n+1:],depTags_sim_dataset[n+1:]]
            accuracy=sent_embd.testing(training_dataset[n+1:],sim_dataset[n+1:],relatedness_scores[n+1:],path,choice,additional_inputs)
        acc="{0:.3}".format(accuracy)
        acc+="%"

    # Saving the trained Model:
    
    # print(z)
    file_name="SentEmbd_"+str(epochs)+"_"+str(n)+"_"+str(hid_dim)+"_"+acc+"_"+z[0]+"_"+z[1].split('.')[0]+".pkl"
    path = os.path.join(os.path.join(os.path.join(BASE,'states'),'SentEmbd'),file_name)
    sent_embd.save_params(path,epochs)



    print("Current model params saved in- ",file_name)
    # if choice==1:
    #     print(sent_embd.predictx("Sample Sentence",glove))
    # elif choice == 2:
    #     print(sent_embd.predictx("Sample Sentence",glove,dep_tags,nlp))



if __name__ == '__main__':
    main()
