def preprocessSick():
    from Helpers.preprocess import SICK
    SICK.get_final_input()
    return

def train_dtrnn():
	from Model_Trainer.dtrnnTrain import DT_RNN_Train
	DT_RNN_Train(n=8000, epochs=5, hid_dim=50)

if __name__ == '__main__':
    train_dtrnn()


