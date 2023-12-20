from src.neural_net import NeuralClock
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

def train_model(data_filepath: str, model_filepath: str):

    with open(data_filepath, "rb") as file:
        data = np.load(file)

    training_data,  test_data = train_test_split(data, 0.2)
    print(f"Generated training data with shape {training_data.shape} and test data with shape {test_data.shape}")
    
    network = NeuralClock()

    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )

    model = network.make_model(training_data)

    model.fit(
        x=training_data[:, 0:10], 
        y=training_data[:, 10:12], 
        epochs=100, 
        verbose=1, 
        initial_epoch=0, 
        validation_split=0.2,
        callbacks=[model_checkpoint_callback])

    return model


