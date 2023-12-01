import numpy as np
from keras.layers import Dense, Normalization
from keras import Input, Model
import keras.backend as K
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import cast, greater_equal
from tensorflow.python.framework.ops import disable_eager_execution
import src.utils.const as constants
from src.utils.collision import collisionCheck, euclidian_dist
import tensorflow as tf
from typing import Tuple
disable_eager_execution()
print(tf.__version__)

class NeuralClock:
    
    def __init__(self) -> None:
        with open("./src/data/training_data.npy", "rb") as train:
            self.training_data = np.load(train)
        with open("./src/data/target_data.npy", "rb") as target:
            self.target_data = np.load(target)

        self.full_data = np.hstack((self.training_data, self.target_data))
        self.scaler = StandardScaler()

    def make_model(self, training_data: np.ndarray) -> Model:
        mean, variance = self._get_mean_and_variance(training_data=training_data)
        normalizer = Normalization(mean=mean, variance=variance)
        input_layer = Input(shape=(4, ))
        normalized = normalizer(input_layer)
        dense_1 = Dense(units=12, kernel_initializer="he_normal", activation="relu")(normalized)
        dense_2 = Dense(units=12, kernel_initializer="he_normal", activation="relu")(dense_1)
        dense_3 = Dense(units=10, kernel_initializer="he_normal", activation="relu")(dense_2)
        dense_4 = Dense(units=10, kernel_initializer="he_normal", activation="relu")(dense_3)
        dense_5 = Dense(units=8, kernel_initializer="he_normal", activation="relu")(dense_4)
        dense_6 = Dense(units=6, kernel_initializer="he_normal", activation="relu")(dense_5)
        output_layer = Dense(units=2)(dense_6)
        model = Model(input_layer, output_layer)
        model.compile(loss=self.loss_wrapper(input_layer), optimizer="adam")
        return model

    @staticmethod
    def _get_mean_and_variance(training_data: np.ndarray) -> Tuple:
        mean = np.mean(training_data[:, 0:4], axis=0)
        variance = np.var(training_data[:, 0:4], axis=0)

        return mean, variance

    
    def splitter(self, data: np.ndarray, scale: bool=False) -> tuple:
        train, val_test = train_test_split(data, test_size=0.2, random_state=42)
        val, test = train_test_split(val_test, test_size=0.5, random_state=42)
        if scale:
            self.scaler.fit(X=train)
            train = self.scaler.transform(X=train)
            val = self.scaler.transform(X=val)
            test = self.scaler.transform(X=test)

        return train, val, test
    

    def normalize(self, data, fit: bool = False):
        if fit:
            self.scaler.fit(X=data)
        return self.scaler.transform(X=data)
    

    def loss_wrapper(self, input):
        def custom_loss(y_true, y_pred):
            input_tensor = input[:, 0:2]
            full_prediction = input_tensor + y_pred

            euclidian_distance = euclidian_dist(pos1=y_true, pos2=full_prediction)

            board_edge_penalty = cast(greater_equal(full_prediction[:, 0], constants.BOARD_RADIUS - constants.sun["MarkerRadius"]), dtype=np.float32)*10_000

            collision_penalty = cast(greater_equal(collisionCheck(input), 1), dtype=np.float32)*10_000
            
            return K.square(euclidian_distance) + board_edge_penalty + collision_penalty
        return custom_loss
