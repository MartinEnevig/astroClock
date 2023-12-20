import numpy as np
from keras.layers import Dense, Normalization
from keras import Input, Model
import keras.backend as K
from tensorflow import cast, greater_equal
from tensorflow.python.framework.ops import disable_eager_execution
import src.utils.const as constants
from src.utils.collision import is_collision, euclidian_dist
import tensorflow as tf
from typing import Tuple
disable_eager_execution()

class NeuralClock:
    
    def make_model(self, training_data: np.ndarray) -> Model:
        mean, variance = self._get_mean_and_variance(training_data=training_data)
        normalizer = Normalization(mean=mean, variance=variance)
        input_layer = Input(shape=(10, ))
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
        mean = np.mean(training_data[:, 0:10], axis=0)
        variance = np.var(training_data[:, 0:10], axis=0)

        return mean, variance

    def loss_wrapper(self, input):
        def custom_loss(y_true, y_pred):
            input_tensor = input[:, 0:2]
            full_prediction = input_tensor + y_pred

            euclidian_distance = euclidian_dist(pos1=y_true, pos2=full_prediction)

            board_edge_penalty = cast(greater_equal(full_prediction[:, 0], constants.BOARD_RADIUS - constants.sun["MarkerRadius"]), dtype=np.float32)*10_000

            collision_penalty = cast(greater_equal(is_collision(object_positions=input, end_position=full_prediction), 1), dtype=np.float32)*10_000
            
            return K.square(euclidian_distance) + board_edge_penalty + collision_penalty
        return custom_loss
    