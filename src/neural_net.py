import numpy as np
from keras.layers import Dense
from keras import Input, Model
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import cast, greater_equal
from tensorflow.python.framework.ops import disable_eager_execution
import astroClock.src.utils.const as constants

disable_eager_execution()


class NeuralClock_keras:
    
    def __init__(self) -> None:
        with open("./src/data/training_data.npy", "rb") as train:
            self.training_data = np.load(train)
        with open("./src/data/target_data.npy", "rb") as target:
            self.target_data = np.load(target)

        self.full_data = np.hstack((self.training_data, self.target_data))

    def make_model(self) -> Model:

        input_layer = Input(shape=(4, ))
        dense_1 = Dense(units=10, kernel_initializer="he_normal", activation="relu")(input_layer)
        dense_2 = Dense(units=10, kernel_initializer="he_normal", activation="relu")(dense_1)
        dense_3 = Dense(units=10, kernel_initializer="he_normal", activation="relu")(dense_2)
        dense_4 = Dense(units=10, kernel_initializer="he_normal", activation="relu")(dense_3)
        output_layer = Dense(units=2)(dense_4)
        model = Model(input_layer, output_layer)
        model.compile(loss=self.loss_wrapper(input_layer), optimizer="adam")
        return model


    def splitter(self, data: np.ndarray) -> tuple:
        train, val_test = train_test_split(data, test_size=0.2, random_state=42)
        val, test = train_test_split(val_test, test_size=0.5, random_state=42)

        scaler = StandardScaler()
        scaler.fit(X=train)
        train = scaler.transform(X=train)
        val = scaler.transform(X=val)
        test = scaler.transform(X=test)

        return train, val, test
    

    def normalize(self, data, fit: bool = False):
        scaler = StandardScaler()
        if fit:
            scaler.fit(X=data)
        return scaler.transform(X=data)

    @staticmethod
    def loss_wrapper(input):
        def custom_loss(y_true, y_pred):
            input_tensor = input[:, 0:2]
            full_prediction = input_tensor + y_pred
            y_true_cartesian = K.stack([y_true[:, 0]*K.cos(y_true[:, 1]), y_true[:, 0]*K.sin(y_true[:, 1])], axis=-1) 
            y_pred_cartesian = K.stack([full_prediction[:, 0]*K.cos(full_prediction[:, 1]), full_prediction[:, 0]*K.sin(full_prediction[:, 1])], axis=-1)
            
            euclidian_distance = K.sqrt(K.sum(K.square(y_pred_cartesian - y_true_cartesian), axis=-1))

            board_edge_penalty = cast(greater_equal(full_prediction[:, 0], constants.BOARD_RADIUS - constants.sun["MarkerRadius"]), dtype=np.float32)*1000

            return euclidian_distance + board_edge_penalty
        return custom_loss