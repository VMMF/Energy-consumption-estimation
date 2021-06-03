from abstract_error_calc import AbstractErrorCalc
import numpy as np
import tensorflow as tf
from keras import backend as K

class MapeCalc(AbstractErrorCalc):

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        super().__str__()
        return "MAPE"


    def calc_error(self,y_true, y_pred):
        """
        A method to calculate Mean Absolute Percentage Error
        """
        
        y_true = K.maximum(y_true, 1e-7) # prevent errors multiplying by zero
        error = K.mean(K.abs((y_true - y_pred) / y_true)) * 100

        # error = tf.keras.losses.MAPE(y_true,y_pred)
        return error