
from tensorflow.keras import backend as K
from abstract_error_calc import AbstractErrorCalc

class RmseCalc(AbstractErrorCalc):

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        super().__str__()
        return "RMSE"


    def calc_error(self,y_true, y_pred):
        """
        A method to calculate root mean square error. (RMSE) punishes large errors 
        and results in a score that is in the same units as the forecast data,
        TODO test rmse = sqrt(mean_squared_error(test, predictions))
        https://www.kaggle.com/learn-forum/52081
        """
        error = K.sqrt(K.mean(K.square(y_pred - y_true)))
        return error