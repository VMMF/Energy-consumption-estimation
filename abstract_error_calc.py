
from abc import ABC, abstractmethod

class AbstractErrorCalc(ABC):

    @abstractmethod
    def calc_error(y_true, y_pred):
        pass